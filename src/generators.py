"""Data generators: cargan las imágenes lote a lote (tf.data) para no llenar la RAM."""
import numpy as np
import pandas as pd
import tensorflow as tf

from . import config
from .config import BATCH_SIZE, IMG_SIZE, SEED


def make_label_index(labels) -> dict:
    return {c: i for i, c in enumerate(sorted(set(labels)))}


def _augmenter() -> tf.keras.Sequential:
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
    ])


def make_dataset(df: pd.DataFrame, label_index: dict, batch_size: int = BATCH_SIZE,
                 training: bool = False, augment: bool = False,
                 preprocess=tf.keras.applications.mobilenet_v2.preprocess_input, img_size=None):
    """tf.data pipeline: rutas -> decode -> resize -> (augment) -> preprocess -> batch.

    Los pesos de normalización/augmentation NO se ajustan con val/test (sin leakage);
    `training=True` solo activa shuffle y `augment=True` solo debe usarse en train.
    """
    size = tuple(img_size or config.IMG_SIZE)
    paths = df["filepath"].values
    y = df["label"].map(label_index).values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, size)
        return img, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if augment:
        aug = _augmenter()
        ds = ds.map(lambda x, l: (aug(x, training=True), l), num_parallel_calls=tf.data.AUTOTUNE)
    if preprocess is not None:
        ds = ds.map(lambda x, l: (preprocess(x), l), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


class ImageSequence(tf.keras.utils.Sequence):
    """Equivalente clásico con keras Sequence (ilustra el concepto de generator)."""

    def __init__(self, df, label_index, batch_size=BATCH_SIZE, shuffle=False):
        super().__init__()
        self.df, self.li, self.bs, self.shuffle = df.reset_index(drop=True), label_index, batch_size, shuffle
        self.idx = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.bs))

    def __getitem__(self, i):
        rows = self.df.iloc[self.idx[i * self.bs:(i + 1) * self.bs]]
        x = np.stack([tf.keras.utils.img_to_array(
            tf.keras.utils.load_img(p, target_size=IMG_SIZE)) for p in rows["filepath"]])
        y = rows["label"].map(self.li).values
        return tf.keras.applications.mobilenet_v2.preprocess_input(x), y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.default_rng(SEED).shuffle(self.idx)


# --------------------------------------------------------------------------------------------------
# Versión 5: el modelo recibe la imagen en [-1, 1] (contrato de la web) y hace su propio reescalado.
# Evaluación/servicio: decodificar -> tf.image.resize bilineal a LxL (igual que web/src/lib/predictor/
# preprocesar.ts) -> x/127.5 - 1. El aumento solo se usa en train y no cambia la proporción de aspecto
# respecto al servicio: el recorte toma la MISMA fracción del ancho y del alto antes de redimensionar.
# --------------------------------------------------------------------------------------------------
def a_rango_modelo(img_0_255):
    """0-255 -> [-1, 1] (equivale a x/127.5 - 1)."""
    return tf.cast(img_0_255, tf.float32) / 127.5 - 1.0


def _decodificar(path):
    return tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)


def _rotar(img, angulo):
    """Rotación alrededor del centro con relleno por reflexión (img: HxWx3 float)."""
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    c, s = tf.cos(angulo), tf.sin(angulo)
    x0 = (w - 1) / 2 - (c * (w - 1) / 2 - s * (h - 1) / 2)
    y0 = (h - 1) / 2 - (s * (w - 1) / 2 + c * (h - 1) / 2)
    transform = tf.stack([c, -s, x0, s, c, y0, 0.0, 0.0])[None]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None], transforms=transform, output_shape=tf.shape(img)[:2], fill_value=0.0,
        interpolation="BILINEAR", fill_mode="REFLECT")
    return out[0]


def aumentar_v5(path, semilla, lado: int, escala_min: float = 0.65, rot_max: float = 0.35):
    """Aumento fuerte pero seguro para la proporción de aspecto. Devuelve la imagen en [-1, 1].

    `semilla`: tensor int64 [2] (aumento sin estado y reproducible). Pasos: recorte aleatorio (misma
    fracción del ancho y del alto, 65-100 %), redimensionado bilineal a LxL, volteos horizontal/vertical,
    rotación (±20°), brillo/contraste/saturación y un leve cambio de tono.
    """
    s = tf.random.experimental.stateless_split(tf.cast(semilla, tf.int64), num=9)
    img = _decodificar(path)
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)
    frac = tf.random.stateless_uniform([], s[0], escala_min, 1.0)
    ch, cw = tf.cast(h * frac, tf.int32), tf.cast(w * frac, tf.int32)
    oy = tf.cast(tf.random.stateless_uniform([], s[1], 0.0, 1.0) * tf.cast(tf.shape(img)[0] - ch + 1, tf.float32), tf.int32)
    ox = tf.cast(tf.random.stateless_uniform([], s[2], 0.0, 1.0) * tf.cast(tf.shape(img)[1] - cw + 1, tf.float32), tf.int32)
    oy = tf.minimum(oy, tf.shape(img)[0] - ch)
    ox = tf.minimum(ox, tf.shape(img)[1] - cw)
    img = tf.image.crop_to_bounding_box(img, oy, ox, ch, cw)
    img = tf.image.resize(img, (lado, lado)) / 255.0                       # bilineal, como en el servicio
    img = tf.image.stateless_random_flip_left_right(img, s[3])
    img = tf.image.stateless_random_flip_up_down(img, s[4])
    img = _rotar(img, tf.random.stateless_uniform([], s[5], -rot_max, rot_max))
    img = tf.image.stateless_random_brightness(img, 0.15, s[6])
    img = tf.image.stateless_random_contrast(img, 0.75, 1.25, s[7])
    img = tf.image.stateless_random_saturation(img, 0.75, 1.25, s[8])
    img = tf.clip_by_value(img, 0.0, 1.0)
    img = tf.ensure_shape(img, (lado, lado, 3))
    return img * 2.0 - 1.0


def make_dataset_v5(df: pd.DataFrame, label_index: dict, lado: int, batch_size: int = BATCH_SIZE,
                    training: bool = False, seed: int = SEED, cache: bool = True, one_hot: bool = True):
    """tf.data para la receta v5. Etiquetas one-hot (para label smoothing). Entrada al modelo en [-1, 1].

    training=True: barajado + aumento reproducible (semillas sin estado derivadas de `seed`).
    training=False: solo el preprocesamiento de servicio; se guarda en memoria (val/test son pequeños).
    """
    paths = df["filepath"].astype(str).values
    y = df["label"].map(label_index).values.astype("int32")
    n_clases = len(label_index)
    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    def etiqueta(l):
        return tf.one_hot(l, n_clases) if one_hot else l

    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
        semillas = tf.data.Dataset.random(seed=seed, rerandomize_each_iteration=True).batch(2)
        ds = tf.data.Dataset.zip((ds, semillas))
        ds = ds.map(lambda pl, sd: (aumentar_v5(pl[0], sd, lado), etiqueta(pl[1])),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size)
    else:
        ds = ds.map(lambda p, l: (a_rango_modelo(tf.image.resize(_decodificar(p), (lado, lado))), etiqueta(l)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        if cache:
            ds = ds.cache()
        ds = ds.batch(batch_size)
    return ds.prefetch(tf.data.AUTOTUNE)
