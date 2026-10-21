"""Data generators: cargan las imágenes lote a lote (tf.data) para no llenar la RAM."""
import numpy as np
import pandas as pd
import tensorflow as tf

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
                 preprocess=tf.keras.applications.mobilenet_v2.preprocess_input):
    """tf.data pipeline: rutas -> decode -> resize -> (augment) -> preprocess -> batch.

    Los pesos de normalización/augmentation NO se ajustan con val/test (sin leakage);
    `training=True` solo activa shuffle y `augment=True` solo debe usarse en train.
    """
    paths = df["filepath"].values
    y = df["label"].map(label_index).values.astype("int32")
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if training:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
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
