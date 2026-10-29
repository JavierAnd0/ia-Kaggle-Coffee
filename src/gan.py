"""GAN (DCGAN condicional) para sintetizar hojas de la clase minoritaria. Entrenar SOLO con train."""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .config import SEED

LATENT, GAN_SIZE = 128, 64


def build_generator(num_classes: int) -> tf.keras.Model:
    z = layers.Input((LATENT,))
    c = layers.Input((1,), dtype="int32")
    ce = layers.Flatten()(layers.Embedding(num_classes, 32)(c))
    x = layers.Dense(8 * 8 * 256, use_bias=False)(layers.Concatenate()([z, ce]))
    x = layers.Reshape((8, 8, 256))(layers.LeakyReLU(0.2)(layers.BatchNormalization()(x)))
    for f in (128, 64, 32):  # 8 -> 16 -> 32 -> 64
        x = layers.Conv2DTranspose(f, 4, 2, "same", use_bias=False)(x)
        x = layers.LeakyReLU(0.2)(layers.BatchNormalization()(x))
    out = layers.Conv2D(3, 3, padding="same", activation="tanh")(x)
    return tf.keras.Model([z, c], out)


def build_discriminator(num_classes: int) -> tf.keras.Model:
    img = layers.Input((GAN_SIZE, GAN_SIZE, 3))
    c = layers.Input((1,), dtype="int32")
    ce = layers.Reshape((GAN_SIZE, GAN_SIZE, 1))(
        layers.Dense(GAN_SIZE * GAN_SIZE)(layers.Flatten()(layers.Embedding(num_classes, 16)(c))))
    x = layers.Concatenate()([img, ce])
    for f in (64, 128, 256):
        x = layers.Dropout(0.3)(layers.LeakyReLU(0.2)(layers.Conv2D(f, 4, 2, "same")(x)))
    return tf.keras.Model([img, c], layers.Dense(1)(layers.Flatten()(x)))


class CGAN:
    def __init__(self, num_classes: int):
        self.n, self.g, self.d = num_classes, build_generator(num_classes), build_discriminator(num_classes)
        self.g_opt = tf.keras.optimizers.Adam(2e-4, 0.5)
        self.d_opt = tf.keras.optimizers.Adam(2e-4, 0.5)
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def _step(self, real, labels):
        z = tf.random.normal((tf.shape(real)[0], LATENT))
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = self.g([z, labels], training=True)
            r, f = self.d([real, labels], training=True), self.d([fake, labels], training=True)
            d_loss = self.bce(tf.ones_like(r) * 0.9, r) + self.bce(tf.zeros_like(f), f)
            g_loss = self.bce(tf.ones_like(f), f)
        self.d_opt.apply_gradients(zip(dt.gradient(d_loss, self.d.trainable_variables), self.d.trainable_variables))
        self.g_opt.apply_gradients(zip(gt.gradient(g_loss, self.g.trainable_variables), self.g.trainable_variables))
        return d_loss, g_loss

    def fit(self, images_uint8: np.ndarray, labels: np.ndarray, epochs=200, batch=32):
        """images_uint8: (N,64,64,3) de TRAIN; se escalan a [-1,1]."""
        x = images_uint8.astype("float32") / 127.5 - 1
        ds = tf.data.Dataset.from_tensor_slices((x, labels.reshape(-1, 1))).shuffle(len(x), seed=SEED).batch(batch)
        for ep in range(epochs):
            for xb, yb in ds:
                d, g = self._step(xb, yb)
            if ep % 20 == 0:
                print(f"epoch {ep}: d={float(d):.3f} g={float(g):.3f}")

    def sample(self, class_id: int, n: int) -> np.ndarray:
        z = tf.random.normal((n, LATENT))
        img = self.g([z, tf.fill((n, 1), tf.constant(class_id, tf.int32))], training=False).numpy()
        return ((img + 1) * 127.5).clip(0, 255).astype("uint8")


def load_images(df, size: int = GAN_SIZE) -> np.ndarray:
    """Carga las imágenes de `df` (usar solo TRAIN) a `size`x`size` uint8 para entrenar la GAN."""
    return np.stack([tf.keras.utils.img_to_array(tf.keras.utils.load_img(p, target_size=(size, size)))
                     for p in df["filepath"]]).astype("uint8")


def make_synthetic_df(gan: "CGAN", class_name: str, class_id: int, n: int, out_dir, out_size: int = 224):
    """Genera `n` imágenes de la clase, las guarda como PNG (reescaladas) y devuelve el DataFrame
    [filepath, label, group] para añadirlo SOLO al conjunto de entrenamiento."""
    import pandas as pd
    from pathlib import Path
    from PIL import Image
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, img in enumerate(gan.sample(class_id, n)):
        path = out / f"synth_{class_name}_{i:04d}.png"
        Image.fromarray(img).resize((out_size, out_size), Image.BICUBIC).save(path)
        rows.append({"filepath": str(path), "label": class_name, "group": f"synth_{i}"})
    return pd.DataFrame(rows)
