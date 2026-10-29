"""Modelo: transfer learning con backbone congelado + cabeza densa; build_model(hp) para Hyperband."""
import tensorflow as tf

from . import config
from .config import IMG_SIZE

BACKBONES = {
    "mobilenetv2": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
}


def build_model(hp, num_classes: int) -> tf.keras.Model:
    name = hp.Choice("backbone", list(BACKBONES))
    base_cls, _ = BACKBONES[name]
    base = base_cls(include_top=False, weights="imagenet" if config.PRETRAINED else None, input_shape=IMG_SIZE + (3,))
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1))(x)
    x = tf.keras.layers.Dense(hp.Choice("units", [64, 128, 256]), activation="relu")(x)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(base.input, out)
    lr = hp.Float("learning_rate", 1e-5, 1e-2, sampling="log")
    opt_name = hp.Choice("optimizer", ["adam", "rmsprop"])
    opt = tf.keras.optimizers.Adam(lr) if opt_name == "adam" else tf.keras.optimizers.RMSprop(lr)
    model.compile(opt, "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def unfreeze_top(model: tf.keras.Model, n_layers: int = 30, lr: float = 1e-5) -> tf.keras.Model:
    """Fine-tuning: descongela las últimas `n_layers` del backbone (sin BatchNorm) y recompila."""
    head_start = next(i for i, l in enumerate(model.layers)
                      if isinstance(l, tf.keras.layers.GlobalAveragePooling2D))
    for layer in model.layers[max(0, head_start - n_layers):head_start]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    model.compile(tf.keras.optimizers.Adam(lr), "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
