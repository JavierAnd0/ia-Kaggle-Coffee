"""Modelo: transfer learning con backbone congelado + cabeza densa; build_model(hp) para Hyperband."""
import tensorflow as tf

from . import config

BACKBONES = {
    "mobilenetv2": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "efficientnetb0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input),
}


def build_model(hp, num_classes: int, img_size=None) -> tf.keras.Model:
    size = tuple(img_size or config.IMG_SIZE)
    name = hp.Choice("backbone", list(BACKBONES))
    base_cls, _ = BACKBONES[name]
    base = base_cls(include_top=False, weights="imagenet" if config.PRETRAINED else None, input_shape=size + (3,))
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


# --------------------------------------------------------------------------------------------------
# Versión 5: backbone más fuerte con el preprocesamiento DENTRO del modelo.
# Contrato con la web: entrada NHWC float32 [1, L, L, 3] en [-1, 1]; salida softmax de 3 clases.
# Rescaling(127.5, offset=127.5) lleva [-1, 1] a [0, 255] y luego se aplica la normalización de ImageNet
# que esperan los pesos del backbone; así la web no necesita saber qué backbone se usó.
# --------------------------------------------------------------------------------------------------
BACKBONES_V5 = {
    "efficientnetv2b0": lambda: tf.keras.applications.EfficientNetV2B0,
    "convnext_tiny": lambda: tf.keras.applications.ConvNeXtTiny,
}


@tf.keras.utils.register_keras_serializable(package="cafe")
class NormalizacionImagenet(tf.keras.layers.Layer):
    """(x - 255·media) / (255·desv) sobre [0, 255], con media/desv de ImageNet: es EXACTAMENTE lo que hace
    `include_preprocessing=True` en EfficientNetV2-B* y ConvNeXt. Se reimplementa con constantes dentro del
    grafo porque la capa Normalization de Keras 3 las captura como tensores externos y el ONNX exportado
    quedaba con entradas extra."""

    MEDIA = (0.485, 0.456, 0.406)
    DESV = (0.229, 0.224, 0.225)

    def call(self, x):
        media = tf.constant([255.0 * m for m in self.MEDIA], dtype=x.dtype)
        desv = tf.constant([255.0 * d for d in self.DESV], dtype=x.dtype)
        return (x - media) / desv


def build_model_v5(backbone: str, num_classes: int, lado: int, dropout: float = 0.3,
                   pretrained: bool | None = None, name: str | None = None) -> tf.keras.Model:
    """Modelo v5. El backbone se llama con training=False: sus BatchNorm quedan SIEMPRE en modo inferencia,
    también cuando se descongela para el ajuste fino (recomendación de Keras para transfer learning)."""
    pretrained = config.PRETRAINED if pretrained is None else pretrained
    base = BACKBONES_V5[backbone]()(include_top=False, weights="imagenet" if pretrained else None,
                                     input_shape=(lado, lado, 3), include_preprocessing=False)
    base.trainable = False
    entrada = tf.keras.Input((lado, lado, 3), name="input")
    x = tf.keras.layers.Rescaling(127.5, offset=127.5, name="a_0_255")(entrada)    # [-1, 1] -> [0, 255]
    x = NormalizacionImagenet(name="normalizacion_imagenet")(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    salida = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32", name="probabilidades")(x)
    return tf.keras.Model(entrada, salida, name=name or f"cafe_{backbone}")


def backbone_de(model: tf.keras.Model) -> tf.keras.Model:
    return next(l for l in model.layers if isinstance(l, tf.keras.Model))


def ensamblar_v5(modelos: list, lado: int) -> tf.keras.Model:
    """Un solo grafo que promedia las probabilidades de los miembros (se exporta como un único ONNX).
    Cada miembro debe tener nombre único (se fija al construirlo: Keras 3 no permite reasignar `model.name`)."""
    entrada = tf.keras.Input((lado, lado, 3), name="input")
    salidas = [m(entrada, training=False) for m in modelos]
    return tf.keras.Model(entrada, tf.keras.layers.Average(name="promedio")(salidas), name="ensamble")
