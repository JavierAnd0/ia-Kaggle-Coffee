"""Receta v5: EfficientNetV2-B0 (o ConvNeXt-Tiny) en dos fases, pesos de clase + label smoothing, selección
por F1 macro de validación. Una sola técnica contra el desbalance (pesos de clase "balanced" calculados con el
train v5; SIN sobremuestreo). La receta antigua (src/train.py, 4b) queda intacta para reproducir el informe.
"""
import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from .generators import make_dataset_v5
from .model import backbone_de, build_model_v5

ETIQUETAS_WEB = ["healthy", "red_spider_mite", "rust"]   # orden que espera web/model/labels.json


@dataclass
class RecetaV5:
    backbone: str = "efficientnetv2b0"
    lado: int = 224
    batch_size: int = 32
    dropout: float = 0.3
    label_smoothing: float = 0.1
    epocas_cabeza: int = 8
    lr_cabeza: float = 1e-3
    epocas_ft: int = 30
    lr_ft: float = 1e-4
    weight_decay: float = 0.05
    paciencia_ft: int = 10
    precision_mixta: bool = True
    pretrained: bool = True
    notas: dict = field(default_factory=dict)

    def como_dict(self) -> dict:
        d = asdict(self)
        d.pop("notas")
        return d


# ------------------------------------------------------------------ determinismo
def fijar_determinismo(semilla: int, determinista: bool = True) -> bool:
    """Semillas de Python/NumPy/TF y, si se pide, operaciones deterministas de TF. Devuelve si quedó activo."""
    tf.keras.utils.set_random_seed(semilla)
    if not determinista:
        _desactivar_determinismo()
        return False
    try:
        tf.config.experimental.enable_op_determinism()
        return True
    except Exception as e:                       # versiones/ops sin soporte: se sigue sin determinismo
        print("No se pudo activar el determinismo de TF:", e)
        return False


def _desactivar_determinismo():
    try:
        from tensorflow.python.framework import config as tfc
        if tfc.is_op_determinism_enabled():
            tfc.disable_op_determinism()
    except Exception:
        pass


# ------------------------------------------------------------------ métricas
def class_weights(df: pd.DataFrame, li: dict) -> dict:
    y = df["label"].map(li).values
    w = compute_class_weight("balanced", classes=np.arange(len(li)), y=y)
    return {i: float(v) for i, v in enumerate(w)}


def metricas(probs: np.ndarray, y: np.ndarray, li: dict) -> dict:
    pred = probs.argmax(1)
    f1 = f1_score(y, pred, average=None, labels=range(len(li)), zero_division=0)
    out = {"accuracy": float((pred == y).mean()), "f1_macro": float(f1.mean())}
    out.update({f"f1_{n}": float(f1[i]) for n, i in li.items()})
    return out


def predecir(model, ds, tta_flip: bool = False) -> np.ndarray:
    """Probabilidades sobre un dataset de evaluación; con tta_flip=True usa la imagen volteada horizontalmente."""
    if tta_flip:
        ds = ds.map(lambda x, y: (tf.image.flip_left_right(x), y))
    return model.predict(ds, verbose=0)


class F1MacroVal(tf.keras.callbacks.Callback):
    """Calcula el F1 macro de validación cada época (logs['val_f1_macro']), guarda en memoria los mejores pesos,
    detiene temprano por paciencia y al final restaura los mejores. `mejor_inicial`/`pesos_iniciales` permiten
    encadenar fases: si la fase 2 nunca supera a la 1, se restauran los pesos de la fase 1."""

    def __init__(self, ds_val, y_val, n_clases: int, paciencia: int | None = None,
                 mejor_inicial: float = -1.0, pesos_iniciales=None, verbose: int = 1):
        super().__init__()
        self.ds_val, self.y_val, self.n = ds_val, y_val, n_clases
        self.paciencia, self.verbose = paciencia, verbose
        self.mejor, self.mejores_pesos = mejor_inicial, pesos_iniciales
        self.epoca_mejor, self.historia = -1, []

    def on_train_begin(self, logs=None):
        self.espera = 0

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.ds_val, verbose=0).argmax(1)
        f1 = float(f1_score(self.y_val, pred, average="macro", labels=range(self.n), zero_division=0))
        if logs is not None:
            logs["val_f1_macro"] = f1
        self.historia.append(f1)
        if f1 > self.mejor:
            self.mejor, self.mejores_pesos, self.epoca_mejor, self.espera = f1, self.model.get_weights(), epoch, 0
        else:
            self.espera += 1
        if self.verbose:
            print(f"  época {epoch + 1}: val F1 macro {f1:.4f} (mejor {self.mejor:.4f})")
        if self.paciencia is not None and self.espera >= self.paciencia:
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.mejores_pesos is not None:
            self.model.set_weights(self.mejores_pesos)


# ------------------------------------------------------------------ entrenamiento
def _compilar(model, optimizador, label_smoothing):
    model.compile(optimizador, tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
                  metrics=["accuracy"])


def entrenar_semilla(receta: RecetaV5, train_df, val_df, li: dict, semilla: int, determinista: bool = True,
                     verbose: int = 1, nombre: str | None = None):
    """Entrena un modelo (fase 1: solo cabeza; fase 2: todo el backbone, BatchNorm en inferencia).
    Devuelve (modelo con los mejores pesos por F1 macro de validación, info)."""
    tf.keras.backend.clear_session()
    det = fijar_determinismo(semilla, determinista)
    if receta.precision_mixta and tf.config.list_physical_devices("GPU"):
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    else:
        tf.keras.mixed_precision.set_global_policy("float32")
    t0 = time.time()
    n = len(li)
    tr = make_dataset_v5(train_df, li, receta.lado, receta.batch_size, training=True, seed=semilla)
    va = make_dataset_v5(val_df, li, receta.lado, receta.batch_size)
    y_val = val_df["label"].map(li).values
    cw = class_weights(train_df, li)
    model = build_model_v5(receta.backbone, n, receta.lado, receta.dropout, receta.pretrained, name=nombre)

    # Fase 1: solo la cabeza
    _compilar(model, tf.keras.optimizers.Adam(receta.lr_cabeza), receta.label_smoothing)
    cb1 = F1MacroVal(va, y_val, n, paciencia=None, verbose=verbose)
    model.fit(tr, epochs=receta.epocas_cabeza, class_weight=cw, callbacks=[cb1], verbose=0)

    # Fase 2: todo el backbone, AdamW con decaimiento coseno (y 1 época de calentamiento)
    cb2 = F1MacroVal(va, y_val, n, paciencia=receta.paciencia_ft, mejor_inicial=cb1.mejor,
                     pesos_iniciales=cb1.mejores_pesos, verbose=verbose)
    if receta.epocas_ft > 0:
        backbone_de(model).trainable = True
        pasos = int(np.ceil(len(train_df) / receta.batch_size))
        calent = min(pasos, max(1, pasos * receta.epocas_ft // 10))
        lr = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=receta.lr_ft * 0.1, decay_steps=max(1, pasos * receta.epocas_ft - calent),
            warmup_target=receta.lr_ft, warmup_steps=calent)
        _compilar(model, tf.keras.optimizers.AdamW(lr, weight_decay=receta.weight_decay), receta.label_smoothing)
        model.fit(tr, epochs=receta.epocas_ft, class_weight=cw, callbacks=[cb2], verbose=0)
    info = {"semilla": semilla, "determinista": det, "segundos": round(time.time() - t0),
            "f1_val_fase1": round(cb1.mejor, 4), "f1_val_final": round(cb2.mejor, 4),
            "epoca_mejor_ft": cb2.epoca_mejor + 1 if cb2.epoca_mejor >= 0 else 0,
            "epocas_ft_corridas": len(cb2.historia), "pesos_clase": cw}
    return model, info


def guardar_json(ruta, datos):
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2))
