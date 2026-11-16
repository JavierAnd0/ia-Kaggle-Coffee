"""Experimentos para mejorar el modelo: resolución de entrada × semillas, y conjunto (ensemble) de modelos.

Motivo: la misma receta dio 0,69 y 0,60 de exactitud en prueba (variabilidad por semilla) y las fotos
(1920×1080) se reducen a 224×224, donde las manchas pequeñas de la roya casi desaparecen. Aquí se mide
el efecto de la resolución con VARIAS semillas (media ± desviación) usando solo VALIDACIÓN; el test no se toca.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

from . import config
from .generators import make_dataset
from .model import BACKBONES
from .train import train_model


def predecir(model, hp, df, li, size: int) -> np.ndarray:
    pre = BACKBONES[hp.values["backbone"]][1]
    ds = make_dataset(df, li, hp.values.get("batch_size", 32), preprocess=pre, img_size=(size, size))
    return model.predict(ds, verbose=0)


def metricas(probs: np.ndarray, df, li: dict) -> dict:
    y = df["label"].map(li).values
    pred = probs.argmax(1)
    f1 = f1_score(y, pred, average=None, labels=range(len(li)), zero_division=0)
    out = {"accuracy": float((pred == y).mean()), "f1_macro": float(f1.mean())}
    out.update({f"f1_{n}": float(f1[i]) for n, i in li.items()})
    return out


def ensamblar(modelos: list, size: int) -> tf.keras.Model:
    """Un solo modelo Keras que promedia las probabilidades de varios (se exporta como un único ONNX)."""
    entrada = tf.keras.Input((size, size, 3))
    salidas = []
    for i, m in enumerate(modelos):
        m.name = f"miembro_{i}"
        salidas.append(m(entrada))
    return tf.keras.Model(entrada, tf.keras.layers.Average()(salidas), name="ensamble")


def carpeta_checkpoints(dir_modelos=None) -> Path:
    """Dónde se guardan modelos y resultados parciales. En Colab conviene que sea una carpeta de Google Drive
    (variable de entorno CHECKPOINT_DIR o argumento) para que una desconexión no pierda lo ya entrenado."""
    return Path(dir_modelos or os.environ.get("CHECKPOINT_DIR") or config.ROOT / "checkpoints")


def correr_rejilla(hp, train_df, val_df, li, tamanos=(224, 320, 448), semillas=(0, 1, 2), epocas=30,
                   epocas_ft=10, n_extra=150, dir_modelos=None, verbose=0):
    """Receta 4b (sobremuestreo real + pesos de clase + ajuste fino) para cada tamaño y semilla.

    REANUDABLE: cada corrida terminada deja en `dir_modelos` su modelo (.keras), sus probabilidades de
    validación (.npy) y su fila de resultados (.json). Si se vuelve a ejecutar tras una desconexión o un
    error, las corridas completas se saltan y solo se entrenan las que faltan.
    Devuelve (tabla por corrida, tabla resumen por tamaño).
    """
    dir_modelos = carpeta_checkpoints(dir_modelos)
    dir_modelos.mkdir(parents=True, exist_ok=True)
    minoritaria = train_df["label"].value_counts().idxmin()
    total = len(tamanos) * len(semillas)
    filas, por_tamano, hechas = [], {}, 0
    for tam in tamanos:
        probs_tam = []
        for semilla in semillas:
            ruta_m = dir_modelos / f"m_{tam}_s{semilla}.keras"
            ruta_p = dir_modelos / f"p_val_{tam}_s{semilla}.npy"
            ruta_f = dir_modelos / f"fila_{tam}_s{semilla}.json"
            hechas += 1
            if ruta_m.exists() and ruta_p.exists() and ruta_f.exists():       # corrida completa: se reutiliza
                p, fila = np.load(ruta_p), json.loads(ruta_f.read_text())
                print(f"[{hechas}/{total}] tamaño {tam} semilla {semilla}: ya estaba hecha (F1 macro {fila['f1_macro']:.3f}), se reutiliza")
            else:
                extra = train_df[train_df["label"] == minoritaria].sample(n_extra, replace=True, random_state=semilla)
                tr = pd.concat([train_df, extra], ignore_index=True)
                t0 = time.time()
                m = train_model(hp, tr, val_df, li, epocas, True, epocas_ft, verbose, weight_df=train_df,
                                img_size=(tam, tam), seed=semilla)
                p = predecir(m, hp, val_df, li, tam)
                m.save(ruta_m)
                np.save(ruta_p, p)
                fila = {"tamano": tam, "semilla": semilla, "segundos": round(time.time() - t0), **metricas(p, val_df, li)}
                ruta_f.write_text(json.dumps(fila))                          # al final: su existencia marca "completa"
                print(f"[{hechas}/{total}] tamaño {tam} semilla {semilla}: F1 macro {fila['f1_macro']:.3f}  "
                      f"exactitud {fila['accuracy']:.3f}  ({fila['segundos'] // 60} min)")
                tf.keras.backend.clear_session()
            probs_tam.append(p)
            filas.append(fila)
        por_tamano[tam] = probs_tam
    corridas = pd.DataFrame(filas)
    resumen = corridas.groupby("tamano").agg(
        f1_media=("f1_macro", "mean"), f1_desv=("f1_macro", "std"),
        acc_media=("accuracy", "mean"), acc_desv=("accuracy", "std"), seg_media=("segundos", "mean"))
    resumen["f1_conjunto"] = [metricas(np.mean(por_tamano[t], axis=0), val_df, li)["f1_macro"] for t in resumen.index]
    resumen["acc_conjunto"] = [metricas(np.mean(por_tamano[t], axis=0), val_df, li)["accuracy"] for t in resumen.index]
    return corridas, resumen


def finalizar(hp, train_df, val_df, test_df, li, resumen, semillas, dir_modelos=None, out_dir=None, tamano_final=None):
    """Arma el conjunto de modelos del mejor tamaño, lo evalúa UNA vez en test y lo exporta para la web.

    Devuelve un diccionario con el tamaño elegido, probabilidades, métricas, carpeta de salida y versión.
    """
    from datetime import date

    dir_modelos = carpeta_checkpoints(dir_modelos)
    out_dir = Path(out_dir or config.ROOT / "web" / "model")
    mejor = int(resumen["f1_conjunto"].idxmax()) if tamano_final is None else int(tamano_final)
    modelos = [tf.keras.models.load_model(dir_modelos / f"m_{mejor}_s{s}.keras") for s in semillas]
    ens = ensamblar(modelos, mejor)
    p_val = predecir(ens, hp, val_df, li, mejor)
    p_test = predecir(ens, hp, test_df, li, mejor)
    y_val, y_test = val_df["label"].map(li).values, test_df["label"].map(li).values
    res = {"val": metricas(p_val, val_df, li), "test": metricas(p_test, test_df, li)}
    version = f"ens{len(semillas)}-{mejor}px-{date.today().isoformat()}"
    from . import export

    out = export.export_model(ens, li, hp, res, out_dir, version=version, input_size=mejor,
                              notas={"arquitectura": f"conjunto de {len(semillas)} modelos MobileNetV2 (semillas {list(semillas)})"})
    onnx_path = export.export_onnx(ens, out)
    dif = export.verificar_onnx(ens, onnx_path)
    (out / "paridad.json").write_text(json.dumps(export.casos_paridad(ens, hp, li, test_df), indent=2))
    web = resumen_para_web(p_val, y_val, p_test, y_test, li, version)
    (out / "metricas_web.json").write_text(json.dumps(web, ensure_ascii=False, indent=2))
    return {"mejor": mejor, "p_val": p_val, "p_test": p_test, "y_val": y_val, "y_test": y_test,
            "metricas": res, "out": out, "version": version, "diferencia_onnx": dif}


def simular_combinacion(probs: np.ndarray, y: np.ndarray, ks=(1, 2, 3), intentos: int = 4000, semilla: int = 0) -> dict:
    """Acierto al promediar k hojas de la MISMA clase real elegidas al azar (simulación; no son plantas reales)."""
    azar = np.random.default_rng(semilla)
    clases = np.unique(y)
    salida = {}
    for k in ks:
        aciertos = 0
        for _ in range(intentos):
            c = azar.choice(clases)
            idx = azar.choice(np.flatnonzero(y == c), size=k, replace=False)
            aciertos += int(probs[idx].mean(0).argmax() == c)
        salida[str(k)] = round(aciertos / intentos, 3)
    return salida


def resumen_para_web(p_val, y_val, p_test, y_test, li: dict, version: str) -> dict:
    """Cifras que muestra la web (misma estructura que web/src/data/metricas.json) calculadas con el modelo exportado."""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    nombres = sorted(li, key=li.get)
    pred = p_test.argmax(1)
    pr, rc, f1, soporte = precision_recall_fscore_support(y_test, pred, labels=range(len(li)), zero_division=0)
    matriz = confusion_matrix(y_test, pred, labels=range(len(li)))
    niveles = {}
    for nombre, lo, hi in (("alta", 0.8, 1.01), ("media", 0.6, 0.8), ("baja", 0.0, 0.6)):
        fila = {}
        for etiqueta, p, y in (("validacion", p_val, y_val), ("prueba", p_test, y_test)):
            m = (p.max(1) >= lo) & (p.max(1) < hi)
            fila[etiqueta] = round(float((p.argmax(1)[m] == y[m]).mean()), 2) if m.any() else None
            if etiqueta == "prueba":
                fila["fotos_prueba"] = int(m.sum())
        niveles[nombre] = fila
    return {
        "prueba": {
            "imagenes": int(len(y_test)), "exactitud": round(float((pred == y_test).mean()), 2),
            "f1_macro": round(float(f1.mean()), 2),
            "por_clase": {n: {"precision": round(float(pr[i]), 2), "recall": round(float(rc[i]), 2),
                              "f1": round(float(f1[i]), 2), "soporte": int(soporte[i])} for i, n in enumerate(nombres)},
            "matriz": {n: [int(v) for v in matriz[i]] for i, n in enumerate(nombres)},
            "orden_matriz": nombres,
        },
        "validacion_modelo": {"imagenes": int(len(y_val)), "exactitud": round(float((p_val.argmax(1) == y_val).mean()), 2),
                              "f1_macro": round(float(metricas(p_val, _FakeDf(y_val, li), li)["f1_macro"]), 2)},
        "niveles": niveles,
        "combinacion_simulada": {"prueba": simular_combinacion(p_test, y_test), "validacion": simular_combinacion(p_val, y_val),
                                 "nota": "Simulación con las probabilidades reales del modelo: se eligen al azar 1, 2 o 3 fotos de la misma clase y se promedian. "
                                         "Supone que todas las hojas de la planta tienen el mismo problema; no es una medición con plantas reales."},
        "nota": f"Cifras del modelo exportado (versión {version}), calculadas con sus probabilidades reales.",
    }


class _FakeDf:
    """Adaptador mínimo para reutilizar `metricas` cuando solo se tienen las etiquetas numéricas."""
    def __init__(self, y, li):
        inverso = {i: n for n, i in li.items()}
        self._s = pd.Series([inverso[int(v)] for v in y])

    def __getitem__(self, clave):
        assert clave == "label"
        return self._s
