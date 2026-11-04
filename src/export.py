"""Exporta el modelo entrenado (para la web) junto con sus etiquetas y su ficha técnica."""
import json
from datetime import date
from pathlib import Path

DISCLAIMER = ("Modelo entrenado con hojas de café ROBUSTA (dataset RoCoLe). No se ha validado con "
              "café arábica ni con cultivos colombianos. Es una orientación inicial, no un diagnóstico.")


def export_model(model, li: dict, hp, metrics: dict, out_dir, version: str | None = None) -> Path:
    """Guarda `cafe_robusta.keras`, `labels.json` y `model_card.json` en `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "cafe_robusta.keras")
    labels = [name for name, _ in sorted(li.items(), key=lambda kv: kv[1])]
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    card = {
        "version": version or f"4b-{date.today().isoformat()}",
        "input_size": [224, 224], "preprocess": "mobilenet_v2 (x/127.5 - 1)", "labels": labels,
        "hyperparameters": {k: v for k, v in hp.values.items() if not k.startswith("tuner/")},
        "metrics": metrics, "disclaimer": DISCLAIMER,
    }
    (out / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2))
    return out


def export_onnx(model, out_dir) -> Path:
    """Exporta a ONNX (lo que usa la web con ONNX Runtime). Requiere: pip install onnx tf2onnx onnxruntime."""
    import numpy as np
    import tensorflow as tf

    path = Path(out_dir) / "cafe_robusta.onnx"
    model.predict(np.zeros((1, 224, 224, 3), "float32"), verbose=0)   # el exportador exige haber llamado al modelo
    try:
        model.export(str(path), format="onnx")
    except Exception as error:                                          # Keras antiguo: ruta alternativa con tf2onnx
        print("model.export(onnx) falló, se usa tf2onnx:", type(error).__name__)
        import tf2onnx
        firma = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)

        @tf.function(input_signature=firma)
        def inferir(x):
            return model(x, training=False)

        tf2onnx.convert.from_function(inferir, input_signature=firma, output_path=str(path))
    return path


def verificar_onnx(model, onnx_path, n: int = 4) -> float:
    """Diferencia máxima entre las probabilidades de Keras y de ONNX Runtime (debe ser ~0)."""
    import numpy as np
    import onnxruntime as ort

    x = (np.random.default_rng(0).random((n, 224, 224, 3)) * 2 - 1).astype("float32")
    sesion = ort.InferenceSession(str(onnx_path))
    onnx_out = sesion.run(None, {sesion.get_inputs()[0].name: x})[0]
    return float(np.abs(model.predict(x, verbose=0) - onnx_out).max())


def casos_paridad(model, hp, li: dict, df, por_clase: int = 2) -> list:
    """Probabilidades de Keras para algunas imágenes reales con el pipeline de ENTRENAMIENTO.
    La web las compara con su propio preprocesamiento (web/tests/onnx.test.ts)."""
    from .generators import make_dataset
    from .model import BACKBONES

    muestra = df.groupby("label").head(por_clase).reset_index(drop=True)
    ds = make_dataset(muestra, li, batch_size=1, preprocess=BACKBONES[hp.values["backbone"]][1])
    probs = model.predict(ds, verbose=0)
    nombres = sorted(li, key=li.get)
    from . import config

    def relativa(ruta: str) -> str:                      # los CSV pueden traer rutas absolutas (Colab)
        ruta = Path(ruta)
        return ruta.relative_to(config.ROOT).as_posix() if ruta.is_absolute() and config.ROOT in ruta.parents else ruta.as_posix()

    return [{"archivo": relativa(f), "probabilidades": {n: float(p[i]) for i, n in enumerate(nombres)}}
            for f, p in zip(muestra["filepath"], probs)]
