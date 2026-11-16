"""Exporta el modelo entrenado (para la web) junto con sus etiquetas y su ficha técnica."""
import json
from datetime import date
from pathlib import Path

DISCLAIMER = ("Modelo entrenado con hojas de café ROBUSTA (dataset RoCoLe). No se ha validado con "
              "café arábica ni con cultivos colombianos. Es una orientación inicial, no un diagnóstico.")


PREPROCESS_V4 = "mobilenet_v2 (x/127.5 - 1)"
PREPROCESS_V5 = ("in-model rescaling; input in [-1,1] (bilinear resize to LxL, then x/127.5 - 1; "
                 "the model rescales to [0,255] and applies the backbone normalization)")


def version_v5(backbone: str, ensamble: int = 0, fecha: str | None = None) -> str:
    """`5-<fecha>-<backbone>` (y `-ens<k>` si es un conjunto de k modelos)."""
    v = f"5-{fecha or date.today().isoformat()}-{backbone}"
    return v + (f"-ens{ensamble}" if ensamble > 1 else "")


def export_model(model, li: dict, hp, metrics: dict, out_dir, version: str | None = None,
                 input_size: int | None = None, notas: dict | None = None, preprocess: str = PREPROCESS_V4) -> Path:
    """Guarda `cafe_robusta.keras`, `labels.json` y `model_card.json` en `out_dir`.
    `hp`: HyperParameters de Keras Tuner o un dict simple (receta v5)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "cafe_robusta.keras")
    labels = [name for name, _ in sorted(li.items(), key=lambda kv: kv[1])]
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    valores = hp if isinstance(hp, dict) else hp.values
    card = {
        "version": version or f"4b-{date.today().isoformat()}",
        "input_size": [input_size or int(model.input_shape[1])] * 2, "preprocess": preprocess, "labels": labels,
        "hyperparameters": {k: v for k, v in valores.items() if not k.startswith("tuner/")},
        "metrics": metrics, "disclaimer": DISCLAIMER, **(notas or {}),
    }
    (out / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2))
    return out


def export_onnx(model, out_dir) -> Path:
    """Exporta a ONNX (lo que usa la web con ONNX Runtime). Requiere: pip install onnx tf2onnx onnxruntime."""
    import numpy as np
    import tensorflow as tf

    import onnx

    path = Path(out_dir) / "cafe_robusta.onnx"
    lado = int(model.input_shape[1])
    model.predict(np.zeros((1, lado, lado, 3), "float32"), verbose=0)   # el exportador exige haber llamado al modelo

    def una_entrada() -> bool:
        # Con model.export(format="onnx") las constantes de Normalization (EfficientNetV2/ConvNeXt) pueden
        # quedar como entradas extra del grafo; la web solo alimenta una entrada.
        return len(onnx.load(str(path)).graph.input) == 1

    try:                                                                # tf2onnx congela todas las variables
        import tf2onnx
        firma = (tf.TensorSpec((None, lado, lado, 3), tf.float32, name="input"),)

        @tf.function(input_signature=firma)
        def inferir(x):
            return model(x, training=False)

        tf2onnx.convert.from_function(inferir, input_signature=firma, output_path=str(path), opset=17)
    except Exception as error:
        print("tf2onnx falló, se usa model.export(format='onnx'):", type(error).__name__, error)
        model.export(str(path), format="onnx")
    if not una_entrada():
        raise RuntimeError(f"El ONNX exportado tiene más de una entrada: {[i.name for i in onnx.load(str(path)).graph.input]}")
    return path


def verificar_onnx(model, onnx_path, n: int = 4, x=None) -> float:
    """Diferencia máxima entre las probabilidades de Keras y de ONNX Runtime (debe ser ~0).
    `x`: lote de entrada ya preprocesado (p. ej. fotos reales, ver `entradas_reales`); si no, ruido en [-1, 1]."""
    import numpy as np
    import onnxruntime as ort

    lado = int(model.input_shape[1])
    if x is None:
        x = (np.random.default_rng(0).random((n, lado, lado, 3)) * 2 - 1).astype("float32")
    x = np.asarray(x, dtype="float32")
    sesion = ort.InferenceSession(str(onnx_path))
    nombre = sesion.get_inputs()[0].name
    onnx_out = np.concatenate([sesion.run(None, {nombre: x[i:i + 1]})[0] for i in range(len(x))])   # batch 1, como la web
    keras_out = model.predict(x, verbose=0)
    assert np.allclose(onnx_out.sum(1), 1, atol=1e-3), "la salida ONNX no suma 1 (¿falta el softmax?)"
    return float(np.abs(keras_out - onnx_out).max())


def entradas_reales(df, li: dict, lado: int, n: int = 12):
    """Lote de fotos reales con el preprocesamiento de servicio v5 (resize bilineal + x/127.5 - 1)."""
    import numpy as np
    from .generators import make_dataset_v5

    muestra = df.groupby("label").head(max(1, n // max(1, df["label"].nunique()))).reset_index(drop=True)
    ds = make_dataset_v5(muestra, li, lado, batch_size=len(muestra), cache=False)
    return np.concatenate([x.numpy() for x, _ in ds])


def casos_paridad_v5(model, li: dict, df, lado: int, por_clase: int = 2) -> list:
    """Como `casos_paridad` pero con el pipeline de servicio v5 (el modelo recibe [-1, 1])."""
    from . import config
    from .generators import make_dataset_v5

    muestra = df.groupby("label").head(por_clase).reset_index(drop=True)
    probs = model.predict(make_dataset_v5(muestra, li, lado, batch_size=1, cache=False), verbose=0)
    nombres = sorted(li, key=li.get)

    def relativa(ruta: str) -> str:
        ruta = Path(ruta)
        return ruta.relative_to(config.ROOT).as_posix() if ruta.is_absolute() and config.ROOT in ruta.parents else ruta.as_posix()

    return [{"archivo": relativa(f), "probabilidades": {nm: float(p[i]) for i, nm in enumerate(nombres)}}
            for f, p in zip(muestra["filepath"], probs)]


def casos_paridad(model, hp, li: dict, df, por_clase: int = 2) -> list:
    """Probabilidades de Keras para algunas imágenes reales con el pipeline de ENTRENAMIENTO.
    La web las compara con su propio preprocesamiento (web/tests/onnx.test.ts)."""
    from .generators import make_dataset
    from .model import BACKBONES

    muestra = df.groupby("label").head(por_clase).reset_index(drop=True)
    ds = make_dataset(muestra, li, batch_size=1, preprocess=BACKBONES[hp.values["backbone"]][1],
                      img_size=(int(model.input_shape[1]),) * 2)
    probs = model.predict(ds, verbose=0)
    nombres = sorted(li, key=li.get)
    from . import config

    def relativa(ruta: str) -> str:                      # los CSV pueden traer rutas absolutas (Colab)
        ruta = Path(ruta)
        return ruta.relative_to(config.ROOT).as_posix() if ruta.is_absolute() and config.ROOT in ruta.parents else ruta.as_posix()

    return [{"archivo": relativa(f), "probabilidades": {n: float(p[i]) for i, n in enumerate(nombres)}}
            for f, p in zip(muestra["filepath"], probs)]


def empaquetar(destino, carpetas, extras=(), excluir=(".keras",)) -> Path:
    """Crea un .zip con las carpetas/archivos dados (rutas relativas a la raíz del proyecto), sin los .keras."""
    import zipfile
    from . import config

    destino = Path(destino)
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED) as z:
        for ruta in [Path(c) for c in carpetas] + [Path(e) for e in extras]:
            absoluta = ruta if ruta.is_absolute() else config.ROOT / ruta
            archivos = [absoluta] if absoluta.is_file() else sorted(p for p in absoluta.rglob("*") if p.is_file())
            for a in archivos:
                if a.suffix not in excluir:   # fuera del repo (p. ej. Drive): se conserva la carpeta dada
                    z.write(a, a.relative_to(config.ROOT) if config.ROOT in a.parents else a.relative_to(absoluta.parent))
    return destino
