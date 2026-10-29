"""Exporta el modelo entrenado (para la web) junto con sus etiquetas y su ficha técnica."""
import json
from pathlib import Path

DISCLAIMER = ("Modelo entrenado con hojas de café ROBUSTA (dataset RoCoLe). No se ha validado con "
              "café arábica ni con cultivos colombianos. Es una orientación inicial, no un diagnóstico.")


def export_model(model, li: dict, hp, metrics: dict, out_dir) -> Path:
    """Guarda `cafe_robusta.keras`, `labels.json` y `model_card.json` en `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save(out / "cafe_robusta.keras")
    labels = [name for name, _ in sorted(li.items(), key=lambda kv: kv[1])]
    (out / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2))
    card = {
        "input_size": [224, 224], "preprocess": "mobilenet_v2 (x/127.5 - 1)", "labels": labels,
        "hyperparameters": {k: v for k, v in hp.values.items() if not k.startswith("tuner/")},
        "metrics": metrics, "disclaimer": DISCLAIMER,
    }
    (out / "model_card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2))
    return out
