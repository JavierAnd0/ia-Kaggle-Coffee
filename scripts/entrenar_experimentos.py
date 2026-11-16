"""Experimentos de mejora SIN Jupyter (PyCharm, terminal, Kaggle, servidor).

Hace lo mismo que notebooks/experimentos_mejora.ipynb: receta 4b con varios tamaños de entrada y semillas
(solo validación), arma el conjunto de modelos del mejor tamaño, lo evalúa una vez en test y lo exporta a
web/model/. Es REANUDABLE: si se corta, vuelva a ejecutar el mismo comando y salta lo ya entrenado.

Ejemplos (desde la raíz del repositorio):
    python scripts/entrenar_experimentos.py --prueba            # 2-3 min, verifica que todo está instalado
    python scripts/entrenar_experimentos.py --tamanos 224 320 --semillas 0 1 2
    python scripts/entrenar_experimentos.py --solo-rejilla      # no exporta, solo compara
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
os.chdir(RAIZ)                      # las rutas de data_splits/*.csv son relativas a la raíz
sys.path.insert(0, str(RAIZ))


def leer_argumentos(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tamanos", type=int, nargs="+", default=[224, 320, 448], help="lados de entrada en píxeles")
    p.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epocas", type=int, default=30)
    p.add_argument("--epocas-ft", type=int, default=10, help="épocas de ajuste fino")
    p.add_argument("--checkpoints", default=None, help="carpeta de modelos parciales (por defecto ./checkpoints o $CHECKPOINT_DIR)")
    p.add_argument("--salida", default=str(RAIZ / "web" / "model"), help="carpeta de exportación")
    p.add_argument("--tamano-final", type=int, default=None, help="forzar el tamaño del conjunto final")
    p.add_argument("--limite-horas", type=float, default=None,
                   help="se detiene solo (con todo guardado) si no cabe otra corrida en este tiempo; útil en Kaggle (usar ~10)")
    p.add_argument("--solo-rejilla", action="store_true", help="comparar tamaños sin exportar")
    p.add_argument("--prueba", action="store_true",
                   help="prueba rápida con pocas fotos, sin pesos de ImageNet y en una carpeta temporal (no toca el repo)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = leer_argumentos(argv)
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    from src import config, experiments, export, generators, tuning

    if a.prueba:
        tmp = Path(tempfile.mkdtemp(prefix="cafe_prueba_"))
        config.PRETRAINED = False
        a.tamanos, a.semillas, a.epocas, a.epocas_ft = [64, 96], [0, 1], 1, 1
        a.checkpoints, a.salida = str(tmp / "checkpoints"), str(tmp / "web_model")
        config.RESULTS_DIR = tmp / "results"
        print(f"MODO PRUEBA (resultados sin valor). Archivos temporales en {tmp}")

    gpus = tf.config.list_physical_devices("GPU")
    print("GPU detectada:" if gpus else "SIN GPU: el entrenamiento completo será MUY lento (horas o días).", gpus or "")

    tr = pd.read_csv(config.SPLITS_DIR / "train.csv")
    va = pd.read_csv(config.SPLITS_DIR / "val.csv")
    te = pd.read_csv(config.SPLITS_DIR / "test.csv")
    if a.prueba:
        def poco(d): return d.groupby("label").head(6).reset_index(drop=True)
        tr, va, te = poco(tr), poco(va), poco(te)
    faltan = [f for f in tr["filepath"].head(3) if not Path(f).exists()]
    if faltan:
        print(f"No encuentro las fotos (por ejemplo {faltan[0]}). Ejecute desde el repositorio con la carpeta data/ descargada.")
        return 2

    li = generators.make_label_index(tr["label"])
    hp = tuning.load_best_hp()
    try:
        corridas, resumen = experiments.correr_rejilla(hp, tr, va, li, tamanos=tuple(a.tamanos), semillas=tuple(a.semillas),
                                                       epocas=a.epocas, epocas_ft=a.epocas_ft, dir_modelos=a.checkpoints,
                                                       n_extra=4 if a.prueba else 150,
                                                       limite_segundos=a.limite_horas * 3600 if a.limite_horas else None)
    except experiments.TiempoAgotado as e:
        print(f"\n{e}\nTodo lo entrenado quedó guardado en {experiments.carpeta_checkpoints(a.checkpoints)}. Vuelva a ejecutar el mismo comando para continuar.")
        return 3
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    corridas.to_csv(config.RESULTS_DIR / "experimentos_corridas.csv", index=False)
    resumen.to_csv(config.RESULTS_DIR / "experimentos_resumen.csv")
    print("\nRESUMEN POR TAMAÑO (validación)\n", resumen.round(3).to_string())
    if a.solo_rejilla:
        return 0

    r = experiments.finalizar(hp, tr, va, te, li, resumen, tuple(a.semillas), a.checkpoints, a.salida, a.tamano_final)
    nombres = sorted(li, key=li.get)
    print(f"\nTamaño elegido: {r['mejor']}  |  versión {r['version']}  |  diferencia Keras vs ONNX {r['diferencia_onnx']:.2e}")
    print("VALIDACIÓN:", {k: round(v, 3) for k, v in r["metricas"]["val"].items()})
    print("TEST (una sola vez):", {k: round(v, 3) for k, v in r["metricas"]["test"].items()})
    pred = r["p_test"].argmax(1)
    print(classification_report(r["y_test"], pred, target_names=nombres, zero_division=0))
    print(pd.DataFrame(confusion_matrix(r["y_test"], pred), index=nombres, columns=nombres))
    zip_path = export.empaquetar(Path(a.salida).parent / "modelo_cafe.zip", [Path(a.salida)],
                                 [config.RESULTS_DIR / "experimentos_corridas.csv", config.RESULTS_DIR / "experimentos_resumen.csv"])
    print(f"\nListo. Modelo exportado en {r['out']}\nPaquete para enviar: {zip_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
