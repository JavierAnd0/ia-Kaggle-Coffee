"""Modelo v5 para la web: protocolo completo y honesto en UNA ejecución (pensado para Colab Pro, GPU A100/L4).

1. Splits v5 (data_splits/v5/): sin etiquetas contradictorias, 70/15/15 estratificado y agrupado.
2. Entrena EfficientNetV2-B0 (o ConvNeXt-Tiny) con las semillas 0..4 a 224 px. Fase 1 solo cabeza,
   fase 2 todo el backbone (AdamW + coseno). Pesos de clase + label smoothing 0,1 (sin sobremuestreo).
   Se selecciona por F1 macro de VALIDACIÓN (época a época y entre semillas).
3. Decide ANTES de mirar test qué se exporta: el conjunto de las 5 semillas solo si su F1 macro de
   validación supera al mejor modelo individual por >= 0,01; si no, el mejor individual.
4. Evalúa en TEST una sola vez (mejor individual y conjunto) y guarda el resultado: si se vuelve a
   ejecutar, se reutiliza en lugar de volver a evaluar.
5. Exporta a la carpeta de la web: cafe_robusta.onnx, labels.json, model_card.json, paridad.json y
   metricas_web.json, verificando Keras vs ONNX con fotos reales.

REANUDABLE por semilla: cada semilla terminada deja sus pesos y probabilidades en la carpeta de checkpoints
(--checkpoints o $CHECKPOINT_DIR); si se corta, vuelva a ejecutar el mismo comando.

Ejemplos (desde la raíz del repositorio):
    python scripts/entrenar_v5.py --rapido --sin-imagenet     # prueba de humo en CPU (minutos; resultados sin valor)
    python scripts/entrenar_v5.py                             # protocolo completo (GPU)
    python scripts/entrenar_v5.py --backbone convnext_tiny --checkpoints /content/drive/MyDrive/cafe_v5/ckpt
"""
import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
os.chdir(RAIZ)                      # las rutas de data_splits/v5/*.csv son relativas a la raíz
sys.path.insert(0, str(RAIZ))

MARGEN_CONJUNTO = 0.01              # el conjunto debe ganar por al menos esto en F1 macro de validación


def leer_argumentos(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default="efficientnetv2b0", choices=["efficientnetv2b0", "convnext_tiny"])
    p.add_argument("--tamano", type=int, default=224, help="lado de entrada en píxeles")
    p.add_argument("--semillas", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--epocas-cabeza", type=int, default=8)
    p.add_argument("--epocas-ft", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--checkpoints", default=None, help="carpeta para reanudar (por defecto $CHECKPOINT_DIR o ./checkpoints)")
    p.add_argument("--salida", default=None, help="carpeta de exportación (por defecto web/model; en --rapido, una carpeta aparte)")
    p.add_argument("--resultados", default=None, help="carpeta de CSV/JSON de resultados (por defecto results/v5)")
    p.add_argument("--zip", default=None, help="paquete final (por defecto modelo_cafe_v5.zip en la raíz)")
    p.add_argument("--no-determinismo", action="store_true", help="no activar tf.config.experimental.enable_op_determinism")
    p.add_argument("--sin-precision-mixta", action="store_true", help="entrenar en float32 también en GPU")
    p.add_argument("--regenerar-splits", action="store_true", help="volver a crear data_splits/v5/ desde data/")
    p.add_argument("--sin-imagenet", action="store_true", help="sin pesos preentrenados (solo pruebas sin internet)")
    p.add_argument("--rapido", action="store_true",
                   help="prueba de humo: pocas fotos, 2 semillas, 1+1 épocas, 128 px; NO escribe en web/model")
    p.add_argument("--verbose", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = leer_argumentos(argv)
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import classification_report, confusion_matrix

    from src import config, data_prep, experiments, export
    from src import train_v5 as t5
    from src.model import build_model_v5, ensamblar_v5

    if a.rapido:
        a.semillas, a.epocas_cabeza, a.epocas_ft = a.semillas[:2], 1, 1
        a.tamano = min(a.tamano, 128)
    receta = t5.RecetaV5(backbone=a.backbone, lado=a.tamano, batch_size=a.batch, epocas_cabeza=a.epocas_cabeza,
                         epocas_ft=a.epocas_ft, precision_mixta=not a.sin_precision_mixta, pretrained=not a.sin_imagenet)
    etiqueta_ckpt = f"v5_{a.backbone}_{a.tamano}" + ("_rapido" if a.rapido else "") + ("_sinimagenet" if a.sin_imagenet else "")
    ckpt = Path(a.checkpoints or os.environ.get("CHECKPOINT_DIR") or RAIZ / "checkpoints") / etiqueta_ckpt
    ckpt.mkdir(parents=True, exist_ok=True)
    salida = Path(a.salida) if a.salida else (ckpt / "web_model_rapido" if a.rapido else RAIZ / "web" / "model")
    resultados = Path(a.resultados) if a.resultados else (ckpt / "results_rapido" if a.rapido else RAIZ / "results" / "v5")
    resultados.mkdir(parents=True, exist_ok=True)
    zip_path = Path(a.zip) if a.zip else (ckpt / "modelo_cafe_v5_rapido.zip" if a.rapido else RAIZ / "modelo_cafe_v5.zip")

    # La receta queda fija por carpeta: reanudar con otra receta mezclaría modelos distintos.
    ruta_receta = ckpt / "receta.json"
    clave = {**receta.como_dict(), "semillas": a.semillas, "determinista": not a.no_determinismo}
    if ruta_receta.exists():
        previa = json.loads(ruta_receta.read_text())
        if previa != clave:
            print(f"La carpeta {ckpt} es de otra receta:\n  antes: {previa}\n  ahora: {clave}\n"
                  "Use otra carpeta (--checkpoints) o bórrela para empezar de cero.")
            return 4
    else:
        t5.guardar_json(ruta_receta, clave)

    gpus = tf.config.list_physical_devices("GPU")
    print("TensorFlow", tf.__version__, "| Keras", tf.keras.__version__, "| GPU:", gpus or "NINGUNA (será muy lento)")
    print("Checkpoints:", ckpt, "| exportación:", salida)
    if a.rapido:
        print("MODO RÁPIDO: prueba de humo, sus métricas no sirven para decidir nada.")

    tr, va, te = data_prep.preparar_splits_v5(regenerar=a.regenerar_splits)
    if a.rapido:
        def poco(d, k): return d.groupby("label").head(k).reset_index(drop=True)
        tr, va, te = poco(tr, 8), poco(va, 4), poco(te, 4)
    faltan = [f for f in tr["filepath"].head(3) if not Path(f).exists()]
    if faltan:
        print(f"No encuentro las fotos (p. ej. {faltan[0]}). Ejecute desde el repositorio con la carpeta data/.")
        return 2
    li = {c: i for i, c in enumerate(t5.ETIQUETAS_WEB)}
    assert sorted(set(tr["label"])) == t5.ETIQUETAS_WEB, f"clases inesperadas: {sorted(set(tr['label']))}"
    y_val, y_test = va["label"].map(li).values, te["label"].map(li).values
    print("Pesos de clase (train v5):", {n: round(w, 3) for n, w in zip(t5.ETIQUETAS_WEB, t5.class_weights(tr, li).values())})

    # ---------------------------------------------------------------- 1. semillas (reanudable)
    filas, p_val, p_val_flip = [], {}, {}
    for s in a.semillas:
        r_w, r_p, r_f, r_j = (ckpt / f"s{s}.weights.h5", ckpt / f"s{s}_p_val.npy",
                              ckpt / f"s{s}_p_val_flip.npy", ckpt / f"s{s}.json")
        if all(r.exists() for r in (r_w, r_p, r_f, r_j)):
            fila = json.loads(r_j.read_text())
            print(f"Semilla {s}: ya estaba hecha (val F1 macro {fila['f1_macro']:.4f}), se reutiliza")
        else:
            print(f"\n=== Semilla {s} ===")
            try:
                m, info = t5.entrenar_semilla(receta, tr, va, li, s, not a.no_determinismo, a.verbose)
            except Exception as e:                  # UnimplementedError, InvalidArgumentError, errores de XLA...
                if a.no_determinismo or "determinis" not in str(e).lower():
                    raise
                print("Una operación no tiene versión determinista en esta GPU; se repite la semilla sin determinismo.\n ", str(e)[:300])
                m, info = t5.entrenar_semilla(receta, tr, va, li, s, False, a.verbose)
            ds_val = t5.make_dataset_v5(va, li, receta.lado, receta.batch_size)
            p, pf = t5.predecir(m, ds_val), t5.predecir(m, ds_val, tta_flip=True)
            m.save_weights(r_w)
            np.save(r_p, p)
            np.save(r_f, pf)
            fila = {**{k: v for k, v in info.items() if k != "pesos_clase"}, **t5.metricas(p, y_val, li),
                    "f1_macro_tta_flip": t5.metricas((p + pf) / 2, y_val, li)["f1_macro"]}
            t5.guardar_json(r_j, fila)                  # al final: su existencia marca la semilla como completa
            print(f"Semilla {s}: val F1 macro {fila['f1_macro']:.4f}  exactitud {fila['accuracy']:.4f}  ({fila['segundos'] // 60} min)")
            del m
            tf.keras.backend.clear_session()
        filas.append(fila)
        p_val[s], p_val_flip[s] = np.load(r_p), np.load(r_f)

    corridas = pd.DataFrame(filas).set_index("semilla")
    corridas.to_csv(resultados / "corridas_v5.csv")
    mejor_s = int(corridas["f1_macro"].idxmax())
    p_ens = np.mean([p_val[s] for s in a.semillas], axis=0)
    p_ens_tta = np.mean([(p_val[s] + p_val_flip[s]) / 2 for s in a.semillas], axis=0)
    val_ens = t5.metricas(p_ens, y_val, li)
    resumen = {
        "backbone": a.backbone, "tamano": a.tamano, "semillas": a.semillas,
        "val_f1_macro_media": float(corridas["f1_macro"].mean()),
        "val_f1_macro_desv": float(corridas["f1_macro"].std(ddof=1)) if len(corridas) > 1 else 0.0,
        "val_exactitud_media": float(corridas["accuracy"].mean()),
        "mejor_semilla": mejor_s, "val_f1_macro_mejor": float(corridas.loc[mejor_s, "f1_macro"]),
        "val_f1_macro_conjunto": val_ens["f1_macro"], "val_exactitud_conjunto": val_ens["accuracy"],
        "val_f1_macro_mejor_tta_flip": float(corridas.loc[mejor_s, "f1_macro_tta_flip"]),
        "val_f1_macro_conjunto_tta_flip": t5.metricas(p_ens_tta, y_val, li)["f1_macro"],
    }

    # ---------------------------------------------------------------- 2. decisión ANTES de ver test
    ruta_decision = ckpt / "decision.json"
    if ruta_decision.exists():
        decision = json.loads(ruta_decision.read_text())
        print("Decisión ya tomada antes (se respeta):", decision["exportar"])
    else:
        usa_ens = len(a.semillas) > 1 and resumen["val_f1_macro_conjunto"] - resumen["val_f1_macro_mejor"] >= MARGEN_CONJUNTO
        decision = {"exportar": "conjunto" if usa_ens else "individual", "mejor_semilla": mejor_s,
                    "regla": f"conjunto solo si F1 macro val(conjunto) - F1 macro val(mejor individual) >= {MARGEN_CONJUNTO}",
                    "tta": "no se usa en la web (solo se informa en validación)",
                    "val_f1_macro_mejor": resumen["val_f1_macro_mejor"], "val_f1_macro_conjunto": resumen["val_f1_macro_conjunto"]}
        t5.guardar_json(ruta_decision, decision)
    resumen["exportar"] = decision["exportar"]
    print("\nVALIDACIÓN por semilla\n", corridas[["f1_macro", "accuracy", "f1_healthy", "f1_red_spider_mite", "f1_rust",
                                                  "f1_macro_tta_flip", "segundos"]].round(4).to_string())
    print(f"\nF1 macro val: media {resumen['val_f1_macro_media']:.4f} ± {resumen['val_f1_macro_desv']:.4f} | "
          f"mejor (semilla {mejor_s}) {resumen['val_f1_macro_mejor']:.4f} | conjunto {resumen['val_f1_macro_conjunto']:.4f}")
    print(f"TTA (volteo horizontal, solo val): mejor {resumen['val_f1_macro_mejor_tta_flip']:.4f} | "
          f"conjunto {resumen['val_f1_macro_conjunto_tta_flip']:.4f}  (no se exporta: la web no lo hace)")
    print("SE EXPORTA:", decision["exportar"])

    # ---------------------------------------------------------------- 3. modelos float32 para test y exportación
    tf.keras.backend.clear_session()
    tf.keras.mixed_precision.set_global_policy("float32")
    t5.fijar_determinismo(0, not a.no_determinismo)

    def cargar(s, i):
        m = build_model_v5(a.backbone, len(li), a.tamano, receta.dropout, pretrained=False, name=f"miembro_{i}")
        m.load_weights(ckpt / f"s{s}.weights.h5")
        return m

    miembros = {s: cargar(s, i) for i, s in enumerate(a.semillas)}

    # ---------------------------------------------------------------- 4. test UNA sola vez
    ruta_test = ckpt / "test_una_vez.json"
    if ruta_test.exists():
        test = json.loads(ruta_test.read_text())
        p_test_ind, p_test_ens = np.load(ckpt / "p_test_individual.npy"), np.load(ckpt / "p_test_conjunto.npy")
        print("\nTest ya evaluado antes: se reutiliza el resultado guardado (no se vuelve a evaluar).")
    else:
        ds_te = t5.make_dataset_v5(te, li, a.tamano, receta.batch_size)
        p_por_s = {s: m.predict(ds_te, verbose=0) for s, m in miembros.items()}
        p_test_ind, p_test_ens = p_por_s[decision["mejor_semilla"]], np.mean(list(p_por_s.values()), axis=0)
        np.save(ckpt / "p_test_individual.npy", p_test_ind)
        np.save(ckpt / "p_test_conjunto.npy", p_test_ens)
        test = {"individual": t5.metricas(p_test_ind, y_test, li), "conjunto": t5.metricas(p_test_ens, y_test, li),
                "imagenes": int(len(te))}
        t5.guardar_json(ruta_test, test)
    t5.guardar_json(resultados / "test_v5.json", test)
    resumen.update({"test_f1_macro_individual": test["individual"]["f1_macro"], "test_exactitud_individual": test["individual"]["accuracy"],
                    "test_f1_macro_conjunto": test["conjunto"]["f1_macro"], "test_exactitud_conjunto": test["conjunto"]["accuracy"]})
    t5.guardar_json(resultados / "resumen_v5.json", resumen)
    pd.Series(resumen).to_csv(resultados / "resumen_v5.csv", header=["valor"])
    t5.guardar_json(resultados / "decision_v5.json", decision)
    data_prep.imprimir_conteos(tr, va, te).to_csv(resultados / "splits_v5_conteos.csv")
    print("\nTEST (una sola vez):")
    print("  mejor individual:", {k: round(v, 4) for k, v in test["individual"].items()})
    print("  conjunto        :", {k: round(v, 4) for k, v in test["conjunto"].items()})

    # ---------------------------------------------------------------- 5. exportación del elegido
    usa_ens = decision["exportar"] == "conjunto"
    if usa_ens:
        modelo = ensamblar_v5(list(miembros.values()), a.tamano)
        pv, pt = p_ens, p_test_ens
    else:
        modelo = miembros[decision["mejor_semilla"]]
        pv, pt = p_val[decision["mejor_semilla"]], p_test_ind
    version = export.version_v5(a.backbone, len(a.semillas) if usa_ens else 0) + ("-rapido" if a.rapido else "")
    metricas_card = {"val": t5.metricas(pv, y_val, li), "test": t5.metricas(pt, y_test, li)}
    hp = {**receta.como_dict(), "optimizer": "adam (cabeza) -> adamw + coseno (todo)", "loss": "categorical_crossentropy",
          "imbalance": "class_weight balanced (train v5), sin sobremuestreo"}
    notas = {
        "arquitectura": (f"conjunto de {len(a.semillas)} modelos {a.backbone} (semillas {a.semillas}), promedio de probabilidades"
                         if usa_ens else f"{a.backbone} (semilla {decision['mejor_semilla']})"),
        "splits": "data_splits/v5 (sin etiquetas contradictorias, 70/15/15 estratificado y agrupado por planta+MD5)",
        "seleccion": decision, "validacion_semillas": {k: v for k, v in resumen.items() if k.startswith("val_")},
        "test_ambos": test,
    }
    out = export.export_model(modelo, li, hp, metricas_card, salida, version=version, input_size=a.tamano,
                              notas=notas, preprocess=export.PREPROCESS_V5)
    onnx_path = export.export_onnx(modelo, out)
    x_real = export.entradas_reales(va, li, a.tamano, n=12)
    dif_real = export.verificar_onnx(modelo, onnx_path, x=x_real)
    dif_ruido = export.verificar_onnx(modelo, onnx_path)
    print(f"\nKeras vs ONNX: diferencia máxima {dif_real:.2e} con {len(x_real)} fotos reales, {dif_ruido:.2e} con ruido")
    assert dif_real < 1e-3, "ONNX no reproduce a Keras con fotos reales"
    t5.guardar_json(out / "paridad.json", export.casos_paridad_v5(modelo, li, te, a.tamano))
    web = experiments.resumen_para_web(pv, y_val, pt, y_test, li, version)
    t5.guardar_json(out / "metricas_web.json", web)
    t5.guardar_json(resultados / "onnx_verificacion.json", {"max_dif_fotos_reales": dif_real, "max_dif_ruido": dif_ruido,
                                                             "fotos": int(len(x_real)), "version": version})

    nombres = t5.ETIQUETAS_WEB
    print(f"\nExportado {version} en {out} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
    print(classification_report(y_test, pt.argmax(1), target_names=nombres, zero_division=0))
    print(pd.DataFrame(confusion_matrix(y_test, pt.argmax(1), labels=range(3)), index=nombres, columns=nombres))
    zip_path = export.empaquetar(zip_path, [out], [resultados])
    print(f"\nPaquete: {zip_path}\nFecha: {date.today().isoformat()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
