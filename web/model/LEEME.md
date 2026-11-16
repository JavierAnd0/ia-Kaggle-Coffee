# Carpeta del modelo

Contiene el modelo real (lo genera `notebooks/exportar_modelo.ipynb`):

- `cafe_robusta.onnx` — red MobileNetV2 + cabeza densa, exportada a ONNX.
- `labels.json`, `model_card.json` — etiquetas y ficha (versión, hiperparámetros, métricas, aviso).
- `paridad.json` — probabilidades de Keras para 6 fotos reales; `tests/onnx.test.ts` las compara con Node.

Versión actual: `4b-2026-11-04` (exactitud en prueba 0,60; una corrida anterior de la misma receta dio 0,69).
Es café **robusta**; no se ha validado con arábica ni con cultivos colombianos.

## Modelo v5 (en preparación)
`scripts/entrenar_v5.py` / `notebooks/entrenar_colab_v5.ipynb` generan aquí la versión `5-<fecha>-efficientnetv2b0`
(`-ens5` si se exporta el conjunto de 5 semillas, que es un único ONNX que promedia los miembros). El contrato con la web
no cambia: entrada NHWC float32 `[1, L, L, 3]` en [-1, 1] (resize bilineal + x/127,5 − 1, como `preprocesar.ts`), salida
softmax de 3 clases en el orden de `labels.json`. El reescalado y la normalización propios del backbone van **dentro**
del modelo (`preprocess` en `model_card.json`). También escribe `metricas_web.json` (cifras reales del modelo exportado).
Las cifras de v5 usan `data_splits/v5/` (más limpios), por lo que no son directamente comparables con las de 4b.

Si estos archivos faltan, la web arranca en **modo de prueba** (resultados simulados, aviso rojo en pantalla).
En producción use `PREDICTOR=modelo`: sin ellos el servicio falla en lugar de simular.
