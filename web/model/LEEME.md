# Carpeta del modelo

Contiene el modelo real (lo genera `notebooks/exportar_modelo.ipynb`):

- `cafe_robusta.onnx` — red MobileNetV2 + cabeza densa, exportada a ONNX.
- `labels.json`, `model_card.json` — etiquetas y ficha (versión, hiperparámetros, métricas, aviso).
- `paridad.json` — probabilidades de Keras para 6 fotos reales; `tests/onnx.test.ts` las compara con Node.

Versión actual: `4b-2026-11-04` (exactitud en prueba 0,60; una corrida anterior de la misma receta dio 0,69).
Es café **robusta**; no se ha validado con arábica ni con cultivos colombianos.

Si estos archivos faltan, la web arranca en **modo de prueba** (resultados simulados, aviso rojo en pantalla).
En producción use `PREDICTOR=modelo`: sin ellos el servicio falla en lugar de simular.
