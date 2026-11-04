# Carpeta del modelo

Aquí debe estar el modelo real (lo genera `notebooks/exportar_modelo.ipynb`):

- `cafe_robusta.onnx`
- `labels.json`
- `model_card.json`
- `paridad.json` (opcional, activa la prueba de paridad con Keras)

Mientras falten, la web arranca en **modo de prueba** (resultados simulados, con un aviso rojo en pantalla).
En producción use `PREDICTOR=modelo`: sin estos archivos el servicio falla en lugar de simular.
