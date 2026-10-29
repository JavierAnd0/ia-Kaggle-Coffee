# ia-Kaggle-Coffee

Proyecto medio: clasificación de enfermedades en hojas de café con el dataset [RoCoLe](https://www.kaggle.com/datasets/nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset) (sector agro-industrial).

## Contenido
1. **Preparación de datos** (`src/data_prep.py`, `src/generators.py`, `src/gan.py`): partición 70/15/15 sin data leakage (agrupada por planta/duplicados), CSV de splits en `data_splits/`, preprocesamiento, data generator por lotes (`tf.data`) y GAN condicional.
2. **Hiperparámetros** (`src/model.py`, `src/tuning.py`): Hyperband con Keras Tuner (learning rate, dropout, unidades, backbone, optimizador, batch size, épocas).
3. **Validación cruzada** (`src/cross_val.py`): K-fold, Stratified K-fold, Hold-out, Monte Carlo; Leave-p-out/Leave-one-out/Time-series explicados en el notebook.

4. **Mejoras y evaluación final** (`src/train.py`): pesos de clase, fine-tuning y GAN (imágenes sintéticas solo en train), comparadas sobre validación; el test se usa una sola vez.

## Resultados guardados (`results/`)
- `best_hp.json` y `cv_results.csv`: resultados de Hyperband y de la validación cruzada, ejecutados en Colab (archivos originales descargados de Colab).
- `ablation_val.csv` y `test_report.txt`: comparación de variantes en validación y reporte final en test (variante elegida: sobremuestreo con imágenes reales; exactitud 0,69, F1 macro 0,60).
- `docs/informe_proyecto_medio.docx`: informe académico con todos los resultados.

## Uso
```bash
pip install -r requirements.txt
jupyter notebook notebooks/proyecto_medio.ipynb   # o abrir en Colab
```
El dataset está en `data/` (carpetas `coffee___healthy` 791, `coffee___rust` 602, `coffee___red_spider_mite` 167). Grupo = planta (`C10P10E1.jpg` -> `C10P10`, 390 plantas) para evitar leakage. `config.FAST_MODE = True` para pruebas rápidas.

## Estado
Ejecutado de punta a punta en Colab (T4): Hyperband, validación cruzada, comparación de mejoras y evaluación en test. Pendiente solo completar en el informe los datos del autor (nombre, asignatura, docente, fecha) y revisar las referencias.
