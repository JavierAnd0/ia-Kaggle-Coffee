# ia-Kaggle-Coffee

Proyecto medio: clasificación de enfermedades en hojas de café con el dataset [RoCoLe](https://www.kaggle.com/datasets/nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset) (sector agro-industrial).

## Contenido
1. **Preparación de datos** (`src/data_prep.py`, `src/generators.py`, `src/gan.py`): partición 70/15/15 sin data leakage (agrupada por planta/duplicados), CSV de splits en `data_splits/`, preprocesamiento, data generator por lotes (`tf.data`) y GAN condicional.
2. **Hiperparámetros** (`src/model.py`, `src/tuning.py`): Hyperband con Keras Tuner (learning rate, dropout, unidades, backbone, optimizador, batch size, épocas).
3. **Validación cruzada** (`src/cross_val.py`): K-fold, Stratified K-fold, Hold-out, Monte Carlo; Leave-p-out/Leave-one-out/Time-series explicados en el notebook.

4. **Mejoras y evaluación final** (`src/train.py`): pesos de clase, fine-tuning y GAN (imágenes sintéticas solo en train), comparadas sobre validación; el test se usa una sola vez.

## Resultados guardados (`results/`)
- `best_hp.json` y `cv_results.csv`: resultados de Hyperband y de la validación cruzada, ejecutados en Colab (transcritos de la salida del notebook).
- `ablation_val.csv` y `test_report.txt`: se generan al ejecutar las secciones finales del notebook.

## Uso
```bash
pip install -r requirements.txt
jupyter notebook notebooks/proyecto_medio.ipynb   # o abrir en Colab
```
El dataset está en `data/` (carpetas `coffee___healthy` 791, `coffee___rust` 602, `coffee___red_spider_mite` 167). Grupo = planta (`C10P10E1.jpg` -> `C10P10`, 390 plantas) para evitar leakage. `config.FAST_MODE = True` para pruebas rápidas.

## Estado
Hyperband y la validación cruzada se ejecutaron en Colab (T4). Las mejoras (pesos de clase, fine-tuning, GAN) y la evaluación final están implementadas y probadas con un subconjunto pequeño, pendientes de ejecutarse completas.
