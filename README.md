# ia-Kaggle-Coffee

Proyecto medio: clasificación de enfermedades en hojas de café con el dataset [RoCoLe](https://www.kaggle.com/datasets/nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset) (sector agro-industrial).

## Contenido
1. **Preparación de datos** (`src/data_prep.py`, `src/generators.py`, `src/gan.py`): partición 70/15/15 sin data leakage (agrupada por planta/duplicados), CSV de splits en `data_splits/`, preprocesamiento, data generator por lotes (`tf.data`) y GAN condicional.
2. **Hiperparámetros** (`src/model.py`, `src/tuning.py`): Hyperband con Keras Tuner (learning rate, dropout, unidades, backbone, optimizador, batch size, épocas).
3. **Validación cruzada** (`src/cross_val.py`): K-fold, Stratified K-fold, Hold-out, Monte Carlo; Leave-p-out/Leave-one-out/Time-series explicados en el notebook.

## Uso
```bash
pip install -r requirements.txt
jupyter notebook notebooks/proyecto_medio.ipynb   # o abrir en Colab
```
Descarga con `kagglehub` (requiere `KAGGLE_USERNAME` y `KAGGLE_KEY`). `config.FAST_MODE = True` para pruebas rápidas.

## Estado
Código escrito pero **aún no ejecutado contra el dataset real** (sin acceso a Kaggle desde el entorno de desarrollo). La carga de etiquetas del notebook (sección 0) debe ajustarse tras inspeccionar la estructura real.
