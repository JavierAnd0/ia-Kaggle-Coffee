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

## Web «Cartilla de la Hoja»
En `web/` hay una aplicación (Astro + React + Node.js) que recibe la foto de una hoja y devuelve a qué se parece, con qué seguridad y una guía de pasos para el productor. Es una orientación inicial, no un diagnóstico, y el modelo se entrenó con café **robusta**. Ver `web/README.md` y `docs/DISENO_WEB.md`. Modelo actual en `web/model/`; para mejorarlo: `notebooks/experimentos_mejora.ipynb` (resolución × semillas × conjunto de modelos) y `notebooks/exportar_modelo.ipynb` (reexportar la receta actual).

## Modelo v5 (recomendado para la web)
Un solo comando entrena y evalúa de forma honesta el modelo para la web: `scripts/entrenar_v5.py` (o, en Colab Pro con GPU A100/L4, *Ejecutar todas* en [`notebooks/entrenar_colab_v5.ipynb`](https://colab.research.google.com/github/JavierAnd0/ia-Kaggle-Coffee/blob/claude/modelo-v5/notebooks/entrenar_colab_v5.ipynb)).
- **Datos:** `data_splits/v5/` quita las 24 fotos de los 12 grupos de duplicados exactos con etiquetas contradictorias y las 55 copias exactas restantes; partición 70/15/15 **estratificada** (StratifiedGroupKFold) y agrupada por planta + duplicados, semilla 42. Los CSV antiguos (`data_splits/*.csv`) quedan para reproducir el informe.
- **Receta:** EfficientNetV2-B0 (opcional ConvNeXt-Tiny) con el reescalado dentro del modelo (la entrada sigue en [-1, 1], como espera la web); cabeza 8 épocas y luego todo el backbone con AdamW + coseno; pesos de clase + label smoothing 0,1 (sin sobremuestreo); selección por F1 macro de validación; semillas 0 a 4; determinismo de TF (`--no-determinismo` para desactivarlo).
- **Protocolo:** la decisión entre el mejor modelo individual y el conjunto de las 5 semillas se toma en validación (conjunto solo si gana por ≥ 0,01) y el test se evalúa una sola vez. Resultados en `results/v5/`.
- Prueba rápida: `python scripts/entrenar_v5.py --rapido` (no toca `web/model`).

**Versión de Python:** 3.11 o 3.12 (probado con 3.12.13 y TensorFlow 2.19.1 / Keras 3.15.1; ver `requirements.txt`). En Colab se usa el TensorFlow que ya trae.

## Uso
```bash
pip install -r requirements.txt
jupyter notebook notebooks/proyecto_medio.ipynb   # o abrir en Colab
```
El dataset está en `data/` (carpetas `coffee___healthy` 791, `coffee___rust` 602, `coffee___red_spider_mite` 167). Grupo = planta (`C10P10E1.jpg` -> `C10P10`, 390 plantas) para evitar leakage. `config.FAST_MODE = True` para pruebas rápidas.

## Ejecutar el entrenamiento en tu computador (PyCharm, terminal)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/entrenar_experimentos.py --prueba        # 2-3 min: comprueba que todo está instalado (no entrena de verdad)
python scripts/entrenar_experimentos.py                 # experimentos completos (reanudable: si se corta, repita el comando)
```
- **Sin GPU el entrenamiento completo puede tardar muchas horas o días.** Para eso conviene Colab (GPU T4/L4) o Kaggle Notebooks.
- **Kaggle** (GPU gratis, sigue corriendo con el navegador cerrado): ver `docs/KAGGLE.md`.
- Los notebooks (`notebooks/*.ipynb`) funcionan en Kaggle, Colab, PyCharm y Jupyter: encuentran solos la carpeta del proyecto. En Colab, con `USAR_DRIVE = True`, los modelos parciales se guardan en Google Drive y sobreviven a una desconexión.
- Los modelos parciales se guardan en `checkpoints/` (o en `$CHECKPOINT_DIR`) y no se suben a Git.

## Estado
Ejecutado de punta a punta en Colab (T4): Hyperband, validación cruzada, comparación de mejoras y evaluación en test. Pendiente solo completar en el informe los datos del autor (nombre, asignatura, docente, fecha) y revisar las referencias.
