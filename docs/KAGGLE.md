# Entrenar en Kaggle (paso a paso)

Sirve para `notebooks/experimentos_mejora.ipynb` (9 modelos: 3 tamaños × 3 semillas). Kaggle da GPU gratis con cuota semanal y puede seguir corriendo con el navegador cerrado.

## Antes de empezar (una sola vez)
1. Cuenta en kaggle.com y **verificación del teléfono** (Settings → Phone Verification). Sin ella no se activan GPU ni Internet.

## Crear y configurar el notebook
1. En Kaggle: **Create → New Notebook**.
2. **File → Import Notebook** y suba `notebooks/experimentos_mejora.ipynb` (descárguelo de GitHub: botón «Download raw file»). Si su Kaggle ofrece importar desde un enlace de GitHub, también sirve.
3. Panel derecho **Settings**:
   - **Accelerator:** GPU T4 ×2 (o P100). El código usa una sola GPU.
   - **Internet:** **On** (el notebook clona este repositorio de GitHub; ahí están los datos y el código).
   - Persistence: no hace falta.
4. Revise en la primera celda de configuración que diga `FAST_MODE = False`.

## Ejecutar sin que se corte
- **No** use solo «Run All» con la pestaña abierta: si cierra el navegador, se pierde.
- Use **Save Version → Save & Run All (Commit)**. Corre en segundo plano; puede cerrar la página. Siga el avance en **Versions** (menú de la parte superior) → su ejecución → *Log*.
- El código se **detiene solo a las 10 horas** (`LIMITE_HORAS`) con todo lo entrenado guardado, porque Kaggle corta a las 12 h y no queremos perder nada.
- Cada línea del log `[n/9] tamaño … semilla …` es un modelo terminado.

## Si no alcanzó en una sola sesión
1. Abra la versión terminada → pestaña **Output**: ahí está la carpeta `checkpoints/`.
2. En el notebook, **Add Input → Your Work → Notebook Output** y elija esa versión.
3. Vuelva a ejecutar con **Save & Run All**. El notebook copia los modelos anteriores y **continúa donde quedó**.

## Recoger el resultado
Cuando termine la rejilla y la exportación, la pestaña **Output** tendrá `modelo_cafe.zip`. Descárguelo, copie su carpeta `web/model/` a `web/model/` del repositorio y súbala a GitHub. Pásele a su asistente la tabla del log (`RESUMEN POR TAMAÑO`) y las métricas de test.

## Recortar el experimento (recomendado la primera vez)
En la celda de configuración use `TAMANOS = (224, 320)` y `SEMILLAS = (0, 1)`: son 4 modelos en lugar de 9 y se obtiene una primera respuesta bastante antes.

## Notas y límites
- Kaggle limita las horas de GPU por semana (la cuota actual aparece en su perfil); las cifras de esta guía son aproximadas y pueden cambiar.
- No se ha ejecutado en Kaggle real desde el desarrollo: se simuló su entorno (carpetas `/kaggle/working` y `/kaggle/input`, límite de tiempo y reanudación). Si algo falla, copie el mensaje de error completo del log.
