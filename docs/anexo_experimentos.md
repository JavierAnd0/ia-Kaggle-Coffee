# Anexo: ¿ayuda aumentar la resolución o usar varias semillas?

**Pregunta.** La misma receta (variante 4b) dio 0,69 y 0,60 de exactitud en prueba en dos corridas, y las fotos de 1920×1080 se reducen a 224×224, donde las manchas pequeñas de la roya casi desaparecen. Se probó si entradas más grandes (320 y 448 px) o un conjunto de modelos mejoran el resultado.

**Método.** Receta 4b (sobremuestreo con imágenes reales + pesos de clase + ajuste fino), hiperparámetros de Hyperband sin cambios, 3 tamaños × 3 semillas (0, 1, 2), GPU T4 de Colab. Solo se usó la **validación** (244 fotos); el conjunto de prueba no se tocó. Datos en `results/experimentos_corridas.csv` y `results/experimentos_resumen.csv`.

## Resultados (validación)

| Tamaño | F1 macro (media ± desv.) | Exactitud (media ± desv.) | F1 del conjunto de 3 semillas | Tiempo por modelo |
|---|---|---|---|---|
| **224 px** | **0,662 ± 0,027** | **0,730 ± 0,019** | **0,668** | ~3 min |
| 320 px | 0,632 ± 0,022 | 0,716 ± 0,024 | 0,621 | ~4 min |
| 448 px | 0,610 ± 0,034 | 0,704 ± 0,031 | 0,591 | ~6 min |

Por clase (F1 medio de las 3 semillas): arañita roja 0,49 (224) → 0,41 (320) → 0,40 (448); roya 0,63 → 0,61 → 0,53.

## Lectura

1. **Aumentar la resolución no mejoró el resultado; empeoró.** 224 px fue el mejor en F1, en exactitud y en la clase minoritaria (arañita roja). La diferencia entre 224 y 448 (0,052 de F1) equivale a casi dos desviaciones estándar; la de 224 frente a 320 (0,030) es de aproximadamente una. Con solo 3 semillas por tamaño la evidencia es débil, pero ningún indicio favorece a los tamaños grandes.
2. **El conjunto de 3 semillas casi no aporta.** A 224 px, F1 0,668 frente a 0,662 de la media de los modelos sueltos (exactitud igual, 0,730). Una ganancia de 0,006 está dentro del ruido.
3. **Frente al modelo ya publicado** (validación: exactitud 0,7295, F1 macro 0,679): el conjunto a 224 px no lo supera (F1 0,668). Ese modelo cae dentro del rango normal de las semillas (0,633–0,687), es decir, parece una semilla ordinaria y no una excepción.
4. **Criterio acordado de antemano:** si no supera al modelo actual, no se reemplaza. **Se conserva el modelo actual.**

## Limitaciones de este experimento
- Los hiperparámetros (tasa de aprendizaje, etc.) se eligieron con Hyperband a 224 px; los tamaños grandes podrían necesitar su propio ajuste, que no se probó.
- Los pesos de ImageNet de MobileNetV2 existen solo para 224 px; con otros tamaños Keras avisa (`UserWarning`) y carga los de 224. Las convoluciones aceptan cualquier tamaño, pero es otro factor que no se aisló.
- Solo 3 semillas por tamaño y una sola partición de validación (244 fotos, 29 de arañita roja).
- La validación y la prueba difieren sistemáticamente (el modelo publicado: 0,73 y 0,60 de exactitud), lo que sugiere que las cifras dependen mucho de qué plantas caen en cada partición.

## Qué queda (en orden de prioridad)
1. **Datos de café arábica** (BRACOL o JMuBEN): la brecha de dominio es mayor que la que se puede cerrar con más ajuste sobre RoCoLe.
2. Repetir Hyperband optimizando el **F1 macro** (no la exactitud) a 224 px.
3. Fotos reales colombianas con etiqueta de un técnico, con consentimiento, para validar y reentrenar.
