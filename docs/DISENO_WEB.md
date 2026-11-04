# Diseño de la web «Cartilla de la Hoja»

Documento de decisiones de diseño: qué se construyó, por qué así, qué patrones se usaron y qué queda por hacer a mano.

## 1. Objetivo y usuario

Un productor de café (frecuentemente mayor, con poca práctica con la tecnología, con señal débil) toma una foto de una hoja y recibe:
(1) a qué se parece (sana / roya / arañita roja), (2) con qué seguridad, y (3) una guía de pasos y orientación sobre productos en lenguaje simple, en contexto colombiano/latinoamericano.

**Restricciones honestas que condicionan el diseño**

- El modelo se entrenó con café **robusta** (RoCoLe); en Colombia predomina el **arábica**. No está validado allí.
- Exactitud en prueba del modelo exportado: 0,60 (una corrida anterior de la misma receta dio 0,69: la variación entre corridas es de ~9 puntos); solo conoce 3 clases; recall de roya 0,52 (16 de 91 hojas con roya se clasificaron como sanas).
- Por eso la web **nunca** presenta el resultado como diagnóstico y **dice cuándo no está segura**.

## 2. Arquitectura

```
Navegador (celular)                         Servidor Node (Astro, adaptador node)
┌───────────────────────────┐   foto JPEG   ┌────────────────────────────────────────────┐
│ Isla React <Diagnostico>  │  (≤1920 px)   │ POST /api/diagnostico                      │
│  inicio→previa→analizando │ ────────────▶ │  1 límite de uso  2 tamaño  3 firma de bytes│
│  →resultado | error       │               │  4 Predictor.predecir()  ◀── PUERTO         │
│  voz · WhatsApp · guía    │ ◀──────────── │  5 decidir() → veredicto prudente           │
└───────────────────────────┘   JSON        │  6 une guía (conocimiento.json)            │
 Páginas estáticas: /, /guias/*, /modelo    └───────────────┬────────────────────────────┘
                                              ADAPTADORES: │ PredictorOnnx (modelo real)
                                                           │ PredictorSimulado (pruebas, marcado)
```

## 3. Decisiones y patrones de diseño

| Decisión | Patrón / técnica | Por qué |
|---|---|---|
| **Astro** con páginas estáticas y una sola isla de React | *Islands architecture* | Casi todo es contenido (guías, información). Solo la parte interactiva envía JavaScript: ~70 kB gzip, importante con datos móviles limitados. |
| El motor de predicción es intercambiable | *Ports & Adapters* (Strategy) + *Factory* (`lib/predictor/index.ts`) | La web se construyó y probó entera **antes** de tener el modelo. Cambiar de simulado a real no toca la interfaz. |
| `PREDICTOR=modelo` falla si no hay modelo | *Fail-fast / fail-closed* | Nunca debe servirse un resultado inventado en silencio. En modo simulado la pantalla muestra un aviso rojo permanente. |
| `decidir()` es una función pura | *Functional core, imperative shell* | Lógica crítica de seguridad (cuándo afirmar una enfermedad) sin E/S: fácil de probar y de auditar. |
| Umbrales conservadores y "no concluyente" | *Abstención* (reject option) | Con un modelo modesto es mejor decir «no estoy seguro» que adivinar. Alta: ≥80 %; media: ≥60 % (con 3 clases, ≥60 % ya implica ≥20 puntos de ventaja sobre la 2.ª opción); por debajo, «no concluyente». Con el modelo real se midió el acierto por nivel (validación / prueba): alta 90 % / 82 %, media 66 % / 54 %, baja 57 % / 41 %; es decir, el porcentaje sí informa, pero el nivel «medio» es poco fiable y la web lo dice. Los umbrales no se ajustaron con datos de arábica: ver §6. |
| Un resultado «sana» no tranquiliza del todo | Redacción de seguridad | El recall de roya es 0,52: la pantalla avisa que «no garantiza que la planta esté sana». |
| Contenido agronómico en `conocimiento.json` | *Content as data* + validación | Un agrónomo puede revisarlo sin tocar código; `tests/guia.test.ts` valida su estructura y la longitud de los pasos. |
| Interfaz como máquina de estados (`useReducer`) | *State machine* | Un paso por pantalla, sin estados imposibles; el foco pasa al título de cada paso (lectores de pantalla). |
| La foto se reduce en el celular a 1920 px | *Compresión en el borde* | ~200–400 KB en vez de varios MB. Se eligió 1920 (no 1024) porque las fotos de entrenamiento medían 1920×1080: con el modelo real, reducir a 1024 px desviaba las probabilidades hasta 0,10 respecto a Keras y a 1920 px hasta 0,06. Se corrige la orientación EXIF. |
| Preprocesamiento replicado a mano (`preprocesar.ts`) | *Paridad entrenamiento/servicio* | `sharp` redimensiona con otro filtro que `tf.image.resize`; se implementó el bilineal igual que TensorFlow. Medido sobre fotos reales: diferencia máxima 0,03 en escala −1…1 (≈1 nivel de gris, por el decodificador JPEG). |
| ONNX Runtime en lugar de TF.js | Elección técnica | El conversor TF.js no soporta Keras 3 (`.keras`); `model.export(format='onnx')` sí, y Keras vs ONNX dio diferencia 0,0. Runtime nativo, rápido, sin conversor frágil. |
| Sin base de datos ni guardar fotos | *Privacidad por diseño* | La foto se procesa en memoria y se descarta. No hay cuentas ni datos personales. |

**Accesibilidad (público mayor):** letra base ~20 px (Atkinson Hyperlegible, diseñada para baja visión), títulos en Fraunces; contraste alto; botones de ≥70 px; un solo paso por pantalla; lenguaje de «usted» y frases cortas (las pruebas rechazan pasos de más de 24 palabras); lectura en voz alta con la voz del dispositivo; envío del resumen por WhatsApp; respeta `prefers-reduced-motion`. Verificado con axe (WCAG 2 A/AA) y capturas en un Pixel 5 simulado.

**Seguridad de la API:** se ignora el `Content-Type` y se valida la firma real del archivo (JPEG/PNG/WEBP); límite de 6 MB y de píxeles al decodificar; 12 consultas/min por IP; respuestas sin caché; errores en lenguaje simple sin detalles internos.

## 4. Qué se probó

- **Unitarias (Vitest):** decisión y umbrales, redondeo de porcentajes (siempre suman 100), predictor simulado, fábrica (incluida la negativa a simular), preprocesamiento (interpolación idéntica a TensorFlow, EXIF), validación del contenido.
- **Navegador (Playwright, Pixel 5) + axe:** flujo foto→resultado, la foto sigue visible, barras con ancho, aviso de robusta y de «no diagnóstico», reducción del tamaño enviado, guías, accesibilidad en 3 páginas, rechazo de archivos inválidos/grandes y límite de uso.
- **Integración con ONNX:** servidor real con `PREDICTOR=modelo` responde con la ruta completa; sin modelo responde 503.
- **Con el modelo real (versión 4b-2026-11-04):** se reprodujeron exactamente sus métricas con Keras (val 0,7295; prueba 0,6027); Keras y ONNX difieren menos de 1e-6; la prueba de paridad Node↔Keras (6 fotos reales) mantiene la misma clase y desvía las probabilidades hasta 0,046 con la foto original y 0,06 con la foto reducida a 1920 px y recomprimida. La causa es el decodificador JPEG (≈1 nivel de gris por píxel), que el modelo amplifica un poco.
- **Variabilidad:** la misma receta entrenada dos veces dio 0,69 y 0,60 de exactitud en prueba. Cualquier cifra de este modelo debe leerse como un rango, no como un valor exacto.

## 5. Contenido agronómico: de dónde sale y su estado

Lo recopiló un agente de IA con búsqueda web (Cenicafé, FNC, UPR, SENASICA, UCR…). **No pudo abrir los PDF originales**; usó resultados de búsqueda. La lista de lo no verificado está en `por_verificar` dentro de `conocimiento.json`. La web **no da dosis, marcas ni precios**, solo tipos de productos y la instrucción de usar productos con registro vigente del ICA. **Debe revisarlo un ingeniero agrónomo antes de usarse con productores.**

## 6. Pendiente (manual)

1. ~~Exportar el modelo~~ (hecho: `web/model/`, versión 4b-2026-11-04). Queda desplegar con `PREDICTOR=modelo` y verificar `/api/salud`.
2. **Revisión agronómica** de `web/src/data/conocimiento.json` y de la sección «En Colombia» (variedades, umbrales, contactos reales con teléfonos).
3. **Validar con fotos de cultivos colombianos (arábica)**: medir exactitud y calibrar los umbrales de `decision.ts` con las probabilidades reales de validación (hoy son un punto de partida conservador, no calibrado).
4. **Prueba con productores reales** (lectura, botones, voz) y ajuste de lenguaje; posiblemente ampliar a otros países (hoy los contactos son de Colombia, Perú, México y Centroamérica solo como sitios oficiales).
5. **Desplegar** (Dockerfile incluido, no probado aquí) con `PREDICTOR=modelo`, HTTPS y, si hay mucho tráfico, un límite de uso compartido (el actual vive en memoria del proceso).
6. Mejoras posibles: funcionar sin conexión (service worker), más enfermedades (ojo de gallo, mancha de hierro), recomendar tomar foto del envés, medir calidad de la foto (borrosa/oscura) antes de analizar.

## 7. Cómo se trabajó (agentes y skills)

- **Agente de investigación** (general-purpose, en segundo plano): redactó la base de conocimiento con fuentes; se revisó su resultado y se suavizó una afirmación no verificable.
- **Skill `frontend-design`:** dirección visual «cartilla de campo» (papel, verde café, rojo cereza; Fraunces + Atkinson Hyperlegible).
- **Skill `code-review`:** revisión del código antes de entregar (ver historial del repositorio).
