# Diseño de la web «Cartilla de la Hoja»

Documento de decisiones de diseño: qué se construyó, por qué así, qué patrones se usaron y qué queda por hacer a mano.

## 1. Objetivo y usuario

Un productor de café (frecuentemente mayor, con poca práctica con la tecnología, con señal débil) toma una foto de una hoja y recibe:
(1) a qué se parece (sana / roya / arañita roja), (2) con qué seguridad, y (3) una guía de pasos y orientación sobre productos en lenguaje simple, en contexto colombiano/latinoamericano.

**Restricciones honestas que condicionan el diseño**

- El modelo se entrenó con café **robusta** (RoCoLe); en Colombia predomina el **arábica**. No está validado allí.
- Exactitud en prueba 0,69; solo conoce 3 clases; recall de roya 0,55 (21 de 91 hojas con roya se clasificaron como sanas).
- Por eso la web **nunca** presenta el resultado como diagnóstico y **dice cuándo no está segura**.

## 2. Arquitectura

```
Navegador (celular)                         Servidor Node (Astro, adaptador node)
┌───────────────────────────┐   foto JPEG   ┌────────────────────────────────────────────┐
│ Isla React <Diagnostico>  │  (≤1024 px)   │ POST /api/diagnostico                      │
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
| Umbrales conservadores y "no concluyente" | *Abstención* (reject option) | Con un modelo modesto es mejor decir «no estoy seguro» que adivinar. Alta: ≥80 %; media: ≥60 % (con 3 clases, ≥60 % ya implica ≥20 puntos de ventaja sobre la 2.ª opción); por debajo, «no concluyente». **No están calibrados**: ver §6. |
| Un resultado «sana» no tranquiliza del todo | Redacción de seguridad | El recall de roya es 0,55: la pantalla avisa que «no garantiza que la planta esté sana». |
| Contenido agronómico en `conocimiento.json` | *Content as data* + validación | Un agrónomo puede revisarlo sin tocar código; `tests/guia.test.ts` valida su estructura y la longitud de los pasos. |
| Interfaz como máquina de estados (`useReducer`) | *State machine* | Un paso por pantalla, sin estados imposibles; el foco pasa al título de cada paso (lectores de pantalla). |
| La foto se reduce en el celular a 1024 px | *Compresión en el borde* | ~100–300 KB en vez de varios MB. Se corrige la orientación EXIF. |
| Preprocesamiento replicado a mano (`preprocesar.ts`) | *Paridad entrenamiento/servicio* | `sharp` redimensiona con otro filtro que `tf.image.resize`; se implementó el bilineal igual que TensorFlow. Medido sobre fotos reales: diferencia máxima 0,03 en escala −1…1 (≈1 nivel de gris, por el decodificador JPEG). |
| ONNX Runtime en lugar de TF.js | Elección técnica | El conversor TF.js no soporta Keras 3 (`.keras`); `model.export(format='onnx')` sí, y Keras vs ONNX dio diferencia 0,0. Runtime nativo, rápido, sin conversor frágil. |
| Sin base de datos ni guardar fotos | *Privacidad por diseño* | La foto se procesa en memoria y se descarta. No hay cuentas ni datos personales. |

**Accesibilidad (público mayor):** letra base ~20 px (Atkinson Hyperlegible, diseñada para baja visión), títulos en Fraunces; contraste alto; botones de ≥70 px; un solo paso por pantalla; lenguaje de «usted» y frases cortas (las pruebas rechazan pasos de más de 24 palabras); lectura en voz alta con la voz del dispositivo; envío del resumen por WhatsApp; respeta `prefers-reduced-motion`. Verificado con axe (WCAG 2 A/AA) y capturas en un Pixel 5 simulado.

**Seguridad de la API:** se ignora el `Content-Type` y se valida la firma real del archivo (JPEG/PNG/WEBP); límite de 6 MB y de píxeles al decodificar; 12 consultas/min por IP; respuestas sin caché; errores en lenguaje simple sin detalles internos.

## 4. Qué se probó

- **Unitarias (Vitest):** decisión y umbrales, redondeo de porcentajes (siempre suman 100), predictor simulado, fábrica (incluida la negativa a simular), preprocesamiento (interpolación idéntica a TensorFlow, EXIF), validación del contenido.
- **Navegador (Playwright, Pixel 5) + axe:** flujo foto→resultado, la foto sigue visible, barras con ancho, aviso de robusta y de «no diagnóstico», reducción del tamaño enviado, guías, accesibilidad en 3 páginas, rechazo de archivos inválidos/grandes y límite de uso.
- **Integración con ONNX:** servidor real con `PREDICTOR=modelo` responde con la ruta completa; sin modelo responde 503.
- **Limitación de las pruebas:** el modelo real aún no existe en el repo; la tubería ONNX se verificó con un modelo de la misma arquitectura con pesos **aleatorios**. La prueba de paridad con Keras (`tests/onnx.test.ts`) se activa sola al subir `paridad.json`.

## 5. Contenido agronómico: de dónde sale y su estado

Lo recopiló un agente de IA con búsqueda web (Cenicafé, FNC, UPR, SENASICA, UCR…). **No pudo abrir los PDF originales**; usó resultados de búsqueda. La lista de lo no verificado está en `por_verificar` dentro de `conocimiento.json`. La web **no da dosis, marcas ni precios**, solo tipos de productos y la instrucción de usar productos con registro vigente del ICA. **Debe revisarlo un ingeniero agrónomo antes de usarse con productores.**

## 6. Pendiente (manual)

1. **Exportar el modelo** (`notebooks/exportar_modelo.ipynb`, Colab) y copiarlo a `web/model/` (ver `web/README.md`). Verificar `/api/salud` y correr `npm test` (paridad).
2. **Revisión agronómica** de `web/src/data/conocimiento.json` y de la sección «En Colombia» (variedades, umbrales, contactos reales con teléfonos).
3. **Validar con fotos de cultivos colombianos (arábica)**: medir exactitud y calibrar los umbrales de `decision.ts` con las probabilidades reales de validación (hoy son un punto de partida conservador, no calibrado).
4. **Prueba con productores reales** (lectura, botones, voz) y ajuste de lenguaje; posiblemente ampliar a otros países (hoy los contactos son de Colombia, Perú, México y Centroamérica solo como sitios oficiales).
5. **Desplegar** (Dockerfile incluido, no probado aquí) con `PREDICTOR=modelo`, HTTPS y, si hay mucho tráfico, un límite de uso compartido (el actual vive en memoria del proceso).
6. Mejoras posibles: funcionar sin conexión (service worker), más enfermedades (ojo de gallo, mancha de hierro), recomendar tomar foto del envés, medir calidad de la foto (borrosa/oscura) antes de analizar.

## 7. Cómo se trabajó (agentes y skills)

- **Agente de investigación** (general-purpose, en segundo plano): redactó la base de conocimiento con fuentes; se revisó su resultado y se suavizó una afirmación no verificable.
- **Skill `frontend-design`:** dirección visual «cartilla de campo» (papel, verde café, rojo cereza; Fraunces + Atkinson Hyperlegible).
- **Skill `code-review`:** revisión del código antes de entregar (ver historial del repositorio).
