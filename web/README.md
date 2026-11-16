# Cartilla de la Hoja (web)

Orientación inicial sobre enfermedades de la hoja de café a partir de una foto. Hecha con
**Astro 5** (páginas y rutas), **React** (una "isla" interactiva) y **Node.js** (servidor + ONNX Runtime).

> Es una orientación inicial, **no un diagnóstico**. El modelo se entrenó con café **robusta** (RoCoLe)
> y no se ha validado con café arábica ni con cultivos colombianos.

## Uso rápido

```bash
cd web
npm install
npm run dev            # http://localhost:4321  (modo de prueba si no hay modelo)
npm test               # pruebas unitarias (Vitest)
npm run test:e2e       # pruebas de navegador + accesibilidad (Playwright + axe)
npm run build && npm start
```

## Variables de entorno

| Variable | Valor | Efecto |
|---|---|---|
| `PREDICTOR` | `auto` (por defecto) | Usa el modelo si existe en `model/`; si no, el simulado (con aviso en pantalla). |
| | `modelo` | **Exige** el modelo. Si falta, `/api/diagnostico` responde 503. Recomendado en producción. |
| | `simulado` | Fuerza el modo de prueba. |
| `MODEL_DIR` | ruta | Carpeta del modelo (por defecto `./model`). |
| `TRUST_PROXY` | `1` | Detrás de un proxy inverso/HTTPS: toma la IP del cliente del **último** valor de `X-Forwarded-For` (el que agrega el proxy de Railway, Render, Fly…; los primeros los puede inventar el cliente). Sin esto, todos comparten la IP del proxy. Actívelo **solo** si hay un proxy de confianza delante. |
| `CABECERA_IP` | nombre de cabecera | Opcional, con `TRUST_PROXY=1`: usa esa cabecera en vez de `X-Forwarded-For` (p. ej. `x-real-ip`, `fly-client-ip`, `cf-connecting-ip`). Úsela solo si su proxy la **reescribe** siempre. |
| `LIMITE_POR_MINUTO` | número (20) | Fotos por minuto por IP en `/api/diagnostico`. `/api/combinar` (no corre el modelo) tiene un cupo aparte del triple. |
| `INFERENCIAS_SIMULTANEAS` | número (2) | Cuántas fotos se decodifican y pasan por el modelo a la vez (1–4); las demás esperan su turno. Evita que varias fotos simultáneas agoten la memoria. |

### Límite de uso

Cada foto revisada cuenta una vez contra `LIMITE_POR_MINUTO`. Revisar varias hojas de una planta
llama además a `/api/combinar`, pero ese paso solo promedia números y tiene su propio cupo, así que
**no** gasta las consultas de foto. El valor por defecto (20) es holgado a propósito: en Colombia los
operadores móviles comparten una misma IP entre muchos celulares (CGNAT), y una vereda entera puede
salir a internet por la misma dirección. Si ve respuestas 429 legítimas, súbalo.
| `HOST`, `PORT` | | Los usa el servidor de Astro (`node dist/server/entry.mjs`). |

## Conectar el modelo real

El modelo exportado (versión 4b-2026-11-04) ya está en `web/model/`. Para reemplazarlo:

1. Ejecute `notebooks/exportar_modelo.ipynb` en Colab (GPU T4, ~10 min) y descargue `modelo_cafe.zip`.
2. Descomprima y copie `web/model/*` (`cafe_robusta.onnx`, `labels.json`, `model_card.json`, `paridad.json`) a esta carpeta `web/model/`, y actualice `src/data/metricas.json` (una prueba avisa si quedan desactualizadas).
3. `PREDICTOR=modelo npm run build && npm start` y abra `/api/salud`: debe decir `"predictor":"onnx","modoSimulado":false`.
4. `npm test`: la prueba de paridad compara las probabilidades de Node con las de Keras.

Diseño, decisiones y límites: [`../docs/DISENO_WEB.md`](../docs/DISENO_WEB.md).
