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
| `HOST`, `PORT` | | Los usa el servidor de Astro (`node dist/server/entry.mjs`). |

## Conectar el modelo real

1. Ejecute `notebooks/exportar_modelo.ipynb` en Colab (GPU T4, ~10 min) y descargue `modelo_cafe.zip`.
2. Descomprima y copie `web/model/*` (`cafe_robusta.onnx`, `labels.json`, `model_card.json`, `paridad.json`) a esta carpeta `web/model/`.
3. `PREDICTOR=modelo npm run build && npm start` y abra `/api/salud`: debe decir `"predictor":"onnx","modoSimulado":false`.
4. `npm test`: la prueba de paridad compara las probabilidades de Node con las de Keras.

Diseño, decisiones y límites: [`../docs/DISENO_WEB.md`](../docs/DISENO_WEB.md).
