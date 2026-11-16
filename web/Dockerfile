# Imagen de producción. Probada con `docker build` y `docker run` (2026-11-16): /api/salud responde
# {"ok":true,"predictor":"onnx","modoSimulado":false}.
FROM node:22-slim AS construir
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build && npm prune --omit=dev \
  # onnxruntime-node trae binarios para Windows y macOS (~220 MB) que en Linux no se usan.
  && rm -rf node_modules/onnxruntime-node/bin/napi-v*/win32 node_modules/onnxruntime-node/bin/napi-v*/darwin

FROM node:22-slim
# Certificados raíz para conexiones HTTPS salientes (p. ej. si se descarga el modelo o se reportan errores).
RUN apt-get update \
  && apt-get install -y --no-install-recommends ca-certificates \
  && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ENV NODE_ENV=production HOST=0.0.0.0 PORT=4321 PREDICTOR=modelo
COPY --from=construir --chown=node:node /app/node_modules ./node_modules
COPY --from=construir --chown=node:node /app/dist ./dist
COPY --from=construir --chown=node:node /app/model ./model
# Sin privilegios de root: si alguien lograra ejecutar código en el proceso, no controla el contenedor.
USER node
EXPOSE 4321
# /api/salud carga el modelo, así que la verificación también lo "calienta" antes de la primera foto.
# Se usa node (la imagen slim no trae curl). start-period da tiempo a cargar ONNX Runtime.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD node -e "fetch('http://127.0.0.1:'+(process.env.PORT||4321)+'/api/salud').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"
# PREDICTOR=modelo: si falta web/model/ el servicio NO arranca el diagnóstico (nunca simula en silencio).
CMD ["node", "./dist/server/entry.mjs"]
