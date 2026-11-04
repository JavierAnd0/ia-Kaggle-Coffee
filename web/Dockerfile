# Imagen de producción (NO probada en este entorno: no había Docker disponible).
FROM node:22-slim AS construir
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build && npm prune --omit=dev

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production HOST=0.0.0.0 PORT=4321 PREDICTOR=modelo
COPY --from=construir /app/node_modules ./node_modules
COPY --from=construir /app/dist ./dist
COPY --from=construir /app/model ./model
EXPOSE 4321
# PREDICTOR=modelo: si falta web/model/ el servicio NO arranca el diagnóstico (nunca simula en silencio).
CMD ["node", "./dist/server/entry.mjs"]
