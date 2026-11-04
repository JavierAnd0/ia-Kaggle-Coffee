// @ts-check
import { defineConfig } from 'astro/config';
import node from '@astrojs/node';
import react from '@astrojs/react';

// Modo "server" con el adaptador de Node: las páginas informativas se pre-generan
// (prerender = true) y solo la API de diagnóstico se ejecuta en el servidor.
export default defineConfig({
  output: 'server',
  adapter: node({ mode: 'standalone' }),
  integrations: [react()],
  security: { checkOrigin: true },
  // Módulos nativos: se cargan desde node_modules en tiempo de ejecución, no se empaquetan.
  vite: { ssr: { external: ['onnxruntime-node', 'sharp'] } },
});
