import { defineConfig, devices } from '@playwright/test';
import { existsSync } from 'node:fs';

// En este entorno Chromium viene preinstalado; en otros equipos Playwright usa el suyo.
const chromium = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome';
const launchOptions = existsSync(chromium) ? { executablePath: chromium, args: ['--no-sandbox'] } : {};

export default defineConfig({
  testDir: './e2e',
  timeout: 60_000,
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:4399', launchOptions },
  projects: [
    { name: 'celular', use: { ...devices['Pixel 5'], launchOptions } },
  ],
  webServer: {
    command: 'npm run build && node dist/server/entry.mjs',
    url: 'http://127.0.0.1:4399/api/salud',
    reuseExistingServer: false,
    timeout: 180_000,
    env: { HOST: '127.0.0.1', PORT: '4399', PREDICTOR: process.env.PREDICTOR ?? 'simulado', LIMITE_POR_MINUTO: '60' },
  },
});
