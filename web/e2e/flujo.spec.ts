import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import path from 'node:path';
import sharp from 'sharp';

const FOTO = path.resolve(process.cwd(), '..', 'data', 'coffee___rust', 'C10P10E2.jpg');

test('la portada avisa que es orientación inicial, café robusta y modo de prueba', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('Tome una foto');
  await expect(page.getByText('orientación inicial, no un diagnóstico').first()).toBeVisible();
  await expect(page.getByText(/café robusta/).first()).toBeVisible();
  await expect(page.getByTestId('modo-prueba')).toContainText('MODO DE PRUEBA');
});

test('flujo completo: foto → vista previa → resultado con porcentajes y guía', async ({ page }) => {
  await page.goto('/');
  await page.getByTestId('entrada-galeria').setInputFiles(FOTO);
  await expect(page.getByRole('heading', { name: /Paso 2/ })).toBeVisible();
  await page.getByRole('button', { name: /Sí, revisar/ }).click();
  await expect(page.getByRole('heading', { name: /Paso 3/ })).toBeVisible({ timeout: 20_000 });

  const valores = await page.locator('.barras .valor').allInnerTexts();
  expect(valores).toHaveLength(3);
  expect(valores.map((v) => parseInt(v, 10)).reduce((a, b) => a + b, 0)).toBe(100);
  await expect(page.locator('.sello')).toBeVisible();
  // La foto revisada debe seguir visible (regresión: antes se liberaba su URL demasiado pronto)
  expect(await page.getByAltText('Foto revisada').evaluate((i: HTMLImageElement) => i.naturalWidth)).toBeGreaterThan(0);
  // Las barras deben tener ancho visible
  const anchos = await page.locator('.barras .relleno').evaluateAll((els) => els.map((e) => (e as HTMLElement).getBoundingClientRect().width));
  expect(Math.max(...anchos)).toBeGreaterThan(5);
  await expect(page.getByRole('link', { name: /WhatsApp/ })).toHaveAttribute('href', /wa\.me/);
  await expect(page.getByText('no es un diagnóstico').or(page.getByText('no un diagnóstico')).first()).toBeVisible();

  await page.getByRole('button', { name: /Revisar otra planta/ }).click();
  await expect(page.getByRole('heading', { name: /Paso 1/ })).toBeVisible();
});

test('la foto se reduce antes de enviarse (ahorro de datos)', async ({ page }) => {
  // La foto original pesa varios MB; lo que se envía debe ser mucho menor.
  await page.goto('/');
  let enviada = 0;
  await page.route('**/api/diagnostico', async (ruta) => {
    enviada = ruta.request().postDataBuffer()?.length ?? 0;
    await ruta.continue();
  });
  await page.getByTestId('entrada-galeria').setInputFiles(FOTO);
  await page.getByRole('button', { name: /Sí, revisar/ }).click();
  await expect(page.getByRole('heading', { name: /Paso 3/ })).toBeVisible({ timeout: 20_000 });
  console.log(`Bytes enviados: ${enviada}`);
  expect(enviada).toBeGreaterThan(1_000);
  expect(enviada).toBeLessThan(600_000);
});

test('las guías se leen sin enviar fotos y citan fuentes', async ({ page }) => {
  for (const clase of ['rust', 'red_spider_mite', 'healthy']) {
    await page.goto(`/guias/${clase}`);
    await expect(page.getByRole('heading', { name: 'Qué hacer hoy' })).toBeVisible();
    await expect(page.getByText('De dónde sale esta información')).toBeVisible();
  }
});

test('la página del modelo muestra métricas y límites', async ({ page }) => {
  await page.goto('/modelo');
  await expect(page.getByText('café robusta').first()).toBeVisible();
  await expect(page.getByText(/60 %/).first()).toBeVisible();
  await expect(page.getByText(/Café robusta, no arábica/)).toBeVisible();
});

for (const ruta of ['/', '/guias/rust', '/modelo']) {
  test(`accesibilidad (axe) en ${ruta}`, async ({ page }) => {
    await page.goto(ruta);
    const r = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze();
    expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`)).toEqual([]);
  });
}

test('la API rechaza archivos que no son imágenes y fotos demasiado grandes', async ({ request }) => {
  const mala = await request.post('/api/diagnostico', { data: Buffer.from('esto no es una imagen'), headers: { 'Content-Type': 'image/jpeg' } });
  expect(mala.status()).toBe(400);
  expect((await mala.json()).codigo).toBe('imagen_invalida');

  const grande = Buffer.alloc(7 * 1024 * 1024, 0xff);
  grande[0] = 0xff; grande[1] = 0xd8; grande[2] = 0xff;
  const r = await request.post('/api/diagnostico', { data: grande, headers: { 'Content-Type': 'image/jpeg' } });
  expect(r.status()).toBe(413);
});


const FOTO2 = path.resolve(process.cwd(), '..', 'data', 'coffee___healthy', 'C10P10H1.jpg');

test('varias hojas de la misma planta: se combinan y se muestra cada una', async ({ page }) => {
  await page.goto('/');
  await page.getByTestId('entrada-galeria').setInputFiles(FOTO);
  await page.getByRole('button', { name: /Sí, revisar/ }).click();
  await expect(page.getByRole('heading', { name: /Paso 3: resultado$/ })).toBeVisible({ timeout: 20_000 });
  await page.getByRole('button', { name: /otra hoja de la misma planta/ }).click();
  await expect(page.getByRole('heading', { name: /Hoja 2 de la misma planta/ })).toBeVisible();
  await expect(page.getByText('Ya revisó 1 hoja')).toBeVisible();
  await page.getByTestId('entrada-galeria').setInputFiles(FOTO2);
  await page.getByRole('button', { name: /Sí, revisar/ }).click();
  await expect(page.getByRole('heading', { name: /resultado de 2 hojas/ })).toBeVisible({ timeout: 20_000 });
  await expect(page.getByRole('heading', { name: 'Resultado de cada hoja' })).toBeVisible();
  await expect(page.getByText(/Hoja 1:/)).toBeVisible();
  await expect(page.getByText(/Hoja 2:/)).toBeVisible();
  const valores = await page.locator('.barras .valor').allInnerTexts();
  expect(valores.map((v) => parseInt(v, 10)).reduce((a, b) => a + b, 0)).toBe(100);
  const r = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa']).analyze();
  expect(r.violations.map((v) => `${v.id}: ${v.nodes.map((n) => n.target.join(' ')).join(' | ')}`)).toEqual([]);
  await page.getByRole('button', { name: /otra planta/ }).click();
  await expect(page.getByRole('heading', { name: /Paso 1/ })).toBeVisible();
});

test('una foto borrosa o muy oscura muestra un aviso suave y deja continuar', async ({ page }) => {
  await page.goto('/');
  const borrosa = await sharp(FOTO).resize(1280, 720).blur(8).jpeg().toBuffer();
  await page.getByTestId('entrada-galeria').setInputFiles({ name: 'borrosa.jpg', mimeType: 'image/jpeg', buffer: borrosa });
  await expect(page.getByTestId('aviso-calidad')).toContainText('borrosa');
  await expect(page.getByRole('button', { name: /Sí, revisar/ })).toBeEnabled();   // no bloquea

  await page.getByRole('button', { name: /No, tomar otra/ }).click();
  const oscura = await sharp(FOTO).resize(1280, 720).linear(0.25, 0).jpeg().toBuffer();
  await page.getByTestId('entrada-galeria').setInputFiles({ name: 'oscura.jpg', mimeType: 'image/jpeg', buffer: oscura });
  await expect(page.getByTestId('aviso-calidad')).toContainText('oscura');
});

test('una foto buena no muestra aviso de calidad', async ({ page }) => {
  await page.goto('/');
  await page.getByTestId('entrada-galeria').setInputFiles(FOTO);
  await expect(page.getByRole('heading', { name: /Paso 2/ })).toBeVisible();
  await expect(page.getByTestId('aviso-calidad')).toHaveCount(0);
});

test('/api/combinar valida lo que recibe', async ({ request }) => {
  const ok = await request.post('/api/combinar', { data: { fotos: [{ healthy: 0.1, rust: 0.8, red_spider_mite: 0.1 }, { healthy: 0.2, rust: 0.7, red_spider_mite: 0.1 }] } });
  expect(ok.status()).toBe(200);
  const cuerpo = await ok.json();
  expect(cuerpo.coinciden).toBe(true);
  expect(cuerpo.fotos).toHaveLength(2);
  expect(cuerpo.veredicto.clase).toBe('rust');
  const una = await request.post('/api/combinar', { data: { fotos: [{ healthy: 0.1, rust: 0.8, red_spider_mite: 0.1 }] } });
  expect(una.status()).toBe(400);
  const mala = await request.post('/api/combinar', { data: { fotos: [{ healthy: 1, rust: 1, red_spider_mite: 1 }, { healthy: 1, rust: 0, red_spider_mite: 0 }] } });
  expect(mala.status()).toBe(400);
  const seis = await request.post('/api/combinar', { data: { fotos: Array(6).fill({ healthy: 0.2, rust: 0.7, red_spider_mite: 0.1 }) } });
  expect(seis.status()).toBe(400);
  const texto = await request.post('/api/combinar', { data: 'esto no es json', headers: { 'Content-Type': 'application/json' } });
  expect(texto.status()).toBe(400);
  // Astro bloquea POST con tipos de formulario (text/plain) desde otro origen: protección contra CSRF
  const form = await request.post('/api/combinar', { data: 'x', headers: { 'Content-Type': 'text/plain' } });
  expect(form.status()).toBe(403);
});

// Debe ir al final: agota el límite de uso del servidor de pruebas.
test('la API limita las consultas repetidas', async ({ request }) => {
  const estados: number[] = [];
  for (let i = 0; i < 70; i += 1) {
    estados.push((await request.post('/api/diagnostico', { data: Buffer.from('x'), headers: { 'Content-Type': 'image/jpeg' } })).status());
  }
  expect(estados).toContain(429);
});
