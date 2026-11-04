import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';
import path from 'node:path';

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

  await page.getByRole('button', { name: /Revisar otra hoja/ }).click();
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

test('la API limita las consultas repetidas', async ({ request }) => {
  const estados: number[] = [];
  for (let i = 0; i < 40; i += 1) {
    estados.push((await request.post('/api/diagnostico', { data: Buffer.from('x'), headers: { 'Content-Type': 'image/jpeg' } })).status());
  }
  expect(estados).toContain(429);
});
