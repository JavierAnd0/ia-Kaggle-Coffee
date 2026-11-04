/**
 * Pruebas del modelo real. Se omiten si no hay modelo exportado (TEST_MODEL_DIR o ./model).
 * Con un modelo de pesos aleatorios solo validan la "tubería" (forma de entrada/salida).
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { modeloDisponible, PredictorOnnx } from '../src/lib/predictor/onnx';
import { CLASES } from '../src/lib/predictor/tipos';

const dir = process.env.TEST_MODEL_DIR ?? path.resolve(process.cwd(), 'model');
const hayModelo = modeloDisponible(dir);
const repo = path.resolve(process.cwd(), '..');

describe.skipIf(!hayModelo)('PredictorOnnx', () => {
  it('devuelve 3 probabilidades que suman 1 para una imagen JPEG', async () => {
    const predictor = await PredictorOnnx.cargar(dir);
    const jpeg = await sharp({ create: { width: 640, height: 360, channels: 3, background: { r: 60, g: 140, b: 60 } } }).jpeg().toBuffer();
    const r = await predictor.predecir(jpeg);
    expect(r.origen).toBe('modelo');
    expect(CLASES.reduce((a, c) => a + r.probabilidades[c], 0)).toBeCloseTo(1, 4);
  });
});

// Paridad con Keras: solo si el export generó paridad.json (imágenes reales del repo).
const archivoParidad = path.join(dir, 'paridad.json');
describe.skipIf(!hayModelo || !existsSync(archivoParidad))('paridad con Keras', () => {
  const casos = existsSync(archivoParidad)
    ? (JSON.parse(readFileSync(archivoParidad, 'utf8')) as { archivo: string; probabilidades: Record<string, number> }[])
    : [];

  it('la imagen completa da las mismas probabilidades que Keras (±0,02)', async () => {
    const predictor = await PredictorOnnx.cargar(dir);
    for (const caso of casos) {
      const r = await predictor.predecir(readFileSync(path.join(repo, caso.archivo)));
      for (const c of CLASES) expect(Math.abs(r.probabilidades[c] - caso.probabilidades[c])).toBeLessThan(0.02);
    }
  });

  it('la imagen reducida a 1024 px (como la envía el celular) cambia poco el resultado', async () => {
    const predictor = await PredictorOnnx.cargar(dir);
    let peor = 0;
    for (const caso of casos) {
      const reducida = await sharp(readFileSync(path.join(repo, caso.archivo))).resize({ width: 1024, height: 1024, fit: 'inside' }).jpeg({ quality: 85 }).toBuffer();
      const r = await predictor.predecir(reducida);
      for (const c of CLASES) peor = Math.max(peor, Math.abs(r.probabilidades[c] - caso.probabilidades[c]));
    }
    console.log(`Diferencia máxima con imagen reducida: ${peor.toFixed(3)}`);
    expect(peor).toBeLessThan(0.25);
  });
});
