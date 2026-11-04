import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import metricas from '../src/data/metricas.json';

// Si hay un modelo exportado, las cifras que muestra la página «Acerca de» no deben quedar desactualizadas.
const tarjeta = path.resolve(process.cwd(), 'model', 'model_card.json');
describe.skipIf(!existsSync(tarjeta))('metricas.json vs model_card.json', () => {
  it('la exactitud en prueba mostrada coincide con la del modelo exportado (±0,02)', () => {
    const card = JSON.parse(readFileSync(tarjeta, 'utf8')) as { metrics?: { test?: { accuracy?: number } } };
    const real = card.metrics?.test?.accuracy;
    expect(real, 'model_card.json no trae metrics.test.accuracy').toBeTypeOf('number');
    expect(Math.abs((real as number) - metricas.prueba.exactitud)).toBeLessThanOrEqual(0.02);
  });
});
