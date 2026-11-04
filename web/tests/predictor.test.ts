import { afterEach, describe, expect, it } from 'vitest';
import { obtenerPredictor, reiniciarPredictor } from '../src/lib/predictor';
import { PredictorSimulado } from '../src/lib/predictor/simulado';
import { CLASES } from '../src/lib/predictor/tipos';

afterEach(() => {
  delete process.env.PREDICTOR;
  delete process.env.MODEL_DIR;
  reiniciarPredictor();
});

describe('PredictorSimulado', () => {
  it('es determinístico y devuelve probabilidades válidas', async () => {
    const s = new PredictorSimulado();
    const a = await s.predecir(new Uint8Array([1, 2, 3, 4]));
    const b = await s.predecir(new Uint8Array([1, 2, 3, 4]));
    const c = await s.predecir(new Uint8Array([9, 9, 9]));
    expect(a).toEqual(b);
    expect(a.probabilidades).not.toEqual(c.probabilidades);
    expect(a.origen).toBe('simulado');
    expect(CLASES.reduce((x, k) => x + a.probabilidades[k], 0)).toBeCloseTo(1, 6);
  });
});

describe('obtenerPredictor', () => {
  it('auto sin modelo → simulado', async () => {
    process.env.MODEL_DIR = '/ruta/que/no/existe';
    expect((await obtenerPredictor()).origen).toBe('simulado');
  });
  it('modelo exigido y ausente → falla (nunca simula en silencio)', async () => {
    process.env.PREDICTOR = 'modelo';
    process.env.MODEL_DIR = '/ruta/que/no/existe';
    await expect(obtenerPredictor()).rejects.toThrow(/Faltan archivos/);
  });
  it('valor inválido → falla', async () => {
    process.env.PREDICTOR = 'otro';
    await expect(obtenerPredictor()).rejects.toThrow(/inválido/);
  });
});
