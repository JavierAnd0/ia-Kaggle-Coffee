/**
 * ADAPTADOR de prueba: genera probabilidades determinísticas a partir de los bytes de la imagen
 * (misma foto → mismo resultado). NO analiza la hoja. Existe para desarrollar y probar la
 * interfaz mientras no haya un modelo entrenado; la web lo muestra siempre como "modo de prueba".
 */
import { CLASES, type Predictor, type ResultadoPredictor } from './tipos';

function fnv1a(bytes: Uint8Array): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i += 1) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

function mulberry32(semilla: number): () => number {
  let a = semilla;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class PredictorSimulado implements Predictor {
  readonly nombre = 'simulado';
  readonly version = 'simulado-0';
  readonly origen = 'simulado' as const;

  async predecir(imagen: Uint8Array): Promise<ResultadoPredictor> {
    const azar = mulberry32(fnv1a(imagen));
    const logits = CLASES.map(() => (azar() - 0.5) * 6);
    const max = Math.max(...logits);
    const exp = logits.map((l) => Math.exp(l - max));
    const suma = exp.reduce((a, b) => a + b, 0);
    const probabilidades = Object.fromEntries(CLASES.map((c, i) => [c, exp[i] / suma])) as ResultadoPredictor['probabilidades'];
    return { probabilidades, origen: 'simulado', version: 'simulado-0' };
  }
}
