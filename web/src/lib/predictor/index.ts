/**
 * FÁBRICA del predictor (patrón Factory + Singleton perezoso).
 *   PREDICTOR=auto      (por defecto) usa el modelo real si está en `model/`; si no, el simulado.
 *   PREDICTOR=modelo    EXIGE el modelo real: si falta, falla (recomendado en producción).
 *   PREDICTOR=simulado  fuerza el modo de prueba.
 *   MODEL_DIR=ruta      cambia la carpeta del modelo (por defecto ./model).
 */
import path from 'node:path';
import { modeloDisponible, PredictorOnnx } from './onnx';
import { PredictorSimulado } from './simulado';
import type { Predictor } from './tipos';

let instancia: Promise<Predictor> | undefined;

export function carpetaModelo(): string {
  return process.env.MODEL_DIR ? path.resolve(process.env.MODEL_DIR) : path.resolve(process.cwd(), 'model');
}

async function crear(): Promise<Predictor> {
  const modo = (process.env.PREDICTOR ?? 'auto').toLowerCase();
  const dir = carpetaModelo();
  if (modo === 'simulado') return new PredictorSimulado();
  if (modo === 'modelo') return PredictorOnnx.cargar(dir);
  if (modo === 'auto') return modeloDisponible(dir) ? PredictorOnnx.cargar(dir) : new PredictorSimulado();
  throw new Error(`PREDICTOR inválido: "${modo}". Use auto, modelo o simulado.`);
}

export function obtenerPredictor(): Promise<Predictor> {
  instancia ??= crear().catch((error) => {
    instancia = undefined; // permite reintentar tras corregir la configuración
    throw error;
  });
  return instancia;
}

/** Solo para pruebas. */
export function reiniciarPredictor(): void {
  instancia = undefined;
}
