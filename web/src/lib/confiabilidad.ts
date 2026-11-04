/** Qué tanto acertó el programa en las fotos de prueba según su nivel de seguridad (datos reales de metricas.json). */
import metricas from '../data/metricas.json';
import type { Nivel } from './decision';

export function fraseConfiabilidad(nivel: Nivel): string {
  const de10 = Math.round(metricas.niveles[nivel].prueba * 10);
  const cuando = nivel === 'baja' ? 'cuando no estaba seguro' : `cuando su seguridad fue ${nivel}`;
  return `En las fotos de prueba, ${cuando}, acertó unas ${de10} de cada 10 veces.`;
}
