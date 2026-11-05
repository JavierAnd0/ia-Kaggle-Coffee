/**
 * Combina los resultados de varias hojas de la misma planta (promedio simple de probabilidades).
 * Núcleo puro: sin E/S. Nota: NO se ha medido con datos reales que combinar mejore el acierto
 * (RoCoLe no tiene varias hojas por planta con la misma etiqueta); solo se evaluó en simulación.
 */
import { CLASES, type Clase, type Probabilidades } from './predictor/tipos';

export const MAX_FOTOS = 5;

export function promediar(lista: Probabilidades[]): Probabilidades {
  if (lista.length === 0) throw new Error('No hay resultados para combinar.');
  return Object.fromEntries(CLASES.map((c) => [c, lista.reduce((s, p) => s + p[c], 0) / lista.length])) as Probabilidades;
}

export function claseMayor(p: Probabilidades): Clase {
  return CLASES.reduce((a, c) => (p[c] > p[a] ? c : a), CLASES[0]);
}

/** ¿Todas las hojas apuntan a la misma clase? Si no, la interfaz avisa que no coinciden. */
export function coinciden(lista: Probabilidades[]): boolean {
  return new Set(lista.map(claseMayor)).size === 1;
}
