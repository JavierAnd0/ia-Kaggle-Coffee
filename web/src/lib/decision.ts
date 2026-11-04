/**
 * NÚCLEO PURO (sin E/S): convierte probabilidades del modelo en un veredicto prudente.
 *
 * Decisión de diseño: el modelo es modesto (exactitud 0,69 en prueba) y solo conoce 3 clases.
 * Por eso NO se afirma una enfermedad salvo que la probabilidad sea alta y la diferencia con la
 * segunda opción sea clara (con 3 clases, p ≥ 0,6 ya garantiza esa ventaja). En los demás casos se responde "no concluyente" y se pide otra foto.
 * Los umbrales NO están calibrados con datos reales: son un punto de partida conservador que
 * debe ajustarse con las probabilidades del conjunto de validación (ver docs/DISENO_WEB.md).
 */
import { CLASES, type Clase, type Probabilidades } from './predictor/tipos';

/**
 * Con 3 clases que suman 1, p ≥ 0,6 ya implica una ventaja de ≥ 0,2 sobre la segunda opción,
 * por eso basta un umbral mínimo (un umbral de «margen» aparte nunca tendría efecto).
 */
export const UMBRALES = { alta: 0.8, media: 0.6 } as const;

export type Nivel = 'alta' | 'media' | 'baja';

export interface Veredicto {
  /** Clase afirmada. `null` cuando el resultado no es concluyente. */
  clase: Clase | null;
  /** Clase más probable (aunque no sea concluyente); sirve para mostrar "lo más parecido". */
  claseProbable: Clase;
  nivel: Nivel;
  /** Probabilidad (0-100, entera) de `claseProbable`. */
  porcentaje: number;
  segunda: { clase: Clase; porcentaje: number };
}

function validar(p: Probabilidades): void {
  const suma = CLASES.reduce((acc, c) => acc + p[c], 0);
  if (CLASES.some((c) => !Number.isFinite(p[c]) || p[c] < 0 || p[c] > 1) || Math.abs(suma - 1) > 0.02) {
    throw new Error('Probabilidades inválidas: deben estar entre 0 y 1 y sumar 1.');
  }
}

/** Redondea a enteros que suman exactamente 100 (método del mayor resto). */
export function redondearPorcentajes(p: Probabilidades): Record<Clase, number> {
  validar(p);
  const base = CLASES.map((c) => ({ c, v: p[c] * 100, piso: Math.floor(p[c] * 100) }));
  let faltan = 100 - base.reduce((a, x) => a + x.piso, 0);
  [...base]
    .sort((a, b) => b.v - b.piso - (a.v - a.piso))
    .forEach((x) => {
      if (faltan > 0) {
        x.piso += 1;
        faltan -= 1;
      }
    });
  return Object.fromEntries(base.map((x) => [x.c, x.piso])) as Record<Clase, number>;
}

export function decidir(p: Probabilidades): Veredicto {
  validar(p);
  const pct = redondearPorcentajes(p);
  const orden = [...CLASES].sort((a, b) => p[b] - p[a]);
  const [primera, segunda] = orden;

  let nivel: Nivel = 'baja';
  if (p[primera] >= UMBRALES.alta) nivel = 'alta';
  else if (p[primera] >= UMBRALES.media) nivel = 'media';

  return {
    clase: nivel === 'baja' ? null : primera,
    claseProbable: primera,
    nivel,
    porcentaje: pct[primera],
    segunda: { clase: segunda, porcentaje: pct[segunda] },
  };
}
