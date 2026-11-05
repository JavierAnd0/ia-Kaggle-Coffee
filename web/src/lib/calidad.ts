/**
 * Control de calidad de la foto, 100 % en el navegador y antes de enviarla.
 * Trabaja sobre una versión reducida en gris (los umbrales se calibraron con las 1560 fotos de
 * RoCoLe y con versiones desenfocadas/oscurecidas de ellas; ver tests/calidad.test.ts).
 */
export const LADO_CALIDAD = 256;

export interface Calidad {
  borrosa: boolean;
  oscura: boolean;
  muyClara: boolean;
  nitidez: number;
  luminancia: number;
}

/**
 * Umbrales calibrados con RoCoLe (520 fotos) y con versiones degradadas de ellas, a 256 px:
 *   nitidez   fotos buenas p1 = 524, mediana 1896; desenfoque fuerte (σ=4) mediana 284 → umbral 250
 *   luminancia fotos buenas 81–145; oscurecidas ×0,35 mediana 41; aclaradas +120 mediana 223 → 60 y 200
 * Limitación: RoCoLe tiene fondo de tierra con mucha textura; una hoja sobre fondo liso da menos
 * «nitidez» sin estar borrosa. Por eso el aviso es suave (nunca bloquea) y el umbral es bajo.
 */
export const UMBRALES_CALIDAD = { nitidezMinima: 250, luminanciaMinima: 60, luminanciaMaxima: 200 };

/** Pasa RGBA a gris (luminancia) en un arreglo de ancho*alto. */
export function aGris(rgba: Uint8ClampedArray | Uint8Array, ancho: number, alto: number): Float32Array {
  const gris = new Float32Array(ancho * alto);
  for (let i = 0; i < gris.length; i += 1) {
    gris[i] = 0.299 * rgba[4 * i] + 0.587 * rgba[4 * i + 1] + 0.114 * rgba[4 * i + 2];
  }
  return gris;
}

/** Varianza del Laplaciano: baja = imagen sin detalle fino (borrosa). */
export function nitidezLaplaciano(gris: Float32Array, ancho: number, alto: number): number {
  let suma = 0;
  let suma2 = 0;
  let n = 0;
  for (let y = 1; y < alto - 1; y += 1) {
    for (let x = 1; x < ancho - 1; x += 1) {
      const i = y * ancho + x;
      const l = gris[i - 1] + gris[i + 1] + gris[i - ancho] + gris[i + ancho] - 4 * gris[i];
      suma += l;
      suma2 += l * l;
      n += 1;
    }
  }
  const media = suma / n;
  return suma2 / n - media * media;
}

export function luminanciaMedia(gris: Float32Array): number {
  let s = 0;
  for (let i = 0; i < gris.length; i += 1) s += gris[i];
  return s / gris.length;
}

export function evaluarCalidad(rgba: Uint8ClampedArray | Uint8Array, ancho: number, alto: number): Calidad {
  const gris = aGris(rgba, ancho, alto);
  const nitidez = nitidezLaplaciano(gris, ancho, alto);
  const luminancia = luminanciaMedia(gris);
  return {
    nitidez,
    luminancia,
    borrosa: nitidez < UMBRALES_CALIDAD.nitidezMinima,
    oscura: luminancia < UMBRALES_CALIDAD.luminanciaMinima,
    muyClara: luminancia > UMBRALES_CALIDAD.luminanciaMaxima,
  };
}

export function hayProblema(c: Calidad): boolean {
  return c.borrosa || c.oscura || c.muyClara;
}
