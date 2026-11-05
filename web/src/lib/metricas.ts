/**
 * Métricas que muestra la web. Base: src/data/metricas.json. Si junto al modelo hay un
 * `model/metricas_web.json` (lo genera notebooks/experimentos_mejora.ipynb con las cifras reales del
 * modelo exportado), sus valores reemplazan a los de la base: así las cifras de la página nunca
 * quedan desactualizadas respecto al modelo que está sirviendo.
 */
import base from '../data/metricas.json';

const modulos = import.meta.glob('../../model/metricas_web.json', { eager: true, import: 'default' }) as Record<string, Partial<typeof base>>;
const extra = Object.values(modulos)[0] ?? {};

export const metricas: typeof base = { ...base, ...extra };
export const metricasDelModelo = Object.keys(extra).length > 0;
