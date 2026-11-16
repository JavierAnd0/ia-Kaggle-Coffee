import { decidir, redondearPorcentajes } from './decision';
import { claseMayor, coinciden } from './combinar';
import { AVISO_ROBUSTA, base, fuentesDe, obtenerGuia } from './guia';
import type { Probabilidades } from './predictor/tipos';
import type { RespuestaDiagnostico } from './tipos';

/** Arma la respuesta completa (veredicto + guía) a partir de probabilidades. La usan /api/diagnostico y /api/combinar. */
export function armarRespuesta(
  probabilidades: Probabilidades,
  origen: 'modelo' | 'simulado',
  version: string,
  porFoto?: Probabilidades[],
): RespuestaDiagnostico {
  const veredicto = decidir(probabilidades);
  const guia = veredicto.clase ? obtenerGuia(veredicto.clase) : null;
  return {
    modoSimulado: origen === 'simulado',
    version,
    veredicto,
    probabilidades,
    porcentajes: redondearPorcentajes(probabilidades),
    guia,
    fuentes: guia ? fuentesDe(guia.fuentes) : [],
    advertencia: `${base.advertencia_general} ${AVISO_ROBUSTA}`,
    ...(porFoto && porFoto.length > 1
      ? {
          fotos: porFoto.map((p) => ({ claseProbable: claseMayor(p), porcentaje: redondearPorcentajes(p)[claseMayor(p)] })),
          coinciden: coinciden(porFoto),
        }
      : {}),
  };
}
