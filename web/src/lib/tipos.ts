import type { Veredicto } from './decision';
import type { GuiaClase, Fuente } from './guia';
import type { Clase, Probabilidades } from './predictor/tipos';

/** Contrato entre la API (servidor) y la isla de React (navegador). */
export interface RespuestaDiagnostico {
  modoSimulado: boolean;
  version: string;
  veredicto: Veredicto;
  /** Probabilidades sin redondear (para combinar varias hojas). */
  probabilidades: Probabilidades;
  porcentajes: Record<Clase, number>;
  /** Solo al combinar varias hojas: resultado de cada una y si coinciden. */
  fotos?: { claseProbable: Clase; porcentaje: number }[];
  coinciden?: boolean;
  /** Guía de la clase afirmada; `null` si el resultado no es concluyente. */
  guia: GuiaClase | null;
  fuentes: Fuente[];
  advertencia: string;
}

export interface ErrorApi {
  error: string; // mensaje listo para mostrar al productor
  codigo: 'imagen_invalida' | 'imagen_grande' | 'demasiadas_consultas' | 'servicio_no_disponible' | 'error';
}
