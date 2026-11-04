import type { Veredicto } from './decision';
import type { GuiaClase, Fuente } from './guia';
import type { Clase } from './predictor/tipos';

/** Contrato entre la API (servidor) y la isla de React (navegador). */
export interface RespuestaDiagnostico {
  modoSimulado: boolean;
  version: string;
  veredicto: Veredicto;
  porcentajes: Record<Clase, number>;
  /** Guía de la clase afirmada; `null` si el resultado no es concluyente. */
  guia: GuiaClase | null;
  fuentes: Fuente[];
  advertencia: string;
}

export interface ErrorApi {
  error: string; // mensaje listo para mostrar al productor
  codigo: 'imagen_invalida' | 'imagen_grande' | 'demasiadas_consultas' | 'servicio_no_disponible' | 'error';
}
