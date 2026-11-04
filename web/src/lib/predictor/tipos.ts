/**
 * PUERTO (Ports & Adapters): contrato que cualquier "motor" de predicción debe cumplir.
 * La web solo conoce esta interfaz; no sabe si detrás hay un modelo real o uno simulado.
 */
export type Clase = 'healthy' | 'rust' | 'red_spider_mite';

/** Mismo orden que `labels.json` del modelo exportado (orden alfabético del entrenamiento). */
export const CLASES: readonly Clase[] = ['healthy', 'red_spider_mite', 'rust'] as const;

export type Probabilidades = Record<Clase, number>;

export interface ResultadoPredictor {
  probabilidades: Probabilidades;
  /** 'modelo' = red neuronal entrenada; 'simulado' = datos de prueba (NO son reales). */
  origen: 'modelo' | 'simulado';
  version: string;
}

export interface Predictor {
  readonly nombre: string;
  readonly origen: ResultadoPredictor['origen'];
  predecir(imagen: Uint8Array): Promise<ResultadoPredictor>;
}
