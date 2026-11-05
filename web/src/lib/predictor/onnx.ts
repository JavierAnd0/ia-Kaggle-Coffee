/**
 * ADAPTADOR del modelo real: ejecuta la red exportada a ONNX con ONNX Runtime (Node).
 * Archivos esperados en `model/`: cafe_robusta.onnx, labels.json y model_card.json
 * (los genera notebooks/exportar_modelo.ipynb).
 */
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { CLASES, type Clase, type Predictor, type ResultadoPredictor } from './tipos';
import { LADO, prepararEntrada } from './preprocesar';

export const ARCHIVOS_MODELO = ['cafe_robusta.onnx', 'labels.json', 'model_card.json'] as const;

export function modeloDisponible(dir: string): boolean {
  return ARCHIVOS_MODELO.every((f) => existsSync(path.join(dir, f)));
}

export class PredictorOnnx implements Predictor {
  readonly nombre = 'onnx';
  readonly origen = 'modelo' as const;

  private constructor(
    private readonly sesion: import('onnxruntime-node').InferenceSession,
    private readonly ort: typeof import('onnxruntime-node'),
    private readonly etiquetas: Clase[],
    readonly version: string,
    private readonly lado: number,
  ) {}

  static async cargar(dir: string): Promise<PredictorOnnx> {
    if (!modeloDisponible(dir)) throw new Error(`Faltan archivos del modelo en ${dir}: ${ARCHIVOS_MODELO.join(', ')}`);
    const etiquetas = JSON.parse(readFileSync(path.join(dir, 'labels.json'), 'utf8')) as Clase[];
    if (etiquetas.length !== CLASES.length || !CLASES.every((c) => etiquetas.includes(c))) {
      throw new Error(`labels.json no coincide con las clases esperadas: ${JSON.stringify(etiquetas)}`);
    }
    const tarjeta = JSON.parse(readFileSync(path.join(dir, 'model_card.json'), 'utf8')) as { version?: string; input_size?: number[] };
    // El tamaño de entrada viene de la ficha del modelo (224 por defecto; un modelo reentrenado puede usar otro).
    const lado = tarjeta.input_size?.[0] ?? LADO;
    if (!Number.isInteger(lado) || lado < 64 || lado > 1024) throw new Error(`input_size inválido en model_card.json: ${lado}`);
    const ort = await import('onnxruntime-node');
    const sesion = await ort.InferenceSession.create(path.join(dir, 'cafe_robusta.onnx'));
    return new PredictorOnnx(sesion, ort, etiquetas, tarjeta.version ?? 'sin-version', lado);
  }

  async predecir(imagen: Uint8Array): Promise<ResultadoPredictor> {
    const entrada = await prepararEntrada(imagen, this.lado);
    const tensor = new this.ort.Tensor('float32', entrada, [1, this.lado, this.lado, 3]);
    const salida = await this.sesion.run({ [this.sesion.inputNames[0]]: tensor });
    const valores = salida[this.sesion.outputNames[0]].data as Float32Array;
    if (valores.length !== this.etiquetas.length) throw new Error('La salida del modelo no tiene 3 clases.');
    const suma = valores.reduce((a, b) => a + b, 0);
    if (Math.abs(suma - 1) > 0.02) throw new Error(`La salida del modelo no son probabilidades (suma ${suma.toFixed(3)}): ¿se exportó sin softmax?`);
    const probabilidades = Object.fromEntries(this.etiquetas.map((c, i) => [c, valores[i]])) as ResultadoPredictor['probabilidades'];
    return { probabilidades, origen: 'modelo', version: this.version };
  }
}
