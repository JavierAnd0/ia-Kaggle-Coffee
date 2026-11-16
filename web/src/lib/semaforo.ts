/**
 * Semáforo mínimo: deja pasar a lo sumo `maximo` tareas a la vez; las demás esperan en orden.
 * Se usa alrededor del modelo para que varias fotos simultáneas no agoten la memoria ni la CPU
 * (cada foto decodificada ocupa decenas de MB y ONNX Runtime ya usa varios hilos por inferencia).
 */
export class Semaforo {
  private activos = 0;
  private readonly cola: (() => void)[] = [];

  constructor(readonly maximo: number) {
    if (!Number.isInteger(maximo) || maximo < 1) throw new Error(`Semáforo inválido: ${maximo}`);
  }

  get enEspera(): number {
    return this.cola.length;
  }

  async usar<T>(tarea: () => Promise<T>): Promise<T> {
    if (this.activos >= this.maximo) await new Promise<void>((r) => this.cola.push(r));
    else this.activos += 1;
    try {
      return await tarea();
    } finally {
      const siguiente = this.cola.shift();
      if (siguiente) siguiente(); // el cupo pasa directo al siguiente (activos no cambia)
      else this.activos -= 1;
    }
  }
}

/** INFERENCIAS_SIMULTANEAS (1–4, por defecto 2). */
export function inferenciasSimultaneas(): number {
  const valor = Number(process.env.INFERENCIAS_SIMULTANEAS);
  return Number.isInteger(valor) && valor >= 1 && valor <= 4 ? valor : 2;
}
