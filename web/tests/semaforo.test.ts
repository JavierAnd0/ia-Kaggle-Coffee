import { describe, expect, it } from 'vitest';
import { Semaforo } from '../src/lib/semaforo';

describe('semáforo de inferencia', () => {
  it('no deja pasar más de `maximo` tareas a la vez y atiende en orden', async () => {
    const s = new Semaforo(2);
    let activas = 0;
    let pico = 0;
    const orden: number[] = [];
    const tarea = (i: number) => s.usar(async () => {
      activas += 1; pico = Math.max(pico, activas);
      await new Promise((r) => setTimeout(r, 5));
      orden.push(i); activas -= 1;
      return i;
    });
    expect(await Promise.all([0, 1, 2, 3, 4].map(tarea))).toEqual([0, 1, 2, 3, 4]);
    expect(pico).toBe(2);
    expect(orden.slice(2)).toEqual([2, 3, 4]);
  });

  it('libera el cupo aunque la tarea falle', async () => {
    const s = new Semaforo(1);
    await expect(s.usar(async () => { throw new Error('x'); })).rejects.toThrow('x');
    expect(await s.usar(async () => 7)).toBe(7);
  });
});
