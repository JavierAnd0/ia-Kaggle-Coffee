import { readdirSync } from 'node:fs';
import path from 'node:path';
import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { evaluarCalidad, hayProblema, LADO_CALIDAD } from '../src/lib/calidad';

async function calidadDe(buf: Buffer | string) {
  const { data, info } = await sharp(buf).rotate().resize(LADO_CALIDAD, LADO_CALIDAD, { fit: 'inside' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  return evaluarCalidad(new Uint8Array(data), info.width, info.height);
}
const rgba = (valor: number, n = 64) => new Uint8Array(n * n * 4).fill(valor);

describe('calidad de la foto (sintética)', () => {
  it('una imagen sin detalle se marca borrosa', () => expect(evaluarCalidad(rgba(120), 64, 64).borrosa).toBe(true));
  it('una imagen muy oscura se marca oscura', () => expect(evaluarCalidad(rgba(20), 64, 64).oscura).toBe(true));
  it('una imagen casi blanca se marca muy clara', () => expect(evaluarCalidad(rgba(245), 64, 64).muyClara).toBe(true));
  it('una imagen con mucho detalle y luz normal no tiene problemas', () => {
    const r = new Uint8Array(64 * 64 * 4);
    for (let i = 0; i < 64 * 64; i += 1) { const v = (i * 7919) % 200 + 20; r.set([v, v, v, 255], 4 * i); }
    expect(hayProblema(evaluarCalidad(r, 64, 64))).toBe(false);
  });
});

// Con fotos reales del conjunto de datos (se omite si no están descargadas)
const raiz = path.resolve(process.cwd(), '..', 'data', 'coffee___healthy');
const fotos = (() => { try { return readdirSync(raiz).slice(0, 12).map((f) => path.join(raiz, f)); } catch { return []; } })();
describe.skipIf(fotos.length === 0)('calidad con fotos reales de RoCoLe', () => {
  it('las fotos buenas casi nunca se marcan (a lo sumo 1 de 12)', async () => {
    let marcadas = 0;
    for (const f of fotos) if (hayProblema(await calidadDe(f))) marcadas += 1;
    expect(marcadas).toBeLessThanOrEqual(1);
  });
  it('desenfoque fuerte (σ=5), oscurecer y aclarar se detectan en la mayoría', async () => {
    const casos: [string, (s: sharp.Sharp) => sharp.Sharp][] = [
      ['borrosa', (s) => s.blur(5)], ['oscura', (s) => s.linear(0.3, 0)], ['muyClara', (s) => s.linear(1, 130)],
    ];
    for (const [campo, f] of casos) {
      let detectadas = 0;
      for (const foto of fotos) {
        const buf = await f(sharp(foto).rotate().resize(1280, 720, { fit: 'inside' })).jpeg().toBuffer();
        if ((await calidadDe(buf))[campo as 'borrosa' | 'oscura' | 'muyClara']) detectadas += 1;
      }
      expect(detectadas, campo).toBeGreaterThanOrEqual(Math.ceil(fotos.length * 0.8));
    }
  });
});
