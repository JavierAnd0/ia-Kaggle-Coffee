import { describe, expect, it } from 'vitest';
import { claseMayor, coinciden, promediar } from '../src/lib/combinar';
import { decidir } from '../src/lib/decision';
import { esProbabilidades } from '../src/pages/api/combinar';
import { armarRespuesta } from '../src/lib/respuesta';

const p = (healthy: number, rust: number, red_spider_mite: number) => ({ healthy, rust, red_spider_mite });

describe('combinar varias hojas', () => {
  it('promedia y sigue sumando 1', () => {
    const m = promediar([p(0.2, 0.7, 0.1), p(0.1, 0.5, 0.4)]);
    expect(m.healthy).toBeCloseTo(0.15, 6);
    expect(m.healthy + m.rust + m.red_spider_mite).toBeCloseTo(1, 6);
  });
  it('dos hojas que no coinciden bajan la seguridad', () => {
    const lista = [p(0.05, 0.9, 0.05), p(0.05, 0.05, 0.9)];
    expect(coinciden(lista)).toBe(false);
    expect(decidir(promediar(lista)).clase).toBeNull();      // ya no afirma nada
  });
  it('dos hojas que coinciden mantienen la conclusión', () => {
    const lista = [p(0.1, 0.7, 0.2), p(0.1, 0.8, 0.1)];
    expect(coinciden(lista)).toBe(true);
    expect(claseMayor(promediar(lista))).toBe('rust');
    expect(decidir(promediar(lista)).clase).toBe('rust');
  });
  it('armarRespuesta incluye el detalle por hoja solo si hay varias', () => {
    const una = armarRespuesta(p(0.1, 0.8, 0.1), 'simulado', 'x');
    expect(una.fotos).toBeUndefined();
    const varias = armarRespuesta(promediar([p(0.1, 0.8, 0.1), p(0.2, 0.7, 0.1)]), 'simulado', 'x', [p(0.1, 0.8, 0.1), p(0.2, 0.7, 0.1)]);
    expect(varias.fotos).toHaveLength(2);
    expect(varias.coinciden).toBe(true);
  });
  it('valida las probabilidades que llegan por la API', () => {
    expect(esProbabilidades(p(0.2, 0.3, 0.5))).toBe(true);
    expect(esProbabilidades(p(0.5, 0.5, 0.5))).toBe(false);
    expect(esProbabilidades({ healthy: 'a' })).toBe(false);
    expect(esProbabilidades(null)).toBe(false);
  });
});
