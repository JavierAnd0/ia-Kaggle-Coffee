import { describe, expect, it } from 'vitest';
import { decidir, redondearPorcentajes } from '../src/lib/decision';

const p = (healthy: number, rust: number, red_spider_mite: number) => ({ healthy, rust, red_spider_mite });

describe('redondearPorcentajes', () => {
  it('siempre suma 100', () => {
    for (const caso of [p(1 / 3, 1 / 3, 1 / 3), p(0.505, 0.255, 0.24), p(0.999, 0.0005, 0.0005), p(0.7, 0.2, 0.1)]) {
      const r = redondearPorcentajes(caso);
      expect(r.healthy + r.rust + r.red_spider_mite).toBe(100);
    }
  });
  it('rechaza probabilidades inválidas', () => {
    expect(() => redondearPorcentajes(p(0.5, 0.5, 0.5))).toThrow();
    expect(() => redondearPorcentajes(p(NaN, 0.5, 0.5))).toThrow();
    expect(() => redondearPorcentajes(p(-0.1, 0.6, 0.5))).toThrow();
  });
});

describe('decidir', () => {
  it('nivel alto cuando domina con claridad', () => {
    const v = decidir(p(0.05, 0.9, 0.05));
    expect(v).toMatchObject({ clase: 'rust', nivel: 'alta', porcentaje: 90 });
  });
  it('nivel medio', () => {
    const v = decidir(p(0.2, 0.65, 0.15));
    expect(v).toMatchObject({ clase: 'rust', nivel: 'media' });
  });
  it('no concluyente cuando hay dos opciones parecidas', () => {
    const v = decidir(p(0.1, 0.46, 0.44));
    expect(v.clase).toBeNull();
    expect(v.nivel).toBe('baja');
    expect(v.claseProbable).toBe('rust');
    expect(v.segunda.clase).toBe('red_spider_mite');
  });
  it('no concluyente si la probabilidad máxima es baja aunque gane', () => {
    expect(decidir(p(0.5, 0.3, 0.2)).clase).toBeNull();
  });
  it('los umbrales separan alta / media / baja', () => {
    expect(decidir(p(0.8, 0.2, 0)).nivel).toBe('alta');
    expect(decidir(p(0.79, 0.21, 0)).nivel).toBe('media');
    expect(decidir(p(0.6, 0.4, 0)).nivel).toBe('media');
    expect(decidir(p(0.59, 0.41, 0)).nivel).toBe('baja');
  });
  it('con 3 clases, un máximo ≥ 0,6 siempre le saca ≥ 0,2 a la segunda', () => {
    for (let a = 0.6; a <= 1; a += 0.01) {
      const resto = 1 - a;
      const v = decidir(p(a, resto, 0));
      expect(v.nivel).not.toBe('baja');
    }
  });
});
