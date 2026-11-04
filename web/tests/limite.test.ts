import { afterEach, describe, expect, it } from 'vitest';
import { claveCliente, permitir, reiniciarLimite } from '../src/lib/limite';

afterEach(() => { delete process.env.TRUST_PROXY; reiniciarLimite(); });

describe('límite de uso', () => {
  it('bloquea al superar el máximo dentro de la ventana y libera después', () => {
    const t0 = 1_000_000;
    for (let i = 0; i < 3; i += 1) expect(permitir('a', 3, t0 + i)).toBe(true);
    expect(permitir('a', 3, t0 + 10)).toBe(false);
    expect(permitir('b', 3, t0 + 10)).toBe(true);          // otro cliente no se afecta
    expect(permitir('a', 3, t0 + 61_000)).toBe(true);      // pasó el minuto
  });
  it('usa la IP reenviada solo si se confía en el proxy', () => {
    const cab = new Headers({ 'x-forwarded-for': '203.0.113.7, 10.0.0.1' });
    expect(claveCliente(cab, '10.0.0.1')).toBe('10.0.0.1');
    process.env.TRUST_PROXY = '1';
    expect(claveCliente(cab, '10.0.0.1')).toBe('203.0.113.7');
  });
});
