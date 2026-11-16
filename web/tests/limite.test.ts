import { afterEach, describe, expect, it } from 'vitest';
import { claveCliente, LIMITE_POR_DEFECTO, limitePorMinuto, permitir, reiniciarLimite } from '../src/lib/limite';

afterEach(() => {
  delete process.env.TRUST_PROXY;
  delete process.env.CABECERA_IP;
  delete process.env.LIMITE_POR_MINUTO;
  reiniciarLimite();
});

describe('límite de uso', () => {
  it('bloquea al superar el máximo dentro de la ventana y libera después', () => {
    const t0 = 1_000_000;
    for (let i = 0; i < 3; i += 1) expect(permitir('a', 3, t0 + i)).toBe(true);
    expect(permitir('a', 3, t0 + 10)).toBe(false);
    expect(permitir('b', 3, t0 + 10)).toBe(true);          // otro cliente no se afecta
    expect(permitir('a', 3, t0 + 61_000)).toBe(true);      // pasó el minuto
  });

  it('cada grupo (diagnóstico / combinar) tiene su propio cupo', () => {
    const t0 = 1_000_000;
    for (let i = 0; i < 2; i += 1) expect(permitir('a', 2, t0 + i)).toBe(true);
    expect(permitir('a', 2, t0 + 5)).toBe(false);
    expect(permitir('a', 2, t0 + 5, 'combinar')).toBe(true);
  });

  it('límite por defecto y configurable', () => {
    expect(limitePorMinuto()).toBe(LIMITE_POR_DEFECTO);
    process.env.LIMITE_POR_MINUTO = '7';
    expect(limitePorMinuto()).toBe(7);
  });

  it('usa la IP reenviada solo si se confía en el proxy', () => {
    const cab = new Headers({ 'x-forwarded-for': '203.0.113.7, 10.0.0.1' });
    expect(claveCliente(cab, '10.0.0.1')).toBe('10.0.0.1');
    process.env.TRUST_PROXY = '1';
    expect(claveCliente(cab, '10.0.0.1')).toBe('10.0.0.1');
  });

  it('con proxy toma el ÚLTIMO salto de X-Forwarded-For (el cliente no puede falsificarlo)', () => {
    process.env.TRUST_PROXY = '1';
    // El cliente envía "1.2.3.4" inventado; el proxy de la plataforma agrega la IP real al final.
    const cab = new Headers({ 'x-forwarded-for': '1.2.3.4, 198.51.100.20' });
    expect(claveCliente(cab, '10.0.0.1')).toBe('198.51.100.20');
    expect(claveCliente(new Headers({ 'x-forwarded-for': '198.51.100.20' }), '10.0.0.1')).toBe('198.51.100.20');
    expect(claveCliente(new Headers({ 'x-forwarded-for': ' , ' }), '10.0.0.1')).toBe('10.0.0.1');
  });

  it('CABECERA_IP permite usar otra cabecera (X-Real-IP, Fly-Client-IP)', () => {
    process.env.TRUST_PROXY = '1';
    process.env.CABECERA_IP = 'Fly-Client-IP';
    const cab = new Headers({ 'x-forwarded-for': '1.2.3.4, 5.6.7.8', 'fly-client-ip': '192.0.2.9' });
    expect(claveCliente(cab, '10.0.0.1')).toBe('192.0.2.9');
    expect(claveCliente(new Headers({ 'x-forwarded-for': '5.6.7.8' }), '10.0.0.1')).toBe('10.0.0.1');
    delete process.env.TRUST_PROXY;
    expect(claveCliente(cab, '10.0.0.1')).toBe('10.0.0.1');
  });
});
