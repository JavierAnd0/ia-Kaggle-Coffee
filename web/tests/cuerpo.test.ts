import { afterEach, describe, expect, it } from 'vitest';
import { leerCuerpoLimitado } from '../src/lib/cuerpo';
import { reiniciarLimite } from '../src/lib/limite';
import { POST as combinar } from '../src/pages/api/combinar';

afterEach(() => reiniciarLimite());

/** Cuerpo en streaming sin Content-Length (como lo enviaría un cliente que lo omite). */
function enStreaming(partes: Uint8Array[]): Request {
  const cuerpo = new ReadableStream<Uint8Array>({
    start(c) { for (const p of partes) c.enqueue(p); c.close(); },
  });
  return new Request('http://x/api', { method: 'POST', body: cuerpo, duplex: 'half' } as RequestInit);
}

describe('lectura limitada del cuerpo', () => {
  it('devuelve los bytes si caben', async () => {
    const r = await leerCuerpoLimitado(enStreaming([new Uint8Array([1, 2]), new Uint8Array([3])]), 3);
    expect(Array.from(r!)).toEqual([1, 2, 3]);
  });
  it('corta al pasar el tope aunque no haya Content-Length', async () => {
    expect(await leerCuerpoLimitado(enStreaming([new Uint8Array(3), new Uint8Array(3)]), 5)).toBeNull();
  });
  it('rechaza de entrada un Content-Length mayor al tope', async () => {
    const req = new Request('http://x/api', { method: 'POST', body: 'abc', headers: { 'content-length': '999' } });
    expect(await leerCuerpoLimitado(req, 10)).toBeNull();
  });
  it('cuenta bytes, no caracteres (UTF-8 de varios bytes)', async () => {
    const req = new Request('http://x/api', { method: 'POST', body: 'ñ'.repeat(3) }); // 6 bytes, 3 caracteres
    expect(await leerCuerpoLimitado(req, 5)).toBeNull();
  });
});

describe('/api/combinar', () => {
  const llamar = (req: Request) => combinar({ request: req, clientAddress: '127.0.0.1' } as never) as Promise<Response>;

  it('responde 413 a un cuerpo grande sin leerlo entero', async () => {
    const r = await llamar(enStreaming([new TextEncoder().encode('{"fotos":[' + ' '.repeat(10_000) + ']}')]));
    expect(r.status).toBe(413);
  });
  it('responde 400 a un JSON inválido', async () => {
    const r = await llamar(new Request('http://x/api', { method: 'POST', body: '{no es json' }));
    expect(r.status).toBe(400);
  });
});
