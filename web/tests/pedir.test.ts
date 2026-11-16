import { describe, expect, it } from 'vitest';
import { MENSAJES, pedirJson } from '../src/lib/pedir';

const responder = (cuerpo: string, status = 200, tipo = 'application/json') =>
  (async () => new Response(cuerpo, { status, headers: { 'Content-Type': tipo } })) as unknown as typeof fetch;

describe('pedirJson', () => {
  it('devuelve los datos de una respuesta JSON correcta', async () => {
    expect(await pedirJson('/x', {}, 1000, responder('{"a":1}'))).toEqual({ ok: true, datos: { a: 1 } });
  });
  it('usa el mensaje de error de la API cuando viene en JSON', async () => {
    const r = await pedirJson('/x', {}, 1000, responder('{"codigo":"error","error":"Mensaje propio"}', 400));
    expect(r).toEqual({ ok: false, mensaje: 'Mensaje propio' });
  });
  it('una página HTML 413 o 502 del proxy NO se presenta como "sin conexión"', async () => {
    expect(await pedirJson('/x', {}, 1000, responder('<html>Too large</html>', 413, 'text/html'))).toEqual({ ok: false, mensaje: MENSAJES.pesada });
    expect(await pedirJson('/x', {}, 1000, responder('<html>Bad gateway</html>', 502, 'text/html'))).toEqual({ ok: false, mensaje: MENSAJES.noDisponible });
  });
  it('un fallo de red sí es "sin conexión"', async () => {
    const f = (async () => { throw new TypeError('Failed to fetch'); }) as unknown as typeof fetch;
    expect(await pedirJson('/x', {}, 1000, f)).toEqual({ ok: false, mensaje: MENSAJES.sinConexion });
  });
  it('si tarda más del límite, avisa de señal lenta', async () => {
    const f = ((_: string, init: RequestInit) => new Promise((_r, rechazar) => {
      init.signal?.addEventListener('abort', () => rechazar(new DOMException('abortado', 'AbortError')));
    })) as unknown as typeof fetch;
    expect(await pedirJson('/x', {}, 20, f)).toEqual({ ok: false, mensaje: MENSAJES.lento });
  });
});
