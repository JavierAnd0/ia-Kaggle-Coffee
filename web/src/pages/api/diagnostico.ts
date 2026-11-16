import type { APIRoute } from 'astro';
import { leerCuerpoLimitado } from '../../lib/cuerpo';
import { claveCliente, limitePorMinuto, permitir } from '../../lib/limite';
import { obtenerPredictor } from '../../lib/predictor';
import { ImagenInvalidaError } from '../../lib/predictor/preprocesar';
import { armarRespuesta } from '../../lib/respuesta';
import type { ErrorApi, RespuestaDiagnostico } from '../../lib/tipos';

export const prerender = false;

const MAX_BYTES = 6 * 1024 * 1024;

/** Verifica los primeros bytes (no se confía en el Content-Type que envía el navegador). */
export function esImagenSoportada(b: Uint8Array): boolean {
  const jpeg = b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff;
  const png = b[0] === 0x89 && b[1] === 0x50 && b[2] === 0x4e && b[3] === 0x47;
  const webp = b.length > 12 && String.fromCharCode(...b.slice(0, 4)) === 'RIFF' && String.fromCharCode(...b.slice(8, 12)) === 'WEBP';
  return jpeg || png || webp;
}

const json = (cuerpo: RespuestaDiagnostico | ErrorApi, status = 200) =>
  new Response(JSON.stringify(cuerpo), { status, headers: { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' } });

const fallo = (codigo: ErrorApi['codigo'], error: string, status: number) => json({ codigo, error }, status);

export const POST: APIRoute = async ({ request, clientAddress }) => {
  let direccion: string | undefined;
  try { direccion = clientAddress; } catch { /* sin dirección (pruebas) */ }
  if (!permitir(claveCliente(request.headers, direccion), limitePorMinuto())) return fallo('demasiadas_consultas', 'Ha enviado muchas fotos seguidas. Espere un minuto e intente de nuevo.', 429);

  const bytes = await leerCuerpoLimitado(request, MAX_BYTES);
  if (!bytes) return fallo('imagen_grande', 'La foto es muy pesada. Intente con otra foto.', 413);
  if (!esImagenSoportada(bytes)) return fallo('imagen_invalida', 'No pudimos leer la foto. Use una foto en formato JPG, PNG o WEBP.', 400);

  let resultado;
  try {
    resultado = await (await obtenerPredictor()).predecir(bytes);
  } catch (error) {
    console.error('Fallo del predictor:', error);
    return error instanceof ImagenInvalidaError
      ? fallo('imagen_invalida', 'No pudimos leer la foto. Intente con otra.', 400)
      : fallo('servicio_no_disponible', 'El servicio no está disponible en este momento. Intente más tarde.', 503);
  }

  const cuerpo = armarRespuesta(resultado.probabilidades, resultado.origen, resultado.version);
  return json(cuerpo);
};
