import type { APIRoute } from 'astro';
import { MAX_FOTOS, promediar } from '../../lib/combinar';
import { claveCliente, limitePorMinuto, permitir } from '../../lib/limite';
import { obtenerPredictor } from '../../lib/predictor';
import { CLASES, type Probabilidades } from '../../lib/predictor/tipos';
import { armarRespuesta } from '../../lib/respuesta';
import type { ErrorApi } from '../../lib/tipos';

export const prerender = false;

const MAX_BYTES = 4096;

const cabeceras = { 'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store' };
const fallo = (codigo: ErrorApi['codigo'], error: string, status: number) =>
  new Response(JSON.stringify({ codigo, error } satisfies ErrorApi), { status, headers: cabeceras });

export function esProbabilidades(x: unknown): x is Probabilidades {
  if (typeof x !== 'object' || x === null) return false;
  const p = x as Record<string, unknown>;
  const valores = CLASES.map((c) => p[c]);
  return valores.every((v) => typeof v === 'number' && Number.isFinite(v) && v >= 0 && v <= 1) &&
    Math.abs((valores as number[]).reduce((a, b) => a + b, 0) - 1) <= 0.02;
}

/**
 * Combina los resultados de varias hojas. Recibe SOLO probabilidades (no fotos), por eso no guarda
 * ni procesa imágenes. La decisión y la guía se arman en el servidor, igual que con una sola foto.
 */
export const POST: APIRoute = async ({ request, clientAddress }) => {
  let direccion: string | undefined;
  try { direccion = clientAddress; } catch { /* pruebas */ }
  if (!permitir(claveCliente(request.headers, direccion), limitePorMinuto())) {
    return fallo('demasiadas_consultas', 'Ha enviado muchas consultas seguidas. Espere un minuto e intente de nuevo.', 429);
  }
  const texto = await request.text();
  if (texto.length > MAX_BYTES) return fallo('imagen_grande', 'La solicitud es demasiado grande.', 413);
  let cuerpo: unknown;
  try { cuerpo = JSON.parse(texto); } catch { return fallo('error', 'No pudimos leer la solicitud.', 400); }
  const lista = (cuerpo as { fotos?: unknown })?.fotos;
  if (!Array.isArray(lista) || lista.length < 2 || lista.length > MAX_FOTOS || !lista.every(esProbabilidades)) {
    return fallo('error', `Envíe entre 2 y ${MAX_FOTOS} resultados válidos.`, 400);
  }
  try {
    const predictor = await obtenerPredictor();
    const respuesta = armarRespuesta(promediar(lista), predictor.origen, predictor.version, lista);
    return new Response(JSON.stringify(respuesta), { headers: cabeceras });
  } catch (error) {
    console.error('Fallo al combinar:', error);
    return fallo('servicio_no_disponible', 'El servicio no está disponible en este momento. Intente más tarde.', 503);
  }
};
