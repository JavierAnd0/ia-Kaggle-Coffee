/**
 * Llamadas a la API desde el navegador con tiempo límite y mensajes claros.
 * En el campo la señal es débil: sin tiempo límite la pantalla podía quedarse "revisando" para
 * siempre. Y un proxy puede responder HTML (413, 502…) en vez de JSON: eso no es "sin conexión".
 */
import type { ErrorApi } from './tipos';

export const TIEMPO_LIMITE_MS = 30_000;

export const MENSAJES = {
  lento: 'La señal está muy lenta y la revisión tardó demasiado. Busque un lugar con mejor señal e intente de nuevo.',
  sinConexion: 'No hay conexión con el servicio. Revise su señal e intente de nuevo.',
  pesada: 'La foto es muy pesada. Intente con otra foto.',
  muchas: 'Ha enviado muchas fotos seguidas. Espere un minuto e intente de nuevo.',
  noDisponible: 'El servicio no está disponible en este momento. Intente más tarde.',
  general: 'Algo salió mal. Intente de nuevo.',
} as const;

export type Resultado<T> = { ok: true; datos: T } | { ok: false; mensaje: string };

/** Mensaje cuando la respuesta no trae un error JSON nuestro (p. ej. página HTML del proxy). */
export function mensajePorEstado(status: number): string {
  if (status === 413) return MENSAJES.pesada;
  if (status === 429) return MENSAJES.muchas;
  if (status === 502 || status === 503 || status === 504) return MENSAJES.noDisponible;
  return MENSAJES.general;
}

export async function pedirJson<T>(url: string, init: RequestInit, ms = TIEMPO_LIMITE_MS, f: typeof fetch = fetch): Promise<Resultado<T>> {
  const control = new AbortController();
  const reloj = setTimeout(() => control.abort(), ms);
  let respuesta: Response;
  let texto: string;
  try {
    respuesta = await f(url, { ...init, signal: control.signal });
    texto = await respuesta.text();
  } catch (error) {
    return { ok: false, mensaje: control.signal.aborted || (error as Error)?.name === 'AbortError' ? MENSAJES.lento : MENSAJES.sinConexion };
  } finally {
    clearTimeout(reloj);
  }
  let cuerpo: unknown = null;
  try { cuerpo = JSON.parse(texto); } catch { /* no es JSON: se usa el código HTTP */ }
  if (!respuesta.ok) {
    const error = (cuerpo as ErrorApi | null)?.error;
    return { ok: false, mensaje: typeof error === 'string' && error ? error : mensajePorEstado(respuesta.status) };
  }
  if (cuerpo === null || typeof cuerpo !== 'object') return { ok: false, mensaje: MENSAJES.general };
  return { ok: true, datos: cuerpo as T };
}
