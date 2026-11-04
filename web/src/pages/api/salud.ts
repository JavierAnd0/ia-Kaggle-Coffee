import type { APIRoute } from 'astro';
import { obtenerPredictor } from '../../lib/predictor';

export const prerender = false;

/** Indica si el servicio usa el modelo real o el modo de prueba (la interfaz lo muestra). */
export const GET: APIRoute = async () => {
  try {
    const p = await obtenerPredictor();
    return new Response(JSON.stringify({ ok: true, predictor: p.nombre, modoSimulado: p.origen === 'simulado' }), {
      headers: { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' },
    });
  } catch (error) {
    console.error('Salud: el predictor no carga:', error);
    return new Response(JSON.stringify({ ok: false }), { status: 503, headers: { 'Content-Type': 'application/json' } });
  }
};
