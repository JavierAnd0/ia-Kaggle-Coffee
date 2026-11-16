/**
 * Límite simple de consultas por minuto por IP (memoria del proceso; suficiente para un servidor).
 *
 * Cada ruta usa su propio "cupo" (`grupo`): /api/combinar no corre el modelo (solo promedia
 * probabilidades), así que NO gasta el cupo de /api/diagnostico. Revisar 3 hojas de una planta
 * cuesta 3 consultas de diagnóstico, no 3 + 2.
 */
const ventana = 60_000;
const registro = new Map<string, number[]>();

/** Por defecto 20 por minuto: en Colombia muchos celulares comparten IP (CGNAT de los operadores). */
export const LIMITE_POR_DEFECTO = 20;

export function permitir(clave: string, maximo = LIMITE_POR_DEFECTO, ahora = Date.now(), grupo = 'diagnostico'): boolean {
  const llave = `${grupo}:${clave}`;
  const recientes = (registro.get(llave) ?? []).filter((t) => ahora - t < ventana);
  if (recientes.length >= maximo) {
    registro.set(llave, recientes);
    return false;
  }
  recientes.push(ahora);
  registro.set(llave, recientes);
  if (registro.size > 5000) for (const [k, v] of registro) if (v.every((t) => ahora - t >= ventana)) registro.delete(k);
  return true;
}

export function reiniciarLimite(): void {
  registro.clear();
}

/**
 * IP del cliente.
 * - Sin TRUST_PROXY: la dirección de la conexión.
 * - TRUST_PROXY=1: el ÚLTIMO valor de X-Forwarded-For, que es el que agrega el proxy de la
 *   plataforma (Railway, Render, Fly…). Los primeros valores los puede inventar el cliente.
 * - CABECERA_IP=x-real-ip (o fly-client-ip, cf-connecting-ip…): usa esa cabecera, que el proxy
 *   escribe con la IP real. Solo tiene efecto con TRUST_PROXY=1.
 */
export function claveCliente(cabeceras: Headers, direccion: string | undefined): string {
  if (process.env.TRUST_PROXY === '1') {
    const nombre = process.env.CABECERA_IP?.trim().toLowerCase();
    if (nombre) {
      const valor = cabeceras.get(nombre)?.split(',').at(-1)?.trim();
      if (valor) return valor;
    } else {
      const ultima = cabeceras.get('x-forwarded-for')?.split(',').map((s) => s.trim()).filter(Boolean).at(-1);
      if (ultima) return ultima;
    }
  }
  return direccion ?? 'desconocida';
}

export function limitePorMinuto(): number {
  const valor = Number(process.env.LIMITE_POR_MINUTO);
  return Number.isInteger(valor) && valor > 0 ? valor : LIMITE_POR_DEFECTO;
}

/** /api/combinar es barato (no corre el modelo): cupo propio y más amplio. */
export function limiteCombinarPorMinuto(): number {
  return limitePorMinuto() * 3;
}
