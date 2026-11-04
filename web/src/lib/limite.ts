/** Límite simple de consultas por minuto por IP (memoria del proceso; suficiente para un servidor). */
const ventana = 60_000;
const registro = new Map<string, number[]>();

export function permitir(clave: string, maximo = 12, ahora = Date.now()): boolean {
  const recientes = (registro.get(clave) ?? []).filter((t) => ahora - t < ventana);
  if (recientes.length >= maximo) {
    registro.set(clave, recientes);
    return false;
  }
  recientes.push(ahora);
  registro.set(clave, recientes);
  if (registro.size > 5000) for (const [k, v] of registro) if (v.every((t) => ahora - t >= ventana)) registro.delete(k);
  return true;
}

export function reiniciarLimite(): void {
  registro.clear();
}

/** IP del cliente. Detrás de un proxy inverso (TRUST_PROXY=1) se toma la primera de X-Forwarded-For. */
export function claveCliente(cabeceras: Headers, direccion: string | undefined): string {
  if (process.env.TRUST_PROXY === '1') {
    const reenviada = cabeceras.get('x-forwarded-for')?.split(',')[0]?.trim();
    if (reenviada) return reenviada;
  }
  return direccion ?? 'desconocida';
}

export function limitePorMinuto(): number {
  const valor = Number(process.env.LIMITE_POR_MINUTO);
  return Number.isInteger(valor) && valor > 0 ? valor : 12;
}
