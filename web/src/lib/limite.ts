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
