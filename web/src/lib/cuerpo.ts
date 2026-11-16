/**
 * Lectura del cuerpo de una solicitud con tope en BYTES. Lee por partes y se detiene al pasar el
 * tope, sin confiar en Content-Length (un cliente puede omitirlo o mentir). `null` = demasiado grande.
 */
export async function leerCuerpoLimitado(request: Request, maximo: number): Promise<Uint8Array | null> {
  const declarado = Number(request.headers.get('content-length') ?? 0);
  if (Number.isFinite(declarado) && declarado > maximo) return null;
  if (!request.body) return new Uint8Array(0);
  const lector = request.body.getReader();
  const partes: Uint8Array[] = [];
  let total = 0;
  for (;;) {
    const { done, value } = await lector.read();
    if (done) break;
    total += value.length;
    if (total > maximo) { await lector.cancel(); return null; }
    partes.push(value);
  }
  const salida = new Uint8Array(total);
  let pos = 0;
  for (const parte of partes) { salida.set(parte, pos); pos += parte.length; }
  return salida;
}
