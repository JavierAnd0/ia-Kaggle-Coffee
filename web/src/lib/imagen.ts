/** Utilidades de imagen en el navegador (solo se importan desde la isla de React). */
export const MAX_LADO = 1024;
export const CALIDAD = 0.85;

/**
 * Reduce la foto antes de enviarla: una foto de celular pesa 3–8 MB; reducida pesa ~150–300 KB.
 * Importa mucho en el campo, donde la señal es débil y los datos son costosos.
 * `imageOrientation: 'from-image'` evita que la hoja llegue girada.
 */
export async function reducirImagen(archivo: Blob, lado = MAX_LADO, calidad = CALIDAD): Promise<Blob> {
  const mapa = await createImageBitmap(archivo, { imageOrientation: 'from-image' });
  const escala = Math.min(1, lado / Math.max(mapa.width, mapa.height));
  const ancho = Math.round(mapa.width * escala);
  const alto = Math.round(mapa.height * escala);
  const lienzo = document.createElement('canvas');
  lienzo.width = ancho;
  lienzo.height = alto;
  const ctx = lienzo.getContext('2d');
  if (!ctx) throw new Error('El navegador no permite procesar la imagen.');
  ctx.drawImage(mapa, 0, 0, ancho, alto);
  mapa.close();
  return new Promise((resolver, rechazar) =>
    lienzo.toBlob((b) => (b ? resolver(b) : rechazar(new Error('No se pudo preparar la foto.'))), 'image/jpeg', calidad),
  );
}
