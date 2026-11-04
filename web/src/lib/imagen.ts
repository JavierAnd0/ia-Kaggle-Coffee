/** Utilidades de imagen en el navegador (solo se importan desde la isla de React). */
/**
 * Las fotos de entrenamiento medían 1920×1080. Al enviar a ese tamaño (en vez de 1024) el servidor ve
 * una imagen más parecida a la del entrenamiento: con el modelo real la diferencia de probabilidades
 * frente a Keras bajó de ~0,10 a ~0,06 (ver tests/onnx.test.ts). Sigue ahorrando datos (~200–400 KB).
 */
export const MAX_LADO = 1920;
export const CALIDAD = 0.85;

type Dibujable = ImageBitmap | HTMLImageElement;

/** createImageBitmap con orientación EXIF; si el navegador es antiguo y no lo soporta, usa un <img>. */
async function abrirImagen(archivo: Blob): Promise<Dibujable> {
  try {
    return await createImageBitmap(archivo, { imageOrientation: 'from-image' });
  } catch {
    const url = URL.createObjectURL(archivo);
    try {
      const imagen = new Image();
      imagen.src = url;
      await imagen.decode();
      return imagen;
    } finally {
      URL.revokeObjectURL(url);
    }
  }
}

/**
 * Reduce la foto antes de enviarla: una foto de celular pesa 3–8 MB; reducida pesa ~200–400 KB.
 * Importa mucho en el campo, donde la señal es débil y los datos son costosos.
 * `imageOrientation: 'from-image'` evita que la hoja llegue girada.
 */
export async function reducirImagen(archivo: Blob, lado = MAX_LADO, calidad = CALIDAD): Promise<Blob> {
  const mapa = await abrirImagen(archivo);
  const escala = Math.min(1, lado / Math.max(mapa.width, mapa.height));
  const ancho = Math.round(mapa.width * escala);
  const alto = Math.round(mapa.height * escala);
  const lienzo = document.createElement('canvas');
  lienzo.width = ancho;
  lienzo.height = alto;
  const ctx = lienzo.getContext('2d');
  if (!ctx) throw new Error('El navegador no permite procesar la imagen.');
  ctx.drawImage(mapa, 0, 0, ancho, alto);
  if ('close' in mapa) mapa.close();
  return new Promise((resolver, rechazar) =>
    lienzo.toBlob((b) => (b ? resolver(b) : rechazar(new Error('No se pudo preparar la foto.'))), 'image/jpeg', calidad),
  );
}
