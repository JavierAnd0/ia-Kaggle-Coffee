/** Utilidades de imagen en el navegador (solo se importan desde la isla de React). */
/**
 * Tamaño de envío. Las fotos de entrenamiento medían 1920×1080, pero el servidor las reduce a 224 px
 * de todas formas. Medido con el modelo real frente a Keras (tests/onnx.test.ts): 1024 px desviaba
 * hasta ~0,10; 1920 px a 0,85 hasta 0,060; 1280 px a 0,8 hasta 0,075, siempre con la misma clase.
 * Se eligió 1280/0,8 porque pesa menos de la mitad (~100–200 KB) y en el campo los datos son caros.
 */
export const MAX_LADO = 1280;
export const CALIDAD = 0.8;

import { evaluarCalidad, LADO_CALIDAD, type Calidad } from './calidad';

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

/** Mide nitidez y luz sobre una versión pequeña de la foto (rápido, sin enviar nada). */
export async function medirCalidad(foto: Blob): Promise<Calidad | null> {
  try {
    const mapa = await abrirImagen(foto);
    const escala = Math.min(1, LADO_CALIDAD / Math.max(mapa.width, mapa.height));
    const ancho = Math.max(8, Math.round(mapa.width * escala));
    const alto = Math.max(8, Math.round(mapa.height * escala));
    const lienzo = document.createElement('canvas');
    lienzo.width = ancho;
    lienzo.height = alto;
    const ctx = lienzo.getContext('2d', { willReadFrequently: true });
    if (!ctx) return null;
    ctx.drawImage(mapa, 0, 0, ancho, alto);
    if ('close' in mapa) mapa.close();
    return evaluarCalidad(ctx.getImageData(0, 0, ancho, alto).data, ancho, alto);
  } catch {
    return null; // si no se puede medir, simplemente no se avisa
  }
}
