/**
 * Preprocesamiento de imagen en el servidor. Debe reproducir el del entrenamiento
 * (src/generators.py): decodificar → tf.image.resize bilineal (sin antialias, centros de píxel)
 * a 224×224 → preprocess_input de MobileNetV2 (x / 127,5 − 1).
 * Se implementa el redimensionado a mano porque `sharp` usa otros filtros (con antialias) y eso
 * cambiaría ligeramente lo que ve el modelo. `tests/paridad.test.ts` mide la diferencia.
 */
import sharp from 'sharp';

export const LADO = 224;
const MAX_PIXELES = 40_000_000; // protege la memoria ante imágenes gigantes

export interface ImagenRGB {
  datos: Uint8Array; // RGB intercalado, 3 bytes por píxel
  ancho: number;
  alto: number;
}

export async function decodificarRGB(bytes: Uint8Array): Promise<ImagenRGB> {
  const { data, info } = await sharp(bytes, { limitInputPixels: MAX_PIXELES })
    .rotate() // respeta la orientación EXIF (las fotos de celular suelen venir "de lado")
    .removeAlpha()
    .toColourspace('srgb')
    .raw()
    .toBuffer({ resolveWithObject: true });
  return { datos: new Uint8Array(data.buffer, data.byteOffset, data.byteLength), ancho: info.width, alto: info.height };
}

/** Réplica de tf.image.resize(method='bilinear', antialias=False). Devuelve valores 0-255. */
export function redimensionarBilineal(img: ImagenRGB, lado = LADO): Float32Array {
  const { datos, ancho, alto } = img;
  const salida = new Float32Array(lado * lado * 3);
  const escalaX = ancho / lado;
  const escalaY = alto / lado;

  const indicesX = new Int32Array(lado * 2);
  const pesosX = new Float32Array(lado);
  for (let x = 0; x < lado; x += 1) {
    const entrada = (x + 0.5) * escalaX - 0.5;
    const piso = Math.floor(entrada);
    indicesX[2 * x] = Math.max(piso, 0);
    indicesX[2 * x + 1] = Math.min(Math.ceil(entrada), ancho - 1);
    pesosX[x] = entrada - piso;
  }

  for (let y = 0; y < lado; y += 1) {
    const entradaY = (y + 0.5) * escalaY - 0.5;
    const pisoY = Math.floor(entradaY);
    const y0 = Math.max(pisoY, 0);
    const y1 = Math.min(Math.ceil(entradaY), alto - 1);
    const wy = entradaY - pisoY;
    for (let x = 0; x < lado; x += 1) {
      const x0 = indicesX[2 * x];
      const x1 = indicesX[2 * x + 1];
      const wx = pesosX[x];
      for (let c = 0; c < 3; c += 1) {
        const a = datos[(y0 * ancho + x0) * 3 + c];
        const b = datos[(y0 * ancho + x1) * 3 + c];
        const d = datos[(y1 * ancho + x0) * 3 + c];
        const e = datos[(y1 * ancho + x1) * 3 + c];
        const arriba = a + (b - a) * wx;
        const abajo = d + (e - d) * wx;
        salida[(y * lado + x) * 3 + c] = arriba + (abajo - arriba) * wy;
      }
    }
  }
  return salida;
}

/** preprocess_input de MobileNetV2: lleva 0-255 a [-1, 1]. */
export function normalizarMobileNetV2(valores: Float32Array): Float32Array {
  const salida = new Float32Array(valores.length);
  for (let i = 0; i < valores.length; i += 1) salida[i] = valores[i] / 127.5 - 1;
  return salida;
}

export async function prepararEntrada(bytes: Uint8Array): Promise<Float32Array> {
  return normalizarMobileNetV2(redimensionarBilineal(await decodificarRGB(bytes)));
}
