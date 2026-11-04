import sharp from 'sharp';
import { describe, expect, it } from 'vitest';
import { decodificarRGB, LADO, normalizarMobileNetV2, redimensionarBilineal } from '../src/lib/predictor/preprocesar';

describe('preprocesar', () => {
  it('imagen de color uniforme se conserva al redimensionar', () => {
    const datos = new Uint8Array(10 * 6 * 3).fill(100);
    const out = redimensionarBilineal({ datos, ancho: 10, alto: 6 });
    expect(out).toHaveLength(LADO * LADO * 3);
    expect(out.every((v) => Math.abs(v - 100) < 1e-3)).toBe(true);
  });

  it('interpola como tf.image.resize bilineal (2x2 → 4x4, centros de píxel)', () => {
    // Valores de referencia de tf.image.resize([[0,100],[100,200]], (4,4), 'bilinear')
    const datos = new Uint8Array([0, 0, 0, 100, 100, 100, 100, 100, 100, 200, 200, 200]);
    const out = redimensionarBilineal({ datos, ancho: 2, alto: 2 }, 4);
    const fila0 = [0, 1, 2, 3].map((x) => out[x * 3]);
    expect(fila0).toEqual([0, 25, 75, 100]);
    expect(out[(3 * 4 + 3) * 3]).toBeCloseTo(200, 4);
  });

  it('normaliza a [-1, 1]', () => {
    const n = normalizarMobileNetV2(new Float32Array([0, 127.5, 255]));
    expect(Array.from(n)).toEqual([-1, 0, 1]);
  });

  it('decodifica un JPEG real y respeta la orientación EXIF', async () => {
    const base = await sharp({ create: { width: 40, height: 20, channels: 3, background: { r: 200, g: 30, b: 30 } } })
      .jpeg()
      .withMetadata({ orientation: 6 }) // rotada 90° al mostrarla
      .toBuffer();
    const img = await decodificarRGB(base);
    expect([img.ancho, img.alto]).toEqual([20, 40]);
    expect(Math.abs(img.datos[0] - 200)).toBeLessThan(6);
  });
});

describe('errores de imagen', () => {
  it('una foto truncada lanza ImagenInvalidaError (error del usuario, no del servicio)', async () => {
    const { ImagenInvalidaError } = await import('../src/lib/predictor/preprocesar');
    const completa = await sharp({ create: { width: 300, height: 200, channels: 3, background: { r: 10, g: 200, b: 10 } } }).jpeg().toBuffer();
    const truncada = completa.subarray(0, Math.floor(completa.length / 3));
    await expect(decodificarRGB(truncada)).rejects.toBeInstanceOf(ImagenInvalidaError);
    await expect(decodificarRGB(new Uint8Array([0xff, 0xd8, 0xff, 0, 1, 2]))).rejects.toBeInstanceOf(ImagenInvalidaError);
  });
});
