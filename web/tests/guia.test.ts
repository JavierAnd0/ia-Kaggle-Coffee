import { describe, expect, it } from 'vitest';
import { base, validarConocimiento } from '../src/lib/guia';

describe('conocimiento.json', () => {
  it('tiene la estructura completa y fuentes consistentes', () => {
    expect(validarConocimiento()).toEqual([]);
  });
  it('detecta contenido roto', () => {
    const roto = structuredClone(base);
    roto.clases.rust.que_hacer_hoy = [];
    roto.clases.rust.fuentes = ['f999'];
    const errores = validarConocimiento(roto);
    expect(errores.join(' ')).toMatch(/lista vacía/);
    expect(errores.join(' ')).toMatch(/f999/);
  });
  it('no incluye dosis ni precios', () => {
    const texto = JSON.stringify(base.clases).toLowerCase();
    expect(texto).not.toMatch(/\b\d+\s?(ml|cc|g|gr|kg|l)\b\s?(por|\/)/);
    expect(texto).not.toMatch(/\$\s?\d|cop\b|pesos/);
  });
});
