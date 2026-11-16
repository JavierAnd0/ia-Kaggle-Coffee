import { readdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

// Regresión: un notebook con JSON inválido (p. ej. comillas sin escapar) no abre en Colab.
const dir = path.resolve(process.cwd(), '..', 'notebooks');
const archivos = readdirSync(dir).filter((f) => f.endsWith('.ipynb'));

describe('notebooks', () => {
  it('hay notebooks que revisar', () => expect(archivos.length).toBeGreaterThan(0));
  for (const f of archivos) {
    it(`${f} es JSON válido con celdas`, () => {
      const nb = JSON.parse(readFileSync(path.join(dir, f), 'utf8')) as { cells: { source: string[] }[]; nbformat: number };
      expect(nb.nbformat).toBe(4);
      expect(nb.cells.length).toBeGreaterThan(0);
    });
    it(`${f} no usa comandos mágicos de Colab (!, %) y funciona también en PyCharm/Jupyter`, () => {
      const nb = JSON.parse(readFileSync(path.join(dir, f), 'utf8')) as { cells: { cell_type: string; source: string[] }[] };
      const lineas = nb.cells.filter((c) => c.cell_type === 'code').flatMap((c) => c.source);
      expect(lineas.filter((l) => /^\s*[!%]/.test(l))).toEqual([]);
    });
  }
});
