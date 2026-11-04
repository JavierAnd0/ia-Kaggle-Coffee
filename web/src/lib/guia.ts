/**
 * Acceso a la base de conocimiento (src/data/conocimiento.json).
 * Patrón "contenido como datos": el texto agronómico vive en un JSON versionado y revisable por
 * un técnico, separado del código. `validarConocimiento` se ejecuta en las pruebas para que un
 * cambio de contenido no rompa la interfaz.
 */
import conocimiento from '../data/conocimiento.json';
import type { Clase } from './predictor/tipos';

export interface TipoProducto { tipo: string; para_que: string; advertencia: string }
export interface GuiaClase {
  nombre_comun: string;
  nombre_cientifico: string | null;
  resumen: string;
  como_reconocerla: string[];
  que_hacer_hoy: string[];
  prevencion: string[];
  cuando_llamar_al_tecnico: string[];
  productos: { tipos: TipoProducto[]; nota_registro: string };
  contexto_colombia: string;
  fuentes: string[];
}
export interface Fuente { id: string; titulo: string; entidad: string; url: string; consultada: string }
export interface Contacto { pais: string; entidad: string; como_contactar: string }
export interface Conocimiento {
  version: string;
  generado: string;
  advertencia_general: string;
  clases: Record<Clase, GuiaClase>;
  contactos: Contacto[];
  fuentes: Fuente[];
  por_verificar?: string[];
}

export const base = conocimiento as unknown as Conocimiento;

export const NOMBRE_CORTO: Record<Clase, string> = {
  healthy: 'Hoja sana',
  rust: 'Roya',
  red_spider_mite: 'Arañita roja',
};

/** Consejos cuando el resultado no es concluyente (el modelo no está seguro). */
export const GUIA_NO_CONCLUYENTE = {
  titulo: 'No estoy seguro de lo que veo',
  pasos: [
    'Tome otra foto de día, con buena luz y sin sombra sobre la hoja.',
    'Ponga una sola hoja, entera, sobre un fondo claro, y acérquese un poco.',
    'Revise varias hojas y varias plantas, mirando también el envés (la cara de abajo).',
    'Si ve manchas, polvillo anaranjado o color bronceado, consulte a un técnico de extensión.',
  ],
};

export const AVISO_ROBUSTA =
  'Esta herramienta fue entrenada con fotos de hojas de café robusta (Coffea canephora). ' +
  'No se ha probado con café arábica ni con cultivos colombianos, así que puede equivocarse.';

export function obtenerGuia(clase: Clase): GuiaClase {
  return base.clases[clase];
}

export function fuentesDe(ids: string[]): Fuente[] {
  return ids.map((id) => base.fuentes.find((f) => f.id === id)).filter((f): f is Fuente => Boolean(f));
}

const LISTAS: (keyof GuiaClase)[] = ['como_reconocerla', 'que_hacer_hoy', 'prevencion', 'cuando_llamar_al_tecnico', 'fuentes'];

/** Devuelve una lista de problemas encontrados (vacía si el contenido es válido). */
export function validarConocimiento(c: Conocimiento = base): string[] {
  const errores: string[] = [];
  for (const clase of ['healthy', 'rust', 'red_spider_mite'] as Clase[]) {
    const g = c.clases[clase];
    if (!g) { errores.push(`falta la clase ${clase}`); continue; }
    if (!g.nombre_comun || !g.resumen || !g.contexto_colombia) errores.push(`${clase}: faltan textos`);
    for (const k of LISTAS) {
      const v = g[k];
      if (!Array.isArray(v) || v.length === 0) errores.push(`${clase}.${String(k)}: lista vacía`);
    }
    if (!g.productos?.nota_registro) errores.push(`${clase}: falta nota_registro`);
    for (const id of g.fuentes ?? []) if (!c.fuentes.some((f) => f.id === id)) errores.push(`${clase}: fuente ${id} no existe`);
    for (const paso of g.que_hacer_hoy ?? []) if (paso.split(/\s+/).length > 24) errores.push(`${clase}: paso demasiado largo`);
  }
  if (!c.advertencia_general) errores.push('falta advertencia_general');
  for (const f of c.fuentes) if (!/^https?:\/\//.test(f.url)) errores.push(`fuente ${f.id}: URL inválida`);
  return errores;
}
