import type { SVGProps } from 'react';

type P = SVGProps<SVGSVGElement>;
const base: P = { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 2, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true };

export const Camara = (p: P) => (<svg {...base} {...p}><path d="M4 8h3l1.5-2h7L17 8h3v11H4z" /><circle cx="12" cy="13.5" r="3.5" /></svg>);
export const Galeria = (p: P) => (<svg {...base} {...p}><rect x="3" y="4" width="18" height="16" rx="2" /><circle cx="9" cy="10" r="1.6" /><path d="m4 18 5-5 4 4 3-3 4 4" /></svg>);
export const Altavoz = (p: P) => (<svg {...base} {...p}><path d="M4 9v6h4l5 4V5L8 9z" /><path d="M16.5 8.5a5 5 0 0 1 0 7M19 6a8.5 8.5 0 0 1 0 12" /></svg>);
export const Detener = (p: P) => (<svg {...base} {...p}><rect x="6" y="6" width="12" height="12" rx="2" /></svg>);
export const Whatsapp = (p: P) => (<svg {...base} {...p}><path d="M4 20l1.4-4.2A8 8 0 1 1 8.3 18.7z" /><path d="M9 9.5c0 3 2.5 5.5 5.5 5.5l1-1.3-2-1-.8.7a3.6 3.6 0 0 1-1.8-1.8l.7-.8-1-2z" /></svg>);
export const Reiniciar = (p: P) => (<svg {...base} {...p}><path d="M4 12a8 8 0 1 0 3-6.2M4 4v4h4" /></svg>);

/** Hoja de café: sirve de logo y de animación de "analizando". */
export const Hoja = (p: P) => (
  <svg viewBox="0 0 48 48" aria-hidden="true" {...p}>
    <path d="M24 44C10 38 6 22 10 8c14 0 28 8 28 24 0 6-4 10-14 12z" fill="#2f7a55" stroke="#153629" strokeWidth="2.5" strokeLinejoin="round" />
    <path d="M24 44C22 32 20 22 12 12M24 36c4-3 8-6 12-12M23 28c3-1 6-4 8-8" stroke="#dfe9d3" strokeWidth="2" strokeLinecap="round" fill="none" />
  </svg>
);
