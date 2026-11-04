/** Lectura en voz alta con la voz del dispositivo (no requiere internet ni servicios externos). */
export function vozDisponible(): boolean {
  return typeof window !== 'undefined' && 'speechSynthesis' in window;
}

export function leer(texto: string, alTerminar: () => void): void {
  if (!vozDisponible()) return;
  const sintetizador = window.speechSynthesis;
  sintetizador.cancel();
  const frase = new SpeechSynthesisUtterance(texto);
  const voces = sintetizador.getVoices();
  const voz = voces.find((v) => v.lang === 'es-CO') ?? voces.find((v) => v.lang.startsWith('es-4') || v.lang === 'es-419') ?? voces.find((v) => v.lang.startsWith('es'));
  frase.lang = voz?.lang ?? 'es-CO';
  if (voz) frase.voice = voz;
  frase.rate = 0.9;
  frase.onend = alTerminar;
  frase.onerror = alTerminar;
  sintetizador.speak(frase);
}

export function detener(): void {
  if (vozDisponible()) window.speechSynthesis.cancel();
}
