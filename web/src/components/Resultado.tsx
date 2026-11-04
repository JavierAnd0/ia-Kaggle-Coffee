import { useEffect, useState } from 'react';
import { fraseConfiabilidad } from '../lib/confiabilidad';
import { GUIA_NO_CONCLUYENTE, NOMBRE_CORTO } from '../lib/guia';
import { detener, leer, vozDisponible } from '../lib/voz';
import type { Clase } from '../lib/predictor/tipos';
import type { RespuestaDiagnostico } from '../lib/tipos';
import GuiaClase from './GuiaClase';
import { Altavoz, Detener, Reiniciar, Whatsapp } from './Iconos';

const ORDEN: Clase[] = ['healthy', 'rust', 'red_spider_mite'];
const NIVEL: Record<string, string> = { alta: 'Seguridad alta', media: 'Seguridad media', baja: 'Seguridad baja' };

function Sello({ porcentaje, nivel }: { porcentaje: number; nivel: string }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  const color = nivel === 'alta' ? 'var(--verde)' : nivel === 'media' ? '#b8860b' : 'var(--cereza)';
  return (
    <svg className="sello" viewBox="0 0 120 120" role="img" aria-label={`${porcentaje} por ciento de seguridad`}>
      <circle cx="60" cy="60" r="58" fill="#fffaf0" stroke="var(--linea)" strokeWidth="2" strokeDasharray="2 5" />
      <circle cx="60" cy="60" r={r} fill="none" stroke="var(--papel-oscuro)" strokeWidth="9" />
      <circle cx="60" cy="60" r={r} fill="none" stroke={color} strokeWidth="9" strokeLinecap="round"
        strokeDasharray={`${(c * porcentaje) / 100} ${c}`} transform="rotate(-90 60 60)" />
      <text x="60" y="64" textAnchor="middle" className="pct">{porcentaje}%</text>
      <text x="60" y="84" textAnchor="middle" className="niv">{nivel === 'alta' ? 'alta' : nivel === 'media' ? 'media' : 'baja'}</text>
    </svg>
  );
}

function textoParaLeer(r: RespuestaDiagnostico): string {
  const v = r.veredicto;
  if (!v.clase || !r.guia) {
    return `No estoy seguro de lo que veo. Lo más parecido es ${NOMBRE_CORTO[v.claseProbable]}, con ${v.porcentaje} por ciento. ${GUIA_NO_CONCLUYENTE.pasos.join(' ')} Esta es una orientación inicial, no un diagnóstico.`;
  }
  const pasos = r.guia.que_hacer_hoy.map((p, i) => `Paso ${i + 1}. ${p}`).join(' ');
  return `Resultado: ${r.guia.nombre_comun}, con ${v.porcentaje} por ciento de seguridad. ${r.guia.resumen} Qué hacer hoy. ${pasos} Esta es una orientación inicial, no un diagnóstico. Confirme con un técnico.`;
}

function textoWhatsapp(r: RespuestaDiagnostico): string {
  const v = r.veredicto;
  const que = v.clase ? `parece ${NOMBRE_CORTO[v.clase].toLowerCase()} (${v.porcentaje}% de seguridad)` : 'no fue concluyente';
  return `Cartilla de la Hoja: en la foto de mi hoja de café ${que}. Es una orientación inicial, no un diagnóstico. Voy a consultarlo con el técnico de extensión.`;
}

export default function Resultado({ datos, alReiniciar }: { datos: RespuestaDiagnostico; alReiniciar: () => void }) {
  const { veredicto: v } = datos;
  const [hablando, setHablando] = useState(false);
  useEffect(() => () => detener(), []);

  const concluyente = v.clase !== null;
  const sano = v.clase === 'healthy';
  const titulo = !concluyente ? 'No estoy seguro' : sano ? 'No se ve roya ni arañita' : `Parece ${NOMBRE_CORTO[v.clase!].toLowerCase()}`;
  const clase = !concluyente ? 'dudoso' : sano ? 'sano' : 'enfermo';

  const alternar = () => {
    if (hablando) { detener(); setHablando(false); return; }
    setHablando(true);
    leer(textoParaLeer(datos), () => setHablando(false));
  };

  return (
    <div className="resultado" aria-live="polite">
      <div className={`veredicto ${clase}`}>
        <div>
          <div className="etiqueta">Resultado de la foto</div>
          <h2>{titulo}</h2>
          <p style={{ margin: 0 }}>
            {concluyente
              ? <>{NIVEL[v.nivel]}: el programa le da {v.porcentaje} de 100 a esta opción.</>
              : <>Lo más parecido es <strong>{NOMBRE_CORTO[v.claseProbable]}</strong> ({v.porcentaje}%), pero no alcanza para afirmarlo.</>}
          </p>
        </div>
        <Sello porcentaje={v.porcentaje} nivel={v.nivel} />
        <p className="nota" style={{ margin: 0, gridColumn: '1 / -1' }}>{fraseConfiabilidad(v.nivel)}</p>
      </div>

      {sano && (
        <div className="aviso" role="note">
          <strong>Ojo: esto no garantiza que la planta esté sana</strong>
          <p>El programa se equivoca a veces. Revise otras hojas, también por el envés (la cara de abajo).</p>
        </div>
      )}

      <section className="ficha" aria-labelledby="r-prob">
        <h3 id="r-prob">Qué tan parecida es la hoja a cada caso</h3>
        <ul className="barras">
          {ORDEN.map((c) => (
            <li key={c}>
              <span>{NOMBRE_CORTO[c]}</span>
              <span className="pista" aria-hidden="true"><span className={`relleno ${c}`} style={{ width: `${datos.porcentajes[c]}%` }} /></span>
              <span className="valor">{datos.porcentajes[c]}%</span>
            </li>
          ))}
        </ul>
        <p className="nota" style={{ marginTop: '0.8rem' }}>Los porcentajes miden el parecido con las fotos con las que se entrenó el programa; no son un diagnóstico.</p>
      </section>

      <div className="fila-botones dos">
        {vozDisponible() && (
          <button type="button" className="boton boton-claro" onClick={alternar}>
            {hablando ? <Detener /> : <Altavoz />}{hablando ? 'Detener lectura' : 'Leer en voz alta'}
          </button>
        )}
        <a className="boton boton-claro" href={`https://wa.me/?text=${encodeURIComponent(textoWhatsapp(datos))}`} target="_blank" rel="noreferrer noopener">
          <Whatsapp />Enviar por WhatsApp
        </a>
      </div>

      {datos.guia ? (
        <GuiaClase guia={datos.guia} fuentes={datos.fuentes} sana={sano} />
      ) : (
        <section className="ficha" aria-labelledby="r-nc">
          <h3 id="r-nc">{GUIA_NO_CONCLUYENTE.titulo}</h3>
          <ol className="lista-pasos">{GUIA_NO_CONCLUYENTE.pasos.map((p) => (<li key={p}>{p}</li>))}</ol>
          <p className="nota">Puede leer las guías de <a href="/guias/rust">roya</a> y <a href="/guias/red_spider_mite">arañita roja</a> para comparar con su hoja.</p>
        </section>
      )}

      <aside className="aviso" role="note"><p>{datos.advertencia}</p></aside>

      <div className="fila-botones">
        <button type="button" className="boton" onClick={() => { detener(); alReiniciar(); }}><Reiniciar />Revisar otra hoja</button>
      </div>
    </div>
  );
}
