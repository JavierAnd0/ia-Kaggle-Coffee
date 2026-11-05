import { useEffect, useReducer, useRef, useState } from 'react';
import type { Calidad } from '../lib/calidad';
import { hayProblema } from '../lib/calidad';
import { MAX_FOTOS } from '../lib/combinar';
import { medirCalidad, reducirImagen } from '../lib/imagen';
import type { Probabilidades } from '../lib/predictor/tipos';
import type { ErrorApi, RespuestaDiagnostico } from '../lib/tipos';
import { Camara, Galeria, Hoja } from './Iconos';
import Resultado from './Resultado';

/**
 * Isla de React: máquina de estados  inicio → previa → analizando → resultado | error.
 * Un solo paso a la vez y botones grandes, para personas poco familiarizadas con la tecnología.
 * `hojas` guarda las probabilidades de cada hoja ya revisada de la misma planta; con 2 o más se
 * pide al servidor (/api/combinar) un resultado combinado.
 */
type Estado =
  | { fase: 'inicio' }
  | { fase: 'previa'; foto: Blob; url: string; calidad: Calidad | null }
  | { fase: 'analizando'; url: string }
  | { fase: 'resultado'; datos: RespuestaDiagnostico; url: string }
  | { fase: 'error'; mensaje: string; url?: string };

type Accion =
  | { tipo: 'foto'; foto: Blob; url: string; calidad: Calidad | null }
  | { tipo: 'analizar'; url: string }
  | { tipo: 'listo'; datos: RespuestaDiagnostico; url: string }
  | { tipo: 'fallo'; mensaje: string; url?: string }
  | { tipo: 'inicio' };

function reducir(_: Estado, a: Accion): Estado {
  switch (a.tipo) {
    case 'foto': return { fase: 'previa', foto: a.foto, url: a.url, calidad: a.calidad };
    case 'analizar': return { fase: 'analizando', url: a.url };
    case 'listo': return { fase: 'resultado', datos: a.datos, url: a.url };
    case 'fallo': return { fase: 'error', mensaje: a.mensaje, url: a.url };
    default: return { fase: 'inicio' };
  }
}

function AvisosCalidad({ calidad }: { calidad: Calidad }) {
  return (
    <div className="aviso" role="note" data-testid="aviso-calidad">
      <strong>La foto podría salir mejor</strong>
      {calidad.borrosa && <p>Parece borrosa. Apoye el celular en algo firme y espere a que enfoque.</p>}
      {calidad.oscura && <p>Está muy oscura. Busque más luz (mejor de día, sin sombra sobre la hoja).</p>}
      {calidad.muyClara && <p>Hay demasiada luz o reflejo. Evite el sol directo sobre la hoja.</p>}
      <p>Puede continuar igual, pero el resultado será menos fiable.</p>
    </div>
  );
}

export default function Diagnostico() {
  const [estado, enviar] = useReducer(reducir, { fase: 'inicio' } as Estado);
  const [hojas, fijarHojas] = useState<Probabilidades[]>([]);
  const [modoPrueba, fijarModoPrueba] = useReducer((_: boolean | null, v: boolean | null) => v, null);
  const camara = useRef<HTMLInputElement>(null);
  const galeria = useRef<HTMLInputElement>(null);
  const titulo = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    fetch('/api/salud').then((r) => r.json()).then((j) => fijarModoPrueba(Boolean(j.modoSimulado))).catch(() => fijarModoPrueba(null));
  }, []);

  // La URL de la foto se reutiliza en varias fases; solo se libera al cambiar de foto o al salir.
  const urlActual = useRef<string | null>(null);
  const liberar = () => { if (urlActual.current) { URL.revokeObjectURL(urlActual.current); urlActual.current = null; } };
  useEffect(() => liberar, []);

  // Al cambiar de paso se lleva el foco al título (lectores de pantalla), pero no al abrir la página.
  const primera = useRef(true);
  useEffect(() => {
    if (primera.current) { primera.current = false; return; }
    titulo.current?.focus({ preventScroll: false });
  }, [estado.fase]);

  /** Otra hoja de la MISMA planta: se conservan los resultados anteriores. */
  const otraHoja = () => { liberar(); enviar({ tipo: 'inicio' }); };
  /** Otra planta: se empieza de cero. */
  const otraPlanta = () => { liberar(); fijarHojas([]); enviar({ tipo: 'inicio' }); };

  async function alElegir(archivo?: File) {
    if (!archivo) return;
    if (!archivo.type.startsWith('image/')) { enviar({ tipo: 'fallo', mensaje: 'Ese archivo no es una foto. Elija una imagen.' }); return; }
    try {
      const reducida = await reducirImagen(archivo);
      const calidad = await medirCalidad(reducida);
      liberar();
      urlActual.current = URL.createObjectURL(reducida);
      enviar({ tipo: 'foto', foto: reducida, url: urlActual.current, calidad });
    } catch {
      enviar({ tipo: 'fallo', mensaje: 'No pudimos abrir esa foto. Intente con otra.' });
    }
  }

  async function combinar(lista: Probabilidades[]): Promise<RespuestaDiagnostico | null> {
    try {
      const r = await fetch('/api/combinar', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ fotos: lista }) });
      return r.ok ? ((await r.json()) as RespuestaDiagnostico) : null;
    } catch {
      return null;
    }
  }

  async function analizar(foto: Blob, url: string) {
    enviar({ tipo: 'analizar', url });
    try {
      const respuesta = await fetch('/api/diagnostico', { method: 'POST', headers: { 'Content-Type': foto.type || 'image/jpeg' }, body: foto });
      const cuerpo = (await respuesta.json()) as RespuestaDiagnostico | ErrorApi;
      if (!respuesta.ok) { enviar({ tipo: 'fallo', mensaje: (cuerpo as ErrorApi).error ?? 'Algo salió mal. Intente de nuevo.', url }); return; }
      const datos = cuerpo as RespuestaDiagnostico;
      const lista = [...hojas, datos.probabilidades];
      fijarHojas(lista);
      // Con 2 o más hojas se muestra el resultado combinado; si falla, se muestra el de esta hoja.
      const combinado = lista.length > 1 ? await combinar(lista) : null;
      enviar({ tipo: 'listo', datos: combinado ?? datos, url });
    } catch {
      enviar({ tipo: 'fallo', mensaje: 'No hay conexión con el servicio. Revise su señal e intente de nuevo.', url });
    }
  }

  const entradas = (
    <>
      <input ref={camara} type="file" accept="image/*" capture="environment" hidden onChange={(e) => { void alElegir(e.target.files?.[0]); e.target.value = ''; }} data-testid="entrada-camara" />
      <input ref={galeria} type="file" accept="image/*" hidden onChange={(e) => { void alElegir(e.target.files?.[0]); e.target.value = ''; }} data-testid="entrada-galeria" />
    </>
  );

  return (
    <section className="isla" aria-label="Revisar una hoja">
      {modoPrueba && (
        <div className="aviso aviso-prueba" role="alert" data-testid="modo-prueba">
          <strong>MODO DE PRUEBA: los resultados son simulados</strong>
          <p>El modelo de inteligencia artificial todavía no está conectado. Lo que verá no analiza su hoja y no sirve para decidir nada.</p>
        </div>
      )}
      {entradas}

      {estado.fase === 'inicio' && (
        <>
          <h2 ref={titulo} tabIndex={-1}>{hojas.length === 0 ? 'Paso 1: tome la foto' : `Hoja ${hojas.length + 1} de la misma planta`}</h2>
          {hojas.length > 0 && <p className="nota">Ya revisó {hojas.length} {hojas.length === 1 ? 'hoja' : 'hojas'}. Los resultados se combinarán.</p>}
          <div className="fila-botones dos">
            <button type="button" className="boton" onClick={() => camara.current?.click()}><Camara />Tomar foto</button>
            <button type="button" className="boton boton-claro" onClick={() => galeria.current?.click()}><Galeria />Elegir de la galería</button>
          </div>
          {hojas.length > 0 && (
            <div className="fila-botones"><button type="button" className="boton boton-claro" onClick={otraPlanta}>Empezar con otra planta</button></div>
          )}
        </>
      )}

      {estado.fase === 'previa' && (
        <>
          <h2 ref={titulo} tabIndex={-1}>Paso 2: ¿se ve bien la hoja?</h2>
          <div className="vista-previa"><img src={estado.url} alt="Foto de la hoja que va a revisar" /></div>
          {estado.calidad && hayProblema(estado.calidad)
            ? <AvisosCalidad calidad={estado.calidad} />
            : <p className="nota" style={{ marginTop: '0.6rem' }}>La hoja debe verse entera, con buena luz y sin borrosidad.</p>}
          <div className="fila-botones dos">
            <button type="button" className="boton" onClick={() => void analizar(estado.foto, estado.url)}>Sí, revisar esta hoja</button>
            <button type="button" className="boton boton-claro" onClick={otraHoja}>No, tomar otra</button>
          </div>
        </>
      )}

      {estado.fase === 'analizando' && (
        <div className="analizando" role="status">
          <Hoja className="hoja-giratoria" />
          <h2 ref={titulo} tabIndex={-1}>Revisando la hoja…</h2>
          <p>Esto puede tardar unos segundos.</p>
        </div>
      )}

      {estado.fase === 'error' && (
        <>
          <h2 ref={titulo} tabIndex={-1}>No se pudo revisar</h2>
          <div className="error" role="alert">{estado.mensaje}</div>
          <div className="fila-botones"><button type="button" className="boton" onClick={otraHoja}>Intentar de nuevo</button></div>
        </>
      )}

      {estado.fase === 'resultado' && (
        <>
          <h2 ref={titulo} tabIndex={-1} style={{ outline: 'none' }}>
            {hojas.length > 1 ? `Paso 3: resultado de ${hojas.length} hojas` : 'Paso 3: resultado'}
          </h2>
          <div className="vista-previa" style={{ marginBottom: '1rem' }}><img src={estado.url} alt="Última foto revisada" /></div>
          <Resultado datos={estado.datos} puedeAgregar={hojas.length < MAX_FOTOS} alAgregar={otraHoja} alReiniciar={otraPlanta} />
        </>
      )}
    </section>
  );
}
