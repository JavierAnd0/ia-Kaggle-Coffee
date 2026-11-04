import { useEffect, useReducer, useRef } from 'react';
import { reducirImagen } from '../lib/imagen';
import type { ErrorApi, RespuestaDiagnostico } from '../lib/tipos';
import { Camara, Galeria, Hoja } from './Iconos';
import Resultado from './Resultado';

/**
 * Isla de React: máquina de estados  inicio → previa → analizando → resultado | error.
 * Un solo paso a la vez y botones grandes, para personas poco familiarizadas con la tecnología.
 */
type Estado =
  | { fase: 'inicio' }
  | { fase: 'previa'; foto: Blob; url: string }
  | { fase: 'analizando'; url: string }
  | { fase: 'resultado'; datos: RespuestaDiagnostico; url: string }
  | { fase: 'error'; mensaje: string; url?: string };

type Accion =
  | { tipo: 'foto'; foto: Blob; url: string }
  | { tipo: 'analizar'; url: string }
  | { tipo: 'listo'; datos: RespuestaDiagnostico; url: string }
  | { tipo: 'fallo'; mensaje: string; url?: string }
  | { tipo: 'reiniciar' };

function reducir(_: Estado, a: Accion): Estado {
  switch (a.tipo) {
    case 'foto': return { fase: 'previa', foto: a.foto, url: a.url };
    case 'analizar': return { fase: 'analizando', url: a.url };
    case 'listo': return { fase: 'resultado', datos: a.datos, url: a.url };
    case 'fallo': return { fase: 'error', mensaje: a.mensaje, url: a.url };
    default: return { fase: 'inicio' };
  }
}

export default function Diagnostico() {
  const [estado, enviar] = useReducer(reducir, { fase: 'inicio' } as Estado);
  const [modoPrueba, fijarModoPrueba] = useReducer((_: boolean | null, v: boolean | null) => v, null);
  const camara = useRef<HTMLInputElement>(null);
  const galeria = useRef<HTMLInputElement>(null);
  const titulo = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    fetch('/api/salud').then((r) => r.json()).then((j) => fijarModoPrueba(Boolean(j.modoSimulado))).catch(() => fijarModoPrueba(null));
  }, []);

  // La URL de la foto se reutiliza en varias fases; solo se libera al reiniciar o al salir.
  const urlActual = useRef<string | null>(null);
  const liberar = () => { if (urlActual.current) { URL.revokeObjectURL(urlActual.current); urlActual.current = null; } };
  useEffect(() => liberar, []);

  // Al cambiar de paso se lleva el foco al título (lectores de pantalla), pero no al abrir la página.
  const primera = useRef(true);
  useEffect(() => {
    if (primera.current) { primera.current = false; return; }
    titulo.current?.focus({ preventScroll: false });
  }, [estado.fase]);

  const reiniciar = () => { liberar(); enviar({ tipo: 'reiniciar' }); };

  async function alElegir(archivo?: File) {
    if (!archivo) return;
    if (!archivo.type.startsWith('image/')) { enviar({ tipo: 'fallo', mensaje: 'Ese archivo no es una foto. Elija una imagen.' }); return; }
    try {
      const reducida = await reducirImagen(archivo);
      liberar();
      urlActual.current = URL.createObjectURL(reducida);
      enviar({ tipo: 'foto', foto: reducida, url: urlActual.current });
    } catch {
      enviar({ tipo: 'fallo', mensaje: 'No pudimos abrir esa foto. Intente con otra.' });
    }
  }

  async function analizar(foto: Blob, url: string) {
    enviar({ tipo: 'analizar', url });
    try {
      const respuesta = await fetch('/api/diagnostico', { method: 'POST', headers: { 'Content-Type': foto.type || 'image/jpeg' }, body: foto });
      const cuerpo = (await respuesta.json()) as RespuestaDiagnostico | ErrorApi;
      if (!respuesta.ok) { enviar({ tipo: 'fallo', mensaje: (cuerpo as ErrorApi).error ?? 'Algo salió mal. Intente de nuevo.', url }); return; }
      enviar({ tipo: 'listo', datos: cuerpo as RespuestaDiagnostico, url });
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
          <h2 ref={titulo} tabIndex={-1}>Paso 1: tome la foto</h2>
          <div className="fila-botones dos">
            <button type="button" className="boton" onClick={() => camara.current?.click()}><Camara />Tomar foto</button>
            <button type="button" className="boton boton-claro" onClick={() => galeria.current?.click()}><Galeria />Elegir de la galería</button>
          </div>
        </>
      )}

      {estado.fase === 'previa' && (
        <>
          <h2 ref={titulo} tabIndex={-1}>Paso 2: ¿se ve bien la hoja?</h2>
          <div className="vista-previa"><img src={estado.url} alt="Foto de la hoja que va a revisar" /></div>
          <p className="nota" style={{ marginTop: '0.6rem' }}>La hoja debe verse entera, con buena luz y sin borrosidad.</p>
          <div className="fila-botones dos">
            <button type="button" className="boton" onClick={() => void analizar(estado.foto, estado.url)}>Sí, revisar esta hoja</button>
            <button type="button" className="boton boton-claro" onClick={reiniciar}>No, tomar otra</button>
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
          <div className="fila-botones"><button type="button" className="boton" onClick={reiniciar}>Intentar de nuevo</button></div>
        </>
      )}

      {estado.fase === 'resultado' && (
        <>
          <h2 ref={titulo} tabIndex={-1} style={{ outline: 'none' }}>Paso 3: resultado</h2>
          <div className="vista-previa" style={{ marginBottom: '1rem' }}><img src={estado.url} alt="Foto revisada" /></div>
          <Resultado datos={estado.datos} alReiniciar={reiniciar} />
        </>
      )}
    </section>
  );
}
