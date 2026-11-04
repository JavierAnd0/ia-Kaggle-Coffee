import type { Fuente, GuiaClase as Guia } from '../lib/guia';

interface Props {
  guia: Guia;
  fuentes: Fuente[];
  /** Muestra "Cómo reconocerla" (en la página de guías) u oculta (ya se ve en la foto). */
  mostrarSenales?: boolean;
  /** Hoja sana: los títulos cambian («Para evitarla» no tiene sentido). */
  sana?: boolean;
}

/** Componente de presentación puro: lo usan la página de guías (sin JS) y el resultado (isla). */
export default function GuiaClase({ guia, fuentes, mostrarSenales = false, sana = false }: Props) {
  return (
    <>
      <section className="ficha" aria-labelledby="g-hacer">
        <h3 id="g-hacer">Qué hacer hoy</h3>
        <ol className="lista-pasos">
          {guia.que_hacer_hoy.map((p) => (<li key={p}>{p}</li>))}
        </ol>
      </section>

      {mostrarSenales && (
        <section className="ficha" aria-labelledby="g-senales">
          <h3 id="g-senales">{sana ? 'Cómo se ve una hoja sana' : 'Cómo reconocerla'}</h3>
          <ul>{guia.como_reconocerla.map((s) => (<li key={s}>{s}</li>))}</ul>
        </section>
      )}

      {guia.productos.tipos.length > 0 && (
        <section className="ficha" aria-labelledby="g-prod">
          <h3 id="g-prod">Productos: solo como orientación</h3>
          {guia.productos.tipos.map((t) => (
            <div className="producto" key={t.tipo}>
              <p><strong>{t.tipo}.</strong> {t.para_que}</p>
              <p className="nota">{t.advertencia}</p>
            </div>
          ))}
          <p className="nota"><strong>Importante:</strong> {guia.productos.nota_registro}</p>
          <p className="nota">No damos dosis ni marcas. Su técnico le dirá qué producto y cuánto usar.</p>
        </section>
      )}

      <section className="ficha" aria-labelledby="g-prev">
        <h3 id="g-prev">{sana ? 'Para mantenerla sana' : 'Para evitarla'}</h3>
        <ul>{guia.prevencion.map((s) => (<li key={s}>{s}</li>))}</ul>
      </section>

      <section className="ficha" aria-labelledby="g-tec">
        <h3 id="g-tec">Llame al técnico si…</h3>
        <ul>{guia.cuando_llamar_al_tecnico.map((s) => (<li key={s}>{s}</li>))}</ul>
      </section>

      <section className="ficha" aria-labelledby="g-col">
        <h3 id="g-col">En Colombia</h3>
        <p>{guia.contexto_colombia}</p>
        {fuentes.length > 0 && (
          <details className="fuentes">
            <summary>De dónde sale esta información</summary>
            <ul>
              {fuentes.map((f) => (
                <li key={f.id}><a href={f.url} target="_blank" rel="noreferrer noopener">{f.titulo}</a> — {f.entidad}</li>
              ))}
            </ul>
          </details>
        )}
      </section>
    </>
  );
}
