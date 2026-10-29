"""Entrenamiento final y comparación (ablation) de mejoras: pesos de clase, fine-tuning y GAN.

Todas las variantes se comparan sobre VALIDACIÓN; el test se usa una sola vez con la mejor.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight

from .config import SEED
from .generators import make_dataset
from .model import BACKBONES, build_model, unfreeze_top


def class_weights(df: pd.DataFrame, li: dict) -> dict:
    y = df["label"].map(li).values
    w = compute_class_weight("balanced", classes=np.arange(len(li)), y=y)
    return {i: float(v) for i, v in enumerate(w)}


def train_model(hp, train_df, val_df, li, epochs=30, use_class_weight=False,
                finetune_epochs=0, verbose=0, weight_df=None):
    """`weight_df`: datos reales con los que se calculan los pesos de clase (por defecto train_df);
    así las imágenes sintéticas no alteran los pesos y el efecto de la GAN queda aislado."""
    tf.keras.utils.set_random_seed(SEED)
    tf.keras.backend.clear_session()
    pre = BACKBONES[hp.values["backbone"]][1]
    bs = hp.values.get("batch_size", 32)
    tr = make_dataset(train_df, li, bs, training=True, augment=True, preprocess=pre)
    va = make_dataset(val_df, li, bs, preprocess=pre)
    cw = class_weights(train_df if weight_df is None else weight_df, li) if use_class_weight else None
    stop = lambda: tf.keras.callbacks.EarlyStopping("val_accuracy", patience=5, restore_best_weights=True)
    model = build_model(hp, len(li))
    model.fit(tr, validation_data=va, epochs=epochs, class_weight=cw, callbacks=[stop()], verbose=verbose)
    if finetune_epochs:
        stage1_w, stage1_acc = model.get_weights(), model.evaluate(va, verbose=0)[1]
        unfreeze_top(model)
        model.fit(tr, validation_data=va, epochs=finetune_epochs, class_weight=cw,
                  callbacks=[stop()], verbose=verbose)
        if model.evaluate(va, verbose=0)[1] < stage1_acc:   # el fine-tuning no mejoró: conservar etapa 1
            model.set_weights(stage1_w)
    return model


def evaluate(model, hp, df, li) -> dict:
    pre = BACKBONES[hp.values["backbone"]][1]
    ds = make_dataset(df, li, hp.values.get("batch_size", 32), preprocess=pre)
    y_true = df["label"].map(li).values
    y_pred = model.predict(ds, verbose=0).argmax(1)
    f1s = f1_score(y_true, y_pred, average=None, labels=range(len(li)), zero_division=0)
    out = {"accuracy": float((y_true == y_pred).mean()),
           "f1_macro": float(f1s.mean())}
    for name, i in li.items():
        out[f"f1_{name}"] = float(f1s[i])
    out["y_true"], out["y_pred"] = y_true, y_pred
    return out


def run_experiments(hp, train_df, val_df, li, synth_df=None, epochs=30, finetune_epochs=10):
    """Entrena las variantes y devuelve (tabla sobre val, {mejor_variante: modelo}).

    Control: `4b` añade la misma cantidad de imágenes REALES de la clase minoritaria (sobremuestreo
    con repetición) para distinguir el efecto de la GAN del simple sobremuestreo. Solo se conserva
    en memoria el mejor modelo (por F1 macro en validación).
    """
    variants = [("1_base", train_df, False, 0),
                ("2_+class_weight", train_df, True, 0),
                ("3_+fine_tuning", train_df, True, finetune_epochs)]
    if synth_df is not None and len(synth_df):
        minority = synth_df["label"].iloc[0]
        over = train_df[train_df["label"] == minority].sample(len(synth_df), replace=True, random_state=SEED)
        variants += [("4_+GAN", pd.concat([train_df, synth_df], ignore_index=True), True, finetune_epochs),
                     ("4b_+oversampling_real", pd.concat([train_df, over], ignore_index=True), True, finetune_epochs)]
    rows, best_name, best_model, best_f1 = [], None, None, -1.0
    for name, tr_df, cw, ft in variants:
        print(f"== {name}  (train={len(tr_df)})")
        m = train_model(hp, tr_df, val_df, li, epochs, cw, ft, weight_df=train_df)
        ev = evaluate(m, hp, val_df, li)
        rows.append({"variant": name, "n_train": len(tr_df),
                     **{k: v for k, v in ev.items() if k not in ("y_true", "y_pred")}})
        if ev["f1_macro"] > best_f1:
            best_name, best_model, best_f1 = name, m, ev["f1_macro"]
    return pd.DataFrame(rows).set_index("variant"), {best_name: best_model}


def confusion_df(ev: dict, li: dict) -> pd.DataFrame:
    names = sorted(li, key=li.get)
    return pd.DataFrame(confusion_matrix(ev["y_true"], ev["y_pred"], labels=range(len(li))),
                        index=[f"real_{n}" for n in names], columns=[f"pred_{n}" for n in names])
