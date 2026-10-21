"""Topic 3: validación cruzada sobre train+val (test intacto).

Implementados: K-fold, Stratified K-fold, Hold-out y Monte Carlo (shuffle-split).
Leave-p-out, Leave-one-out y Time-series se explican en el notebook (inviables / no aplican).
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import (GroupShuffleSplit, KFold, StratifiedGroupKFold,
                                     StratifiedKFold, train_test_split)

from .config import FAST_MODE, RESULTS_DIR, SEED
from .generators import make_dataset, make_label_index
from .model import BACKBONES, build_model


def _fit_eval(hp, tr_df, te_df, li, epochs):
    pre = BACKBONES[hp.values["backbone"]][1]
    bs = hp.values.get("batch_size", 32)
    tr = make_dataset(tr_df, li, bs, training=True, augment=True, preprocess=pre)
    te = make_dataset(te_df, li, bs, preprocess=pre)
    tf.keras.backend.clear_session()
    m = build_model(hp, len(li))
    m.fit(tr, epochs=epochs, verbose=0)
    y_true = te_df["label"].map(li).values
    y_pred = m.predict(te, verbose=0).argmax(1)
    return (y_true == y_pred).mean(), f1_score(y_true, y_pred, average="macro")


def get_splitters(k=5, n_mc=10, seed=SEED):
    return {
        "kfold": ("KFold", lambda df: KFold(k, shuffle=True, random_state=seed).split(df)),
        "stratified_kfold": ("StratifiedKFold", lambda df: StratifiedKFold(
            k, shuffle=True, random_state=seed).split(df, df["label"])),
        "group_stratified_kfold": ("StratifiedGroupKFold", lambda df: StratifiedGroupKFold(
            k, shuffle=True, random_state=seed).split(df, df["label"], df["group"])),
        "holdout": ("Hold-out", lambda df: [train_test_split(
            np.arange(len(df)), test_size=0.2, stratify=df["label"], random_state=seed)]),
        "monte_carlo": ("Monte Carlo", lambda df: GroupShuffleSplit(
            n_mc, test_size=0.2, random_state=seed).split(df, df["label"], df["group"])),
    }


def run_cv(hp, df: pd.DataFrame, methods=None, epochs: int | None = None) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    li = make_label_index(df["label"])
    epochs = epochs or (2 if FAST_MODE else 15)
    rows = []
    for key, (name, split) in get_splitters().items():
        if methods and key not in methods:
            continue
        scores = [_fit_eval(hp, df.iloc[a], df.iloc[b], li, epochs) for a, b in split(df)]
        acc, f1 = np.array(scores).T
        rows.append({"method": name, "n_splits": len(scores), "acc_mean": acc.mean(),
                     "acc_std": acc.std(), "f1_mean": f1.mean(), "f1_std": f1.std()})
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(RESULTS_DIR / "cv_results.csv", index=False)
    return out
