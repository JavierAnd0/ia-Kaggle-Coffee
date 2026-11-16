"""Topic 1: carga, EDA, partición 70/15/15 sin data leakage y guardado en CSV.

El layout exacto de RoCoLe en Kaggle no se asume: `build_dataframe` descubre las
imágenes y toma la etiqueta de (a) una columna del CSV de anotaciones si se indica,
o (b) el nombre de la carpeta padre. Siempre revisar `df.head()` antes de seguir.
"""
import hashlib
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .config import SEED, SPLITS_DIR, TRAIN_FRAC, VAL_FRAC, TEST_FRAC, KAGGLE_HANDLE

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def download_dataset() -> Path:
    import kagglehub
    return Path(kagglehub.dataset_download(KAGGLE_HANDLE))


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG_EXT)


def build_dataframe(root: Path, csv_path: Path | None = None, file_col: str = "File",
                    label_col: str | None = None, label_map: dict | None = None,
                    group_regex: str | None = None) -> pd.DataFrame:
    """Devuelve DataFrame[filepath, label, group].

    - label: columna `label_col` del CSV (unida por nombre de archivo) o carpeta padre.
    - label_map: p. ej. {"rust_level_1": "rust", ...} para agrupar clases.
    - group: id de planta (4 fotos por planta). Se extrae con `group_regex` (primer
      grupo de captura sobre el nombre de archivo); si no hay regex, cada imagen es su
      propio grupo (y se debe documentar el riesgo de leakage).
    """
    paths = find_images(root)
    df = pd.DataFrame({"filepath": [str(p) for p in paths]})
    df["name"] = [p.name for p in paths]
    if csv_path is not None and label_col is not None:
        ann = pd.read_csv(csv_path)
        ann["name"] = ann[file_col].astype(str).map(lambda s: Path(s).name)
        df = df.merge(ann[["name", label_col]], on="name", how="inner")
        df = df.rename(columns={label_col: "label"})
    else:
        df["label"] = [Path(p).parent.name for p in df["filepath"]]
    if label_map:
        df["label"] = df["label"].replace(label_map)
    if group_regex:
        df["group"] = df["name"].map(
            lambda n: (re.search(group_regex, n).group(1) if re.search(group_regex, n) else n))
    else:
        df["group"] = df["name"]
    return df.drop(columns="name").reset_index(drop=True)


def add_file_hash(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in df["filepath"]]
    return df


def eda(df: pd.DataFrame) -> None:
    print(df["label"].value_counts().to_string())
    print("grupos únicos:", df["group"].nunique(), "| imágenes:", len(df))
    if "md5" in df:
        print("duplicados exactos:", df["md5"].duplicated().sum())


def _merged_groups(df: pd.DataFrame) -> pd.Series:
    """Une planta (`group`) y duplicados exactos (`md5`) en componentes conexas, de modo que
    una planta y todas sus copias caigan siempre en el mismo split."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for g, h in zip(df["group"], df["md5"] if "md5" in df else df["group"]):
        parent[find(("g", g))] = find(("h", h))
    return pd.Series([str(find(("g", g))) for g in df["group"]], index=df.index)


def split_dataset(df: pd.DataFrame, seed: int = SEED, save: bool = True):
    """70/15/15 por grupos (sin leakage entre fotos de la misma planta/duplicados).

    Se agrupa por `md5` cuando existe para que duplicados exactos no crucen splits.
    El split se hace ANTES de cualquier augmentation, GAN o normalización ajustada.
    """
    df = df.reset_index(drop=True).copy()
    groups = _merged_groups(df)
    gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=seed)
    tr_idx, rest_idx = next(gss.split(df, df["label"], groups))
    rest = df.iloc[rest_idx]
    gss2 = GroupShuffleSplit(n_splits=1, train_size=VAL_FRAC / (VAL_FRAC + TEST_FRAC),
                             random_state=seed)
    v_idx, t_idx = next(gss2.split(rest, rest["label"], groups.iloc[rest_idx]))
    train, val, test = df.iloc[tr_idx], rest.iloc[v_idx], rest.iloc[t_idx]
    if save:
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        for name, part in (("train", train), ("val", val), ("test", test)):
            part.to_csv(SPLITS_DIR / f"{name}.csv", index=False)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def assert_no_leakage(train, val, test) -> None:
    sets = {k: set(v["group"]) for k, v in (("train", train), ("val", val), ("test", test))}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"]
                or sets["val"] & sets["test"]), "grupos compartidos entre splits"
    if "md5" in train:
        h = {k: set(v["md5"]) for k, v in (("train", train), ("val", val), ("test", test))}
        assert not (h["train"] & h["val"] or h["train"] & h["test"] or h["val"] & h["test"]), \
            "imágenes duplicadas entre splits"


# --------------------------------------------------------------------------------------------------
# Versión 5: limpieza de etiquetas contradictorias + partición estratificada y agrupada.
# Los CSV antiguos (data_splits/*.csv) se conservan intactos para reproducir el informe.
# --------------------------------------------------------------------------------------------------
LABEL_MAP = {"coffee___healthy": "healthy", "coffee___rust": "rust", "coffee___red_spider_mite": "red_spider_mite"}
GROUP_REGEX = r"^(C\d+P\d+)"
SPLITS_V5_DIR = SPLITS_DIR / "v5"


def limpiar_duplicados(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Quita el ruido de etiquetas de los duplicados exactos (mismo MD5).

    - Grupo MD5 con etiquetas CONTRADICTORIAS (p. ej. la misma foto en rust y en red_spider_mite):
      se eliminan TODAS sus copias, porque no se sabe cuál etiqueta es la correcta.
    - Grupo MD5 con la misma etiqueta: se conserva una sola copia (la primera por ruta).
    """
    assert "md5" in df, "falta la columna md5 (use add_file_hash)"
    df = df.sort_values("filepath").reset_index(drop=True)
    n_etiquetas = df.groupby("md5")["label"].nunique()
    tamanos = df.groupby("md5").size()
    conflictivos = set(n_etiquetas[n_etiquetas > 1].index)
    duplicados = tamanos[tamanos > 1]
    sin_conflicto = df[~df["md5"].isin(conflictivos)]
    limpio = sin_conflicto.drop_duplicates("md5", keep="first").reset_index(drop=True)
    if verbose:
        tipos = (df[df["md5"].isin(conflictivos)].groupby("md5")["label"]
                 .agg(lambda s: "|".join(sorted(set(s)))).value_counts())
        print(f"Grupos MD5 duplicados: {len(duplicados)} | con etiquetas contradictorias: {len(conflictivos)} "
              f"{tipos.to_dict()}")
        print(f"Imágenes eliminadas por etiqueta contradictoria: {int(df['md5'].isin(conflictivos).sum())} | "
              f"copias exactas quitadas: {len(sin_conflicto) - len(limpio)} | quedan {len(limpio)} de {len(df)}")
    return limpio


def _particion_por_folds(etiquetas, grupos, n_folds: int, k: int, seed: int):
    """Separa `k` de `n_folds` folds de StratifiedGroupKFold (estratificado por clase, agrupado)."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=range(len(etiquetas)))
    for i, (_, te) in enumerate(sgkf.split(etiquetas, etiquetas, grupos)):
        fold.iloc[te] = i
    elegidos = fold < k
    return (~elegidos).to_numpy(), elegidos.to_numpy()


def split_dataset_v5(df: pd.DataFrame, seed: int = SEED, save_dir: Path | None = SPLITS_V5_DIR,
                     verbose: bool = True):
    """70/15/15 estratificado por clase y agrupado (planta ∪ duplicados MD5), con StratifiedGroupKFold.

    Se hacen 20 folds de ~5 %: 3 folds van a prueba (15 %), del resto 3 de 17 a validación (15 % del total)
    y lo demás a entrenamiento (70 %). Un mismo grupo nunca cruza splits.
    """
    df = df.reset_index(drop=True).copy()
    grupos = _merged_groups(df)
    n = 20
    k_test = round(TEST_FRAC * n)
    k_val = round(VAL_FRAC * n)
    resto_m, test_m = _particion_por_folds(df["label"].to_numpy(), grupos.to_numpy(), n, k_test, seed)
    resto = df[resto_m].reset_index(drop=True)
    g_resto = grupos[resto_m].reset_index(drop=True)
    tr_m, val_m = _particion_por_folds(resto["label"].to_numpy(), g_resto.to_numpy(), n - k_test, k_val, seed)
    train, val, test = resto[tr_m], resto[val_m], df[test_m]
    train, val, test = (p.reset_index(drop=True) for p in (train, val, test))
    assert_no_leakage(train, val, test)
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, part in (("train", train), ("val", val), ("test", test)):
            part.to_csv(save_dir / f"{name}.csv", index=False)
    if verbose:
        imprimir_conteos(train, val, test)
    return train, val, test


def imprimir_conteos(train, val, test) -> pd.DataFrame:
    tabla = pd.concat({k: v["label"].value_counts() for k, v in (("train", train), ("val", val), ("test", test))},
                      axis=1).fillna(0).astype(int)
    tabla.loc["TOTAL"] = tabla.sum()
    tabla["% train"] = (100 * tabla["train"] / tabla[["train", "val", "test"]].sum(1)).round(1)
    print("Imágenes por clase y split:\n" + tabla.to_string())
    return tabla


def _ruta_relativa(p: str) -> str:
    from .config import ROOT
    ruta = Path(p)
    try:
        return ruta.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return ruta.as_posix()


def preparar_splits_v5(data_root: Path | None = None, seed: int = SEED, regenerar: bool = False,
                       save_dir: Path = SPLITS_V5_DIR, verbose: bool = True):
    """Carga data_splits/v5/*.csv si existen (reproducible entre versiones de scikit-learn); si no, los genera
    desde data/: hash MD5 -> limpieza de duplicados -> partición estratificada y agrupada."""
    from .config import ROOT
    save_dir = Path(save_dir)
    archivos = [save_dir / f"{s}.csv" for s in ("train", "val", "test")]
    if not regenerar and all(a.exists() for a in archivos):
        train, val, test = (pd.read_csv(a) for a in archivos)
        if verbose:
            print(f"Splits v5 leídos de {save_dir}")
            imprimir_conteos(train, val, test)
        return train, val, test
    df = build_dataframe(Path(data_root or ROOT / "data"), label_map=LABEL_MAP, group_regex=GROUP_REGEX)
    df = add_file_hash(df)
    df["filepath"] = [_ruta_relativa(p) for p in df["filepath"]]
    df = limpiar_duplicados(df, verbose=verbose)
    return split_dataset_v5(df, seed=seed, save_dir=save_dir, verbose=verbose)


def make_folds(df: pd.DataFrame, k: int = 5, seed: int = SEED) -> pd.DataFrame:
    """Asigna fold estratificado por clase y agrupado por planta; guarda folds.csv."""
    df = df.reset_index(drop=True).copy()
    df["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for i, (_, te) in enumerate(sgkf.split(df, df["label"], df["group"])):
        df.loc[te, "fold"] = i
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLITS_DIR / "folds.csv", index=False)
    return df
