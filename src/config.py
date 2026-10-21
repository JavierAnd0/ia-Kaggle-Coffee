"""Configuración global del proyecto (rutas, semilla, tamaño de imagen)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data_splits"
RESULTS_DIR = ROOT / "results"
RAW_DIR = ROOT / "data_raw"

KAGGLE_HANDLE = "nirmalsankalana/rocole-a-robusta-coffee-leaf-images-dataset"
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# 70 / 15 / 15
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Reduce épocas/trials para probar rápido (smoke test o CPU)
FAST_MODE = False
