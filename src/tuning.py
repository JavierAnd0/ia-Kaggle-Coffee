"""Topic 2: búsqueda de hiperparámetros con Hyperband (keras_tuner). Se evalúa en val, nunca en test."""
import json

import keras_tuner as kt
import tensorflow as tf

from .config import FAST_MODE, RESULTS_DIR, ROOT, SEED
from .generators import make_dataset, make_label_index
from .model import BACKBONES, build_model


def run_hyperband(train_df, val_df, max_epochs: int | None = None):
    li = make_label_index(train_df["label"])
    max_epochs = max_epochs or (3 if FAST_MODE else 30)

    class Hyper(kt.Hyperband):
        # batch_size y backbone (preprocess) son hiperparámetros que afectan al pipeline de datos
        def run_trial(self, trial, *a, **k):
            hp = trial.hyperparameters
            bs = hp.Choice("batch_size", [16, 32, 64])
            pre = BACKBONES[hp.Choice("backbone", list(BACKBONES))][1]
            tr = make_dataset(train_df, li, bs, training=True, augment=True, preprocess=pre)
            va = make_dataset(val_df, li, bs, preprocess=pre)
            return super().run_trial(trial, tr, validation_data=va, **k)

    tuner = Hyper(lambda hp: build_model(hp, len(li)), objective="val_accuracy",
                  max_epochs=max_epochs, factor=3, hyperband_iterations=1, seed=SEED,
                  directory=str(ROOT / "tuner_logs"), project_name="rocole", overwrite=True)
    tuner.search(callbacks=[tf.keras.callbacks.EarlyStopping("val_accuracy", patience=3,
                                                              restore_best_weights=True)])
    best = tuner.get_best_hyperparameters(1)[0]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "best_hp.json").write_text(json.dumps(best.values, indent=2))
    return best
