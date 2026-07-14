"""
Entrenamiento del clasificador de morfología de granos de cacao.

Uso:
    python -m src.train                              # MobileNetV2 (por defecto)
    python -m src.train --backbone efficientnetb0    # el otro modelo
    python -m src.train --no-fine-tune               # solo la cabeza (más rápido)
    python -m src.train --epochs 10                  # menos épocas de la fase 1

Al terminar guarda:
    models/cacao_<backbone>.keras   -> modelo entrenado
    models/class_names.json         -> orden de las clases
    outputs/history_<backbone>.png  -> curvas de accuracy/loss
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import data as data_mod  # noqa: E402
from src import model as model_mod  # noqa: E402


def _callbacks(model_path, best_threshold: float | None = None, use_lr_reduce=True):
    """Callbacks de entrenamiento.

    `best_threshold` fija el umbral inicial del ModelCheckpoint. En la fase 2
    (fine-tuning) se pasa el mejor val_accuracy de la fase 1, de modo que el
    checkpoint SOLO sobrescriba el modelo guardado si el fine-tuning realmente
    lo mejora. Así nunca se pierde el mejor modelo de la fase anterior.

    `use_lr_reduce`: ReduceLROnPlateau es incompatible con un LR programado
    (cosine decay), así que en la fase 2 se desactiva.
    """
    cbs = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_accuracy",
            save_best_only=True, verbose=0,
            initial_value_threshold=best_threshold,
        ),
    ]
    if use_lr_reduce:
        cbs.insert(1, tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ))
    return cbs


def _plot_history(histories, backbone):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no disponible; se omite la gráfica.")
        return

    acc, val_acc, loss, val_loss = [], [], [], []
    for h in histories:
        acc += h.history.get("accuracy", [])
        val_acc += h.history.get("val_accuracy", [])
        loss += h.history.get("loss", [])
        val_loss += h.history.get("val_loss", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(acc, label="train"); ax1.plot(val_acc, label="val")
    ax1.set_title(f"Accuracy ({backbone})"); ax1.set_xlabel("época"); ax1.legend()
    ax2.plot(loss, label="train"); ax2.plot(val_loss, label="val")
    ax2.set_title(f"Loss ({backbone})"); ax2.set_xlabel("época"); ax2.legend()
    fig.tight_layout()
    out = config.OUTPUTS_DIR / f"history_{backbone}.png"
    fig.savefig(out, dpi=120)
    print(f"Curvas guardadas en {out}")


def train_backbone(backbone, train_ds, val_ds, class_names,
                   epochs=config.EPOCHS_HEAD, ft_epochs=config.EPOCHS_FINE_TUNE,
                   do_fine_tune=True, out_path=None, plot=True):
    """Entrena un backbone concreto y devuelve la ruta del mejor modelo guardado.

    Reutilizable por src.compare para entrenar los dos modelos con el mismo
    reparto de datos (comparación justa).
    """
    num_classes = len(class_names)
    model_path = out_path if out_path is not None else config.model_path(backbone)
    class_weights = data_mod.compute_class_weights(train_ds, num_classes)

    ls = getattr(config, "LABEL_SMOOTHING", 0.0)
    loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=ls)

    model = model_mod.build_model(num_classes, backbone=backbone)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LR_HEAD),
        loss=loss,
        metrics=["accuracy"],
    )

    histories = []
    print(f"\n=== [{backbone}] FASE 1: cabeza (backbone congelado) ===")
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, class_weight=class_weights,
        callbacks=_callbacks(model_path), verbose=2,
    )
    histories.append(h1)
    best_val = max(h1.history["val_accuracy"])

    if config.FINE_TUNE and do_fine_tune:
        print(f"\n=== [{backbone}] FASE 2: fine-tuning ===")
        model_mod.enable_fine_tuning(model, backbone=backbone)
        # Cosine decay: baja el LR suavemente durante el fine-tuning para afinar
        # sin desestabilizar los pesos preentrenados.
        steps = int(train_ds.cardinality().numpy())
        lr_sched = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=config.LR_FINE_TUNE,
            decay_steps=max(1, steps * ft_epochs),
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr_sched),
            loss=loss,
            metrics=["accuracy"],
        )
        h2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=ft_epochs, class_weight=class_weights,
            callbacks=_callbacks(model_path, best_threshold=best_val,
                                 use_lr_reduce=False), verbose=2,
        )
        histories.append(h2)

    if plot:
        _plot_history(histories, backbone)
    print(f"[{backbone}] mejor modelo guardado en {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Entrena el clasificador de cacao")
    parser.add_argument("--backbone", choices=config.ALL_BACKBONES,
                        default=config.DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_HEAD)
    parser.add_argument("--ft-epochs", type=int, default=config.EPOCHS_FINE_TUNE)
    parser.add_argument("--no-fine-tune", action="store_true")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)

    train_ds, val_ds, test_ds, class_names = data_mod.load_datasets()
    config.CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2)
    )

    model_path = train_backbone(
        args.backbone, train_ds, val_ds, class_names,
        epochs=args.epochs, ft_epochs=args.ft_epochs,
        do_fine_tune=not args.no_fine_tune,
    )

    print("\n=== Evaluación en el conjunto de TEST (imágenes no vistas) ===")
    best_model = tf.keras.models.load_model(model_path)
    loss, acc = best_model.evaluate(test_ds, verbose=0)
    print(f"[{args.backbone}] Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")


if __name__ == "__main__":
    main()
