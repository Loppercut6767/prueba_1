"""
Entrenamiento del clasificador de morfología de granos de cacao.

Uso:
    python -m src.train                 # entrenamiento completo (cabeza + fine-tuning)
    python -m src.train --no-fine-tune  # solo la cabeza (más rápido)
    python -m src.train --epochs 10     # menos épocas de la fase 1

Al terminar guarda:
    models/cacao_mobilenetv2.keras   -> modelo entrenado
    models/class_names.json          -> orden de las clases
    outputs/history.png              -> curvas de accuracy/loss
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


def _callbacks(best_threshold: float | None = None):
    """Callbacks de entrenamiento.

    `best_threshold` fija el umbral inicial del ModelCheckpoint. En la fase 2
    (fine-tuning) se pasa el mejor val_accuracy de la fase 1, de modo que el
    checkpoint SOLO sobrescriba el modelo guardado si el fine-tuning realmente
    lo mejora. Así nunca se pierde el mejor modelo de la fase anterior.
    """
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(config.MODEL_PATH), monitor="val_accuracy",
            save_best_only=True, verbose=0,
            initial_value_threshold=best_threshold,
        ),
    ]


def _plot_history(histories):
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
    ax1.set_title("Accuracy"); ax1.set_xlabel("época"); ax1.legend()
    ax2.plot(loss, label="train"); ax2.plot(val_loss, label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("época"); ax2.legend()
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "history.png"
    fig.savefig(out, dpi=120)
    print(f"Curvas guardadas en {out}")


def main():
    parser = argparse.ArgumentParser(description="Entrena el clasificador de cacao")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_HEAD)
    parser.add_argument("--ft-epochs", type=int, default=config.EPOCHS_FINE_TUNE)
    parser.add_argument("--no-fine-tune", action="store_true")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)

    train_ds, val_ds, test_ds, class_names = data_mod.load_datasets()
    num_classes = len(class_names)

    # Guarda el orden de las clases para la inferencia
    config.CLASS_NAMES_PATH.write_text(json.dumps(class_names, ensure_ascii=False, indent=2))

    class_weights = data_mod.compute_class_weights(train_ds, num_classes)
    print("Pesos por clase:", {class_names[i]: round(w, 2) for i, w in class_weights.items()})

    model = model_mod.build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LR_HEAD),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    histories = []
    print("\n=== FASE 1: entrenando la cabeza (backbone congelado) ===")
    h1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=args.epochs, class_weight=class_weights,
        callbacks=_callbacks(), verbose=2,
    )
    histories.append(h1)
    best_val = max(h1.history["val_accuracy"])

    if config.FINE_TUNE and not args.no_fine_tune:
        print("\n=== FASE 2: fine-tuning de las capas superiores del backbone ===")
        model_mod.enable_fine_tuning(model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(config.LR_FINE_TUNE),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        # El checkpoint solo sobrescribirá si supera el mejor val de la fase 1
        h2 = model.fit(
            train_ds, validation_data=val_ds,
            epochs=args.ft_epochs, class_weight=class_weights,
            callbacks=_callbacks(best_threshold=best_val), verbose=2,
        )
        histories.append(h2)

    # El mejor modelo (de cualquiera de las dos fases) ya está en disco gracias
    # al ModelCheckpoint. Lo recargamos para evaluar EXACTAMENTE ese modelo.
    print(f"\nMejor modelo guardado en {config.MODEL_PATH}")
    best_model = tf.keras.models.load_model(config.MODEL_PATH)

    print("\n=== Evaluación en el conjunto de TEST (imágenes no vistas) ===")
    loss, acc = best_model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {acc:.4f} | Test loss: {loss:.4f}")

    _plot_history(histories)


if __name__ == "__main__":
    main()
