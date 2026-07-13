"""
Evaluación del modelo entrenado sobre el conjunto de test (imágenes aleatorias
no vistas). Genera un reporte de clasificación y una matriz de confusión.

Uso:
    python -m src.evaluate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import data as data_mod  # noqa: E402


def _load_class_names():
    if config.CLASS_NAMES_PATH.exists():
        return json.loads(config.CLASS_NAMES_PATH.read_text())
    return None


def _plot_confusion(cm, class_names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib no disponible; se omite la matriz de confusión.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión (test)")
    thresh = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=120)
    print(f"Matriz de confusión guardada en {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evalúa un modelo sobre el test")
    parser.add_argument("--backbone", choices=config.AVAILABLE_BACKBONES, default=None,
                        help="Modelo a evaluar (por defecto: el ganador de la comparación)")
    args = parser.parse_args()

    model_path = config.resolve_model_path(args.backbone)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe el modelo {model_path}. Entrena primero con "
            "'python -m src.train' o compara con 'python -m src.compare'."
        )

    from sklearn.metrics import classification_report, confusion_matrix

    print(f"Evaluando modelo: {model_path.name}")
    model = tf.keras.models.load_model(model_path)
    _, _, test_ds, class_names = data_mod.load_datasets()
    saved_names = _load_class_names()
    if saved_names:
        class_names = saved_names

    y_true, y_pred = [], []
    for x, y in test_ds:
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))

    print("\n=== Reporte de clasificación (test) ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))

    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de confusión:\n", cm)
    _plot_confusion(cm, class_names)


if __name__ == "__main__":
    main()
