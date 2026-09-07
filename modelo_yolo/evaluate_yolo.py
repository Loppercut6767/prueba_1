"""
Evalúa el modelo YOLO entrenado sobre el conjunto de TEST (mismo split que
modelo_actual) y reporta accuracy, F1 macro, reporte por clase y matriz de
confusión, para comparar directamente con el modelo actual.

Uso:
    python evaluate_yolo.py
    python evaluate_yolo.py --weights runs/cacao_yolo11n/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _plot_confusion(cm, class_names, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
    ax.set_title("Matriz de confusión YOLO (test)")
    thr = cm.max() / 2.0 if cm.max() else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center",
                    color="white" if cm[i, j] > thr else "black")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out, dpi=120)
    print(f"Matriz de confusión guardada en {out}")


def main():
    parser = argparse.ArgumentParser(description="Evalúa YOLO-cls sobre el test")
    parser.add_argument("--weights",
                        default=str(config.RUNS_DIR / "cacao_yolo11n" / "weights" / "best.pt"))
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"No existe {weights}. Entrena primero con train_yolo.py")

    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # nombres de clase en el orden interno del modelo
    names = [model.names[i] for i in range(len(model.names))]
    name_to_idx = {n: i for i, n in enumerate(names)}

    test_dir = config.DATASET_DIR / "test"
    files, y_true = [], []
    for cls_dir in sorted(test_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.iterdir()):
            if img.suffix.lower() in IMG_EXT:
                files.append(str(img)); y_true.append(name_to_idx[cls_dir.name])

    # predicción por lotes
    y_pred = []
    results = model.predict(files, imgsz=config.IMG_SIZE, device="cpu", verbose=False)
    for r in results:
        y_pred.append(int(r.probs.top1))

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    print(f"\n=== YOLO ({weights.parent.parent.name}) — test ({len(files)} imágenes) ===")
    print(classification_report(y_true, y_pred, target_names=names, digits=3, zero_division=0))
    print(f"Accuracy: {acc:.4f} | F1 macro: {f1:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de confusión:\n", cm)
    _plot_confusion(cm, names, config.ROOT_DIR / "confusion_matrix_yolo.png")


if __name__ == "__main__":
    main()
