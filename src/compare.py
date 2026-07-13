"""
Comparación de DOS modelos de detección para elegir cuál desplegar en la cámara.

Compara MobileNetV2 vs EfficientNetB0 entrenados sobre EXACTAMENTE el mismo
reparto de datos (comparación justa) y mide, sobre el conjunto de test:

  - Exactitud (accuracy) y F1 macro
  - Reporte por clase
  - Latencia de inferencia (ms/imagen) y FPS estimados en CPU  <- clave para cámara
  - Tamaño del modelo en disco y nº de parámetros

Luego elige el ganador y escribe models/best_model.json, que es el puntero que
usan la cámara (src/camera.py) y la predicción (src/predict.py). Así, el modelo
que se usa en la cámara queda JUSTIFICADO por la comparación.

Uso:
    python -m src.compare                 # entrena los que falten y compara
    python -m src.compare --retrain       # reentrena ambos desde cero
    python -m src.compare --epochs 25 --ft-epochs 12
    python -m src.compare --criterio latencia   # prioriza velocidad en el empate

Criterio de selección (por defecto 'accuracy'): gana el de mayor exactitud;
en caso de empate, el de mayor F1 macro y, si persiste, el más rápido.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import data as data_mod  # noqa: E402
from src import train as train_mod  # noqa: E402


def _measure_latency(model, sample, n_runs=100, warmup=10):
    """Latencia media por imagen (batch=1), como en la cámara."""
    x = tf.convert_to_tensor(sample[None, ...], dtype=tf.float32)
    for _ in range(warmup):
        model(x, training=False)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(x, training=False)
    elapsed = time.perf_counter() - t0
    ms_per_img = (elapsed / n_runs) * 1000.0
    return ms_per_img, 1000.0 / ms_per_img  # ms, fps


def evaluate_model(model, test_ds, class_names):
    from sklearn.metrics import (accuracy_score, classification_report,
                                 f1_score)

    y_true, y_pred, one_sample = [], [], None
    for x, y in test_ds:
        if one_sample is None:
            one_sample = x[0].numpy()
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, zero_division=0
    )
    ms, fps = _measure_latency(model, one_sample)
    return {"accuracy": acc, "f1_macro": f1, "report": report,
            "latency_ms": ms, "fps": fps}


def _model_stats(model, model_path):
    size_mb = Path(model_path).stat().st_size / 1e6
    params = model.count_params()
    return size_mb, params


def _get_model(backbone, train_ds, val_ds, class_names, retrain, epochs, ft_epochs):
    path = config.model_path(backbone)
    if path.exists() and not retrain:
        print(f"[{backbone}] cargando modelo existente: {path}")
        return path
    print(f"[{backbone}] entrenando...")
    return train_mod.train_backbone(
        backbone, train_ds, val_ds, class_names,
        epochs=epochs, ft_epochs=ft_epochs,
    )


def _pick_winner(results, criterio):
    """Devuelve el nombre del backbone ganador según el criterio."""
    def key_accuracy(b):
        r = results[b]
        return (r["accuracy"], r["f1_macro"], -r["latency_ms"])

    def key_latencia(b):
        r = results[b]
        return (-r["latency_ms"], r["accuracy"], r["f1_macro"])

    key = key_latencia if criterio == "latencia" else key_accuracy
    return max(results, key=key)


def _plot(results, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    names = list(results)
    acc = [results[n]["accuracy"] * 100 for n in names]
    f1 = [results[n]["f1_macro"] * 100 for n in names]
    ms = [results[n]["latency_ms"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(names))
    w = 0.35
    ax1.bar(x - w / 2, acc, w, label="Accuracy %")
    ax1.bar(x + w / 2, f1, w, label="F1 macro %")
    ax1.set_xticks(x); ax1.set_xticklabels(names)
    ax1.set_ylim(0, 100); ax1.set_title("Exactitud (test)"); ax1.legend()
    for i, v in enumerate(acc):
        ax1.text(i - w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=8)

    bars = ax2.bar(x, ms, color="#c0504d")
    ax2.set_xticks(x); ax2.set_xticklabels(names)
    ax2.set_title("Latencia por imagen (CPU, menos = mejor)")
    ax2.set_ylabel("ms/imagen")
    for b, v in zip(bars, ms):
        ax2.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f} ms",
                 ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\nGráfica comparativa guardada en {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compara dos modelos de detección")
    parser.add_argument("--retrain", action="store_true",
                        help="Reentrena ambos modelos aunque ya existan")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_HEAD)
    parser.add_argument("--ft-epochs", type=int, default=config.EPOCHS_FINE_TUNE)
    parser.add_argument("--criterio", choices=["accuracy", "latencia"],
                        default="accuracy",
                        help="Criterio para elegir el modelo de la cámara")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)

    train_ds, val_ds, test_ds, class_names = data_mod.load_datasets()
    config.CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2)
    )

    results = {}
    for backbone in config.AVAILABLE_BACKBONES:
        path = _get_model(backbone, train_ds, val_ds, class_names,
                          args.retrain, args.epochs, args.ft_epochs)
        model = tf.keras.models.load_model(path)
        print(f"\n=== Evaluando {backbone} en el test ===")
        res = evaluate_model(model, test_ds, class_names)
        res["size_mb"], res["params"] = _model_stats(model, path)
        res["path"] = str(path)
        results[backbone] = res
        print(res["report"])

    # ------------------------------------------------------------------ #
    # Tabla comparativa
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 68)
    print("COMPARACIÓN DE MODELOS (conjunto de test)")
    print("=" * 68)
    header = f"{'modelo':<16}{'accuracy':>10}{'f1_macro':>10}{'ms/img':>10}{'fps':>8}{'MB':>8}"
    print(header)
    print("-" * 68)
    for name, r in results.items():
        print(f"{name:<16}{r['accuracy']*100:>9.1f}%{r['f1_macro']*100:>9.1f}%"
              f"{r['latency_ms']:>10.1f}{r['fps']:>8.1f}{r['size_mb']:>8.1f}")

    winner = _pick_winner(results, args.criterio)
    w = results[winner]
    print("-" * 68)
    print(f"GANADOR ({args.criterio}): {winner}  "
          f"-> accuracy {w['accuracy']*100:.1f}%, {w['latency_ms']:.1f} ms/img")

    # ------------------------------------------------------------------ #
    # Puntero al modelo elegido para la cámara
    # ------------------------------------------------------------------ #
    best = {
        "backbone": winner,
        "archivo": config.model_path(winner).name,   # ruta portable (se reconstruye)
        "criterio": args.criterio,
        "accuracy": round(w["accuracy"], 4),
        "f1_macro": round(w["f1_macro"], 4),
        "latency_ms": round(w["latency_ms"], 2),
        "comparado_con": [b for b in results if b != winner],
    }
    config.BEST_MODEL_PATH.write_text(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"Modelo para la cámara registrado en {config.BEST_MODEL_PATH}")

    _plot(results, config.OUTPUTS_DIR / "comparacion_modelos.png")


if __name__ == "__main__":
    main()
