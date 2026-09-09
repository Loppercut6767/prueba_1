"""
Finaliza los 3 modelos Keras para el despliegue en Raspberry Pi 5:

  1. Reentrena cada backbone con su SEMILLA RECOMENDADA (la de mejor validación,
     leída de outputs/seed_sweep.json) a la resolución actual (160 px) y lo guarda
     en models/cacao_<backbone>.keras.
  2. Mide para cada uno: accuracy de test, latencia (ms/imagen, batch=1),
     FPS, nº de parámetros y tamaño en disco.
  3. Escribe models/best_model.json apuntando al backbone indicado con --deploy.

Uso:
    python -m src.finalize --deploy efficientnetb2
    python -m src.finalize --deploy mobilenetv2      # opción más ligera para el Pi
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


def _recommended_seeds():
    p = config.OUTPUTS_DIR / "seed_sweep.json"
    if p.exists():
        r = json.loads(p.read_text())["resumen"]
        return {b: r[b]["seed_recomendada_por_val"] for b in r}
    return {b: config.SEED for b in config.AVAILABLE_BACKBONES}


def _latency(model, sample, n=60, warmup=8):
    x = tf.convert_to_tensor(sample[None, ...], dtype=tf.float32)
    for _ in range(warmup):
        model(x, training=False)
    t0 = time.perf_counter()
    for _ in range(n):
        model(x, training=False)
    ms = (time.perf_counter() - t0) / n * 1000
    return ms, 1000 / ms


def _test_acc(model, test_ds):
    from sklearn.metrics import accuracy_score, f1_score
    yt, yp = [], []
    one = None
    for x, y in test_ds:
        if one is None:
            one = x[0].numpy()
        p = model.predict(x, verbose=0)
        yp.extend(np.argmax(p, 1)); yt.extend(np.argmax(y.numpy(), 1))
    return accuracy_score(yt, yp), f1_score(yt, yp, average="macro"), one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", choices=config.AVAILABLE_BACKBONES,
                        default="efficientnetb2")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ft-epochs", type=int, default=12)
    args = parser.parse_args()

    seeds = _recommended_seeds()
    print("Semillas recomendadas:", seeds)

    train_ds, val_ds, test_ds, class_names = data_mod.load_datasets()
    config.CLASS_NAMES_PATH.write_text(json.dumps(class_names, ensure_ascii=False, indent=2))

    rows = {}
    for backbone in config.AVAILABLE_BACKBONES:
        seed = seeds.get(backbone, config.SEED)
        print(f"\n### Reentrenando {backbone} (seed={seed}) a {config.IMG_SIZE}px ###")
        tf.keras.utils.set_random_seed(seed)
        train_mod.train_backbone(backbone, train_ds, val_ds, class_names,
                                 epochs=args.epochs, ft_epochs=args.ft_epochs, plot=False)
        path = config.model_path(backbone)
        model = tf.keras.models.load_model(path)
        acc, f1, sample = _test_acc(model, test_ds)
        ms, fps = _latency(model, sample)
        rows[backbone] = {
            "seed": seed, "test_acc": acc, "test_f1": f1,
            "latency_ms": ms, "fps": fps,
            "params": int(model.count_params()),
            "size_mb": path.stat().st_size / 1e6,
        }
        del model; tf.keras.backend.clear_session()

    print("\n" + "=" * 78)
    print(f"MODELOS KERAS FINALES (imgsz={config.IMG_SIZE}, semilla recomendada por val)")
    print("=" * 78)
    print(f"{'modelo':<16}{'seed':>5}{'test_acc':>10}{'f1':>8}{'ms/img':>9}{'fps':>7}{'params':>11}{'MB':>7}")
    for b, r in rows.items():
        print(f"{b:<16}{r['seed']:>5}{r['test_acc']*100:>9.1f}%{r['test_f1']*100:>7.1f}%"
              f"{r['latency_ms']:>9.1f}{r['fps']:>7.1f}{r['params']:>11,}{r['size_mb']:>7.1f}")

    (config.OUTPUTS_DIR / "finalize.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2))

    # Despliegue
    dep = args.deploy
    best = {"backbone": dep, "archivo": config.model_path(dep).name,
            "criterio": "recomendado_para_pi5", "seed": rows[dep]["seed"],
            "test_acc": round(rows[dep]["test_acc"], 4),
            "latency_ms": round(rows[dep]["latency_ms"], 2),
            "img_size": config.IMG_SIZE}
    config.BEST_MODEL_PATH.write_text(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"\nDesplegado para la cámara: {dep} -> {config.BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
