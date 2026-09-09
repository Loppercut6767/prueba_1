"""
Barrido de semillas (seed sweep) para los 3 modelos Keras.

Objetivo: ver cuánto varía el resultado según la semilla y, de forma honesta,
reportar la MEDIA ± DESVIACIÓN (lo robusto) además del mejor caso.

Metodología (para que la comparación sea justa):
  - El REPARTO del dataset es fijo (mismo train/val/test para todos), así todos
    los modelos y semillas se evalúan sobre EXACTAMENTE el mismo test.
  - Solo varía la semilla de ENTRENAMIENTO (inicialización + aumento), que es lo
    que de verdad significa "cambiar la seed".

Sobre "la mejor seed": elegir la semilla con mejor TEST es hacer trampa al test
(se sobreajusta a esas 93 imágenes). La semilla RECOMENDADA se elige por mejor
VALIDACIÓN (defendible); el mejor test se reporta solo como referencia optimista.

Uso:
    python -m src.seed_sweep                        # 3 modelos, semillas 0..4
    python -m src.seed_sweep --seeds 0 1 2 3 42
    python -m src.seed_sweep --backbones mobilenetv2 efficientnetb0
    python -m src.seed_sweep --epochs 18 --ft-epochs 10
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import data as data_mod  # noqa: E402
from src import train as train_mod  # noqa: E402


def _acc_f1(model, ds):
    from sklearn.metrics import accuracy_score, f1_score
    y_true, y_pred = [], []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def main():
    parser = argparse.ArgumentParser(description="Barrido de semillas de los 3 modelos")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--backbones", nargs="+", default=list(config.AVAILABLE_BACKBONES))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--ft-epochs", type=int, default=10)
    args = parser.parse_args()

    # Reparto FIJO (semilla de split = config.SEED), cargado una sola vez
    train_ds, val_ds, test_ds, class_names = data_mod.load_datasets()
    config.CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2)
    )

    # results[backbone] = lista de dicts {seed, val_acc, test_acc, test_f1}
    results = {b: [] for b in args.backbones}

    for backbone in args.backbones:
        for seed in args.seeds:
            print(f"\n########## {backbone} | seed={seed} ##########")
            tf.keras.utils.set_random_seed(seed)   # varía init + aumento
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / f"{backbone}_s{seed}.keras"
                train_mod.train_backbone(
                    backbone, train_ds, val_ds, class_names,
                    epochs=args.epochs, ft_epochs=args.ft_epochs,
                    out_path=out, plot=False,
                )
                model = tf.keras.models.load_model(out)
                val_acc, _ = _acc_f1(model, val_ds)
                test_acc, test_f1 = _acc_f1(model, test_ds)
            results[backbone].append(
                {"seed": seed, "val_acc": val_acc,
                 "test_acc": test_acc, "test_f1": test_f1}
            )
            print(f"  seed={seed}: val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
                  f"test_f1={test_f1:.4f}")
            del model
            tf.keras.backend.clear_session()

    # ------------------------------------------------------------------ #
    # Resumen
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 72)
    print(f"BARRIDO DE SEMILLAS (imgsz={config.IMG_SIZE}, seeds={args.seeds})")
    print("=" * 72)
    summary = {}
    for backbone, runs in results.items():
        test_accs = np.array([r["test_acc"] for r in runs])
        # semilla recomendada: la de mejor VALIDACIÓN (no test)
        best_val_run = max(runs, key=lambda r: r["val_acc"])
        best_test_run = max(runs, key=lambda r: r["test_acc"])
        summary[backbone] = {
            "test_mean": float(test_accs.mean()),
            "test_std": float(test_accs.std()),
            "test_min": float(test_accs.min()),
            "test_max": float(test_accs.max()),
            "seed_recomendada_por_val": best_val_run["seed"],
            "test_de_esa_seed": best_val_run["test_acc"],
            "mejor_test_optimista": best_test_run["test_acc"],
            "seed_mejor_test": best_test_run["seed"],
        }
        print(f"\n{backbone}:")
        print(f"  test accuracy  media={test_accs.mean()*100:.1f}%  "
              f"± {test_accs.std()*100:.1f}   (min {test_accs.min()*100:.1f}%, "
              f"max {test_accs.max()*100:.1f}%)")
        print(f"  seed recomendada (mejor val): {best_val_run['seed']}  "
              f"-> test {best_val_run['test_acc']*100:.1f}%")
        print(f"  (referencia optimista) mejor test: seed {best_test_run['seed']} "
              f"-> {best_test_run['test_acc']*100:.1f}%")

    out_json = config.OUTPUTS_DIR / "seed_sweep.json"
    out_json.write_text(json.dumps(
        {"seeds": args.seeds, "imgsz": config.IMG_SIZE,
         "resultados": results, "resumen": summary},
        ensure_ascii=False, indent=2))
    print(f"\nResumen guardado en {out_json}")

    # Mejor modelo por media (robusto)
    best_model = max(summary, key=lambda b: summary[b]["test_mean"])
    print(f"\nMejor modelo por accuracy media (robusto): {best_model} "
          f"({summary[best_model]['test_mean']*100:.1f}%)")


if __name__ == "__main__":
    main()
