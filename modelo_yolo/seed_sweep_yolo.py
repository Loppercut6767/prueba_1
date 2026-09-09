"""
Barrido de semillas para YOLO11n-cls (misma idea que el de modelo_actual).

Entrena varias veces con distintas semillas (reparto fijo, generado por
prepare_data.py) a imgsz 160 (apto para Raspberry Pi 5) y reporta media ± std
del accuracy en test, además de la semilla recomendada por validación.

Uso:
    python seed_sweep_yolo.py --seeds 0 1 2 3 4 --imgsz 160 --epochs 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _test_metrics(model, imgsz):
    from sklearn.metrics import accuracy_score, f1_score
    names = [model.names[i] for i in range(len(model.names))]
    idx = {n: i for i, n in enumerate(names)}
    files, y_true = [], []
    for cls_dir in sorted((config.DATASET_DIR / "test").iterdir()):
        if cls_dir.is_dir():
            for img in sorted(cls_dir.iterdir()):
                if img.suffix.lower() in IMG_EXT:
                    files.append(str(img)); y_true.append(idx[cls_dir.name])
    y_pred = [int(r.probs.top1) for r in
              model.predict(files, imgsz=imgsz, device="cpu", verbose=False)]
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro"))


def main():
    parser = argparse.ArgumentParser(description="Seed sweep YOLO11n-cls")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--imgsz", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    if not config.DATASET_DIR.exists():
        raise FileNotFoundError("Ejecuta prepare_data.py primero.")

    from ultralytics import YOLO

    runs = []
    for seed in args.seeds:
        print(f"\n########## yolo11n-cls | seed={seed} | imgsz={args.imgsz} ##########")
        model = YOLO(config.MODEL)
        res = model.train(
            data=str(config.DATASET_DIR), epochs=args.epochs, imgsz=args.imgsz,
            batch=config.BATCH, seed=seed, device="cpu",
            project=str(config.RUNS_DIR), name=f"sweep_s{seed}", exist_ok=True,
            patience=15, verbose=False, plots=False,
        )
        # accuracy de validación que reporta Ultralytics (top1)
        val_acc = float(getattr(res, "top1", 0.0) or
                        res.results_dict.get("metrics/accuracy_top1", 0.0))
        best = YOLO(str(config.RUNS_DIR / f"sweep_s{seed}" / "weights" / "best.pt"))
        test_acc, test_f1 = _test_metrics(best, args.imgsz)
        runs.append({"seed": seed, "val_acc": val_acc,
                     "test_acc": test_acc, "test_f1": test_f1})
        print(f"  seed={seed}: val_acc={val_acc:.4f} test_acc={test_acc:.4f} "
              f"test_f1={test_f1:.4f}")

    test_accs = np.array([r["test_acc"] for r in runs])
    best_val = max(runs, key=lambda r: r["val_acc"])
    best_test = max(runs, key=lambda r: r["test_acc"])
    print("\n" + "=" * 60)
    print(f"BARRIDO YOLO11n-cls (imgsz={args.imgsz}, seeds={args.seeds})")
    print("=" * 60)
    print(f"  test accuracy media={test_accs.mean()*100:.1f}% ± {test_accs.std()*100:.1f}"
          f"  (min {test_accs.min()*100:.1f}%, max {test_accs.max()*100:.1f}%)")
    print(f"  seed recomendada (mejor val): {best_val['seed']} -> test {best_val['test_acc']*100:.1f}%")
    print(f"  (optimista) mejor test: seed {best_test['seed']} -> {best_test['test_acc']*100:.1f}%")

    out = config.ROOT_DIR / "seed_sweep_yolo.json"
    out.write_text(json.dumps({"imgsz": args.imgsz, "seeds": args.seeds,
                               "resultados": runs}, ensure_ascii=False, indent=2))
    print(f"\nResumen guardado en {out}")


if __name__ == "__main__":
    main()
