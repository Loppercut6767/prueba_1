"""
Entrena un clasificador YOLO ligero (yolo11n-cls) sobre el dataset de cacao.

yolo11n-cls es el modelo más pequeño de la familia (~1.5 M parámetros), elegido
por su bajo consumo de recursos. Trae aumento de datos e early-stopping propios.

Uso:
    python train_yolo.py                 # entrena con los valores de config.py
    python train_yolo.py --epochs 40     # menos épocas
    python train_yolo.py --model yolo11s-cls.pt   # variante algo mayor

Los pesos entrenados quedan en runs/<name>/weights/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Entrena YOLO-cls sobre cacao")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--imgsz", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch", type=int, default=config.BATCH)
    parser.add_argument("--name", default="cacao_yolo11n")
    args = parser.parse_args()

    if not config.DATASET_DIR.exists():
        raise FileNotFoundError(
            f"No existe {config.DATASET_DIR}. Ejecuta primero 'python prepare_data.py'."
        )

    from ultralytics import YOLO

    model = YOLO(args.model)  # descarga los pesos preentrenados si hace falta
    model.train(
        data=str(config.DATASET_DIR),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=config.SEED,
        device="cpu",
        project=str(config.RUNS_DIR),
        name=args.name,
        exist_ok=True,
        patience=15,          # early stopping
        verbose=True,
    )
    best = config.RUNS_DIR / args.name / "weights" / "best.pt"
    print(f"\nModelo entrenado guardado en {best}")


if __name__ == "__main__":
    main()
