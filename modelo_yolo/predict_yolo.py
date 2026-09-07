"""
Inferencia con el modelo YOLO entrenado sobre imágenes arbitrarias.

Uso:
    python predict_yolo.py --image ruta/a/grano.jpg
    python predict_yolo.py --dir ruta/a/carpeta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    parser = argparse.ArgumentParser(description="Predice morfología con YOLO-cls")
    parser.add_argument("--weights",
                        default=str(config.RUNS_DIR / "cacao_yolo11n" / "weights" / "best.pt"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str)
    group.add_argument("--dir", type=str)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)

    if args.image:
        paths = [args.image]
    else:
        paths = [str(p) for p in sorted(Path(args.dir).iterdir())
                 if p.suffix.lower() in IMG_EXT]

    results = model.predict(paths, imgsz=config.IMG_SIZE, device="cpu", verbose=False)
    for path, r in zip(paths, results):
        top1 = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        print(f"{Path(path).name:<30} -> {model.names[top1]:<11} ({conf:.1%})")


if __name__ == "__main__":
    main()
