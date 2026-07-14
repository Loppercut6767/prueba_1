"""
Predicción de la morfología sobre imágenes arbitrarias / aleatorias.

Uso:
    # una imagen
    python -m src.predict --image ruta/a/grano.jpg

    # una carpeta completa
    python -m src.predict --dir ruta/a/carpeta

    # N imágenes aleatorias del propio dataset (validación rápida del funcionamiento)
    python -m src.predict --random 12

Con --random genera además una cuadrícula en outputs/predicciones_aleatorias.png
con la imagen, la predicción y la etiqueta real, útil para comprobar de un
vistazo que el modelo funciona antes de conectar la cámara.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def load_model_and_classes(backbone: str | None = None):
    path = config.resolve_model_path(backbone)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el modelo {path}. Entrena primero con 'python -m src.train' "
            "o compara los modelos con 'python -m src.compare'."
        )
    print(f"Usando modelo: {path.name}")
    model = tf.keras.models.load_model(path)
    class_names = json.loads(config.CLASS_NAMES_PATH.read_text())
    return model, class_names


def preprocess(path: str) -> np.ndarray:
    from src.data import resize_image  # mismo redimensionado que en entrenamiento

    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = resize_image(img)
    img = tf.cast(img, tf.float32)  # el reescalado ocurre dentro del modelo
    return tf.expand_dims(img, 0)


def predict_one(model, class_names, path: str):
    x = preprocess(path)
    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))
    label = class_names[idx]
    desc = config.CLASS_DESCRIPTIONS_ES.get(label, label)
    return label, desc, float(probs[idx]), probs


def _print_result(path, label, desc, conf, probs, class_names):
    print(f"\n{Path(path).name}")
    print(f"  -> {label} ({desc})  confianza={conf:.1%}")
    order = np.argsort(probs)[::-1]
    for i in order:
        print(f"     {class_names[i]:<12} {probs[i]:.1%}")


def _random_from_dataset(n):
    files = list(config.DATA_DIR.glob("*/*.jpg"))
    random.seed()  # aleatorio de verdad en cada ejecución
    return random.sample(files, min(n, len(files)))


def _grid(results):
    """results: lista de (path, label, conf, real_label)"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        print("matplotlib/PIL no disponibles; se omite la cuadrícula.")
        return

    n = len(results)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (path, label, conf, real) in zip(axes, results):
        ax.imshow(Image.open(path).convert("RGB"))
        ok = (real is None) or (real == label)
        color = "green" if ok else "red"
        title = f"pred: {label} ({conf:.0%})"
        if real is not None:
            title += f"\nreal: {real}"
        ax.set_title(title, color=color, fontsize=9)
    fig.tight_layout()
    out = config.OUTPUTS_DIR / "predicciones_aleatorias.png"
    fig.savefig(out, dpi=120)
    print(f"\nCuadrícula guardada en {out}")


def main():
    parser = argparse.ArgumentParser(description="Predice la morfología del grano de cacao")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="Ruta a una imagen")
    group.add_argument("--dir", type=str, help="Carpeta con imágenes")
    group.add_argument("--random", type=int, help="N imágenes aleatorias del dataset")
    parser.add_argument("--backbone", choices=config.ALL_BACKBONES, default=None,
                        help="Forzar un modelo concreto (por defecto: el ganador de la comparación)")
    args = parser.parse_args()

    model, class_names = load_model_and_classes(args.backbone)

    if args.image:
        label, desc, conf, probs = predict_one(model, class_names, args.image)
        _print_result(args.image, label, desc, conf, probs, class_names)

    elif args.dir:
        paths = sorted(Path(args.dir).glob("*"))
        paths = [p for p in paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        for p in paths:
            label, desc, conf, probs = predict_one(model, class_names, str(p))
            _print_result(str(p), label, desc, conf, probs, class_names)

    elif args.random:
        files = _random_from_dataset(args.random)
        results, correct = [], 0
        for p in files:
            # etiqueta real = nombre de carpeta traducido a español
            real = config.CLASS_MAP_ES.get(p.parent.name, p.parent.name)
            label, desc, conf, probs = predict_one(model, class_names, str(p))
            _print_result(str(p), label, desc, conf, probs, class_names)
            results.append((str(p), label, conf, real))
            correct += int(real == label)
        print(f"\nAciertos: {correct}/{len(files)} ({correct/len(files):.1%})")
        _grid(results)


if __name__ == "__main__":
    main()
