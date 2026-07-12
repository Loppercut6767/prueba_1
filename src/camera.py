"""
Clasificación de morfología de granos de cacao EN TIEMPO REAL usando una cámara.

Este es el objetivo final del proyecto: toma el vídeo de una webcam/cámara,
clasifica cada fotograma y superpone la morfología detectada.

Uso:
    python -m src.camera                 # cámara 0 (webcam por defecto)
    python -m src.camera --source 1      # otra cámara
    python -m src.camera --source video.mp4   # un archivo de vídeo
    python -m src.camera --roi           # analiza solo un recuadro central

Requisitos: opencv-python (ver requirements.txt). Necesita un entorno con
pantalla/cámara; no funciona en un servidor headless.

Controles:
    q  -> salir
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def load_model_and_classes():
    import tensorflow as tf

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No existe el modelo {config.MODEL_PATH}. Entrena primero con "
            "'python -m src.train'."
        )
    model = tf.keras.models.load_model(config.MODEL_PATH)
    class_names = json.loads(config.CLASS_NAMES_PATH.read_text())
    return model, class_names


def preprocess_frame(frame_bgr, roi_box=None):
    """Convierte un fotograma BGR de OpenCV al tensor que espera el modelo."""
    import cv2

    if roi_box is not None:
        x, y, w, h = roi_box
        frame_bgr = frame_bgr[y:y + h, x:x + w]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (config.IMG_SIZE, config.IMG_SIZE))
    return np.expand_dims(resized.astype("float32"), 0)  # reescalado va en el modelo


def main():
    import cv2

    parser = argparse.ArgumentParser(description="Clasificación de cacao en tiempo real")
    parser.add_argument("--source", default="0",
                        help="Índice de cámara (0,1,...) o ruta a un vídeo")
    parser.add_argument("--roi", action="store_true",
                        help="Analizar solo un recuadro central de la imagen")
    parser.add_argument("--conf", type=float, default=0.0,
                        help="Umbral mínimo de confianza para mostrar etiqueta")
    args = parser.parse_args()

    model, class_names = load_model_and_classes()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente de vídeo: {args.source}")

    print("Cámara abierta. Pulsa 'q' para salir.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        roi_box = None
        if args.roi:
            side = int(min(h, w) * 0.5)
            roi_box = ((w - side) // 2, (h - side) // 2, side, side)
            x, y, bw, bh = roi_box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        x_in = preprocess_frame(frame, roi_box)
        probs = model.predict(x_in, verbose=0)[0]
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        label = class_names[idx]
        desc = config.CLASS_DESCRIPTIONS_ES.get(label, label)

        if conf >= args.conf:
            text = f"{label} {conf:.0%} - {desc}"
            cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
            cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Cacao - morfologia", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
