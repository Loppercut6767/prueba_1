"""
Prepara el dataset en formato YOLO-clasificación a partir del MISMO dataset que
usa modelo_actual (data/raw/Cocoa Beans/).

YOLO-cls espera esta estructura:
    dataset/
      train/<clase>/*.jpg
      val/<clase>/*.jpg
      test/<clase>/*.jpg

El reparto es estratificado 70/15/15 con semilla fija (igual proporción que
modelo_actual) para que la comparación sea justa. Se copian las imágenes con el
nombre de clase en español.

Uso:
    python prepare_data.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402


def _list_files_and_labels():
    if not config.RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {config.RAW_DATA_DIR}")
    folders = sorted(p for p in config.RAW_DATA_DIR.iterdir() if p.is_dir())
    files, labels = [], []
    for folder in folders:
        es = config.CLASS_MAP_ES.get(folder.name, folder.name)
        # ordenado para un reparto reproducible
        for img in sorted(folder.glob("*.jpg")):
            files.append(img)
            labels.append(es)
    return files, labels


def main():
    files, labels = _list_files_and_labels()
    classes = sorted(set(labels))
    print(f"Imágenes: {len(files)} | clases: {classes}")

    # 1) separa test; 2) del resto separa val -> ambos estratificados
    x_tr, x_te, y_tr, y_te = train_test_split(
        files, labels, test_size=config.TEST_SPLIT,
        stratify=labels, random_state=config.SEED,
    )
    val_rel = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)
    x_tr, x_va, y_tr, y_va = train_test_split(
        x_tr, y_tr, test_size=val_rel, stratify=y_tr, random_state=config.SEED,
    )

    splits = {"train": (x_tr, y_tr), "val": (x_va, y_va), "test": (x_te, y_te)}

    if config.DATASET_DIR.exists():
        shutil.rmtree(config.DATASET_DIR)
    for split, (xs, ys) in splits.items():
        for cls in classes:
            (config.DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)
        for src, cls in zip(xs, ys):
            dst = config.DATASET_DIR / split / cls / src.name
            shutil.copy(src, dst)
        print(f"  {split}: {len(xs)} imágenes")

    print(f"\nDataset YOLO listo en {config.DATASET_DIR}")


if __name__ == "__main__":
    main()
