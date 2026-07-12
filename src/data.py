"""
Carga y preparación del dataset de granos de cacao.

- Lee las imágenes desde data/raw/Cocoa Beans/<clase>/*.jpg
- Aplica el mapeo de nombres a español (config.CLASS_MAP_ES)
- Hace un reparto ESTRATIFICADO train / val / test (aleatorio pero reproducible)
- Devuelve tf.data.Dataset listos para entrenar/evaluar

El reparto aleatorio del test cumple el requisito de "imágenes aleatorias para
validar el funcionamiento": son imágenes que el modelo NUNCA ve en el entrenamiento.
"""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

AUTOTUNE = tf.data.AUTOTUNE


def _list_files_and_labels():
    """Recorre las carpetas del dataset y devuelve (rutas, etiquetas, clases)."""
    if not config.DATA_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {config.DATA_DIR}.\n"
            "Descomprime el archivo del dataset dentro de data/raw/ de modo que "
            "exista la carpeta 'data/raw/Cocoa Beans/<clase>/*.jpg'."
        )

    folders = sorted(p for p in config.DATA_DIR.iterdir() if p.is_dir())
    if not folders:
        raise FileNotFoundError(f"No hay subcarpetas de clases en {config.DATA_DIR}")

    # Nombre de clase en español; ordenado para tener índices estables
    class_names = sorted(config.CLASS_MAP_ES.get(f.name, f.name) for f in folders)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    filepaths, labels = [], []
    for folder in folders:
        es_name = config.CLASS_MAP_ES.get(folder.name, folder.name)
        for img in folder.glob("*.jpg"):
            filepaths.append(str(img))
            labels.append(class_to_idx[es_name])

    return filepaths, labels, class_names


def _decode(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
    img = tf.cast(img, tf.float32)  # 0-255; el reescalado va dentro del modelo
    return img, label


def _make_ds(paths, labels, num_classes, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (x, tf.one_hot(y, num_classes)), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(config.BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


def load_datasets():
    """Devuelve (train_ds, val_ds, test_ds, class_names)."""
    filepaths, labels, class_names = _list_files_and_labels()
    num_classes = len(class_names)

    # 1) separa test; 2) del resto separa validación -> ambos estratificados
    x_train, x_test, y_train, y_test = train_test_split(
        filepaths, labels,
        test_size=config.TEST_SPLIT,
        stratify=labels,
        random_state=config.SEED,
    )
    val_relative = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train,
        test_size=val_relative,
        stratify=y_train,
        random_state=config.SEED,
    )

    print(f"Clases ({num_classes}): {class_names}")
    print(f"Train: {len(x_train)} | Val: {len(x_val)} | Test: {len(x_test)}")

    train_ds = _make_ds(x_train, y_train, num_classes, shuffle=True)
    val_ds = _make_ds(x_val, y_val, num_classes, shuffle=False)
    test_ds = _make_ds(x_test, y_test, num_classes, shuffle=False)
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_ds, num_classes):
    """Pesos por clase para compensar un ligero desbalance."""
    import numpy as np

    counts = np.zeros(num_classes)
    for _, y in train_ds.unbatch():
        counts[int(tf.argmax(y))] += 1
    total = counts.sum()
    weights = {i: total / (num_classes * c) if c > 0 else 0.0 for i, c in enumerate(counts)}
    return weights


if __name__ == "__main__":
    # Prueba rápida: cuenta ejemplos por split
    tr, va, te, names = load_datasets()
    for name, ds in [("train", tr), ("val", va), ("test", te)]:
        n = sum(int(b[0].shape[0]) for b in ds)
        print(f"{name}: {n} imágenes")
