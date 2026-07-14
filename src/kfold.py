"""
Validación cruzada estratificada (k-fold) para estimar la exactitud REAL del
modelo sin el ruido de un único conjunto de test pequeño (93 imágenes).

Con datasets pequeños, un solo split test da un número inestable (±varias
unidades). El k-fold entrena el modelo k veces sobre particiones distintas y
promedia el resultado -> estimación mucho más fiable, con su desviación típica.
Es la forma correcta de afirmar "el modelo alcanza ~X %".

Uso:
    python -m src.kfold                                  # 5 folds, EfficientNetB0
    python -m src.kfold --folds 5 --backbone efficientnetb0
    python -m src.kfold --epochs 20 --ft-epochs 12 --tta

NOTA: entrena el modelo una vez por fold, así que es lento (pensado para correr
en segundo plano). No modifica los modelos desplegados en models/.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import data as data_mod  # noqa: E402
from src import train as train_mod  # noqa: E402
from src.compare import _predict_tta  # noqa: E402


def _eval(model, ds, tta):
    from sklearn.metrics import accuracy_score, f1_score
    y_true, y_pred = [], []
    for x, y in ds:
        probs = _predict_tta(model, x) if tta else model.predict(x, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(y.numpy(), axis=1))
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="macro")


def main():
    parser = argparse.ArgumentParser(description="Validación cruzada k-fold")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--backbone", choices=config.AVAILABLE_BACKBONES,
                        default="efficientnetb0")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--ft-epochs", type=int, default=12)
    parser.add_argument("--tta", action="store_true")
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(config.SEED)

    filepaths, labels, class_names = data_mod._list_files_and_labels()
    filepaths = np.array(filepaths)
    labels = np.array(labels)
    num_classes = len(class_names)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=config.SEED)
    accs, f1s = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(filepaths, labels), 1):
        print(f"\n########## FOLD {fold}/{args.folds} ({args.backbone}) ##########")
        # dentro del train de este fold, separa una validación para early stopping
        tr_idx, val_idx = train_test_split(
            train_idx, test_size=0.15, stratify=labels[train_idx],
            random_state=config.SEED,
        )
        train_ds = data_mod._make_ds(list(filepaths[tr_idx]), list(labels[tr_idx]),
                                     num_classes, shuffle=True)
        val_ds = data_mod._make_ds(list(filepaths[val_idx]), list(labels[val_idx]),
                                   num_classes, shuffle=False)
        test_ds = data_mod._make_ds(list(filepaths[test_idx]), list(labels[test_idx]),
                                    num_classes, shuffle=False)

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / f"fold{fold}.keras"
            train_mod.train_backbone(
                args.backbone, train_ds, val_ds, class_names,
                epochs=args.epochs, ft_epochs=args.ft_epochs,
                out_path=out, plot=False,
            )
            model = tf.keras.models.load_model(out)
            acc, f1 = _eval(model, test_ds, args.tta)

        accs.append(acc); f1s.append(f1)
        print(f"FOLD {fold}: accuracy={acc:.4f}  f1_macro={f1:.4f}")
        del model
        tf.keras.backend.clear_session()

    accs, f1s = np.array(accs), np.array(f1s)
    print("\n" + "=" * 60)
    print(f"RESULTADO K-FOLD ({args.folds} folds, {args.backbone}"
          f"{', TTA' if args.tta else ''})")
    print("=" * 60)
    for i, (a, f) in enumerate(zip(accs, f1s), 1):
        print(f"  fold {i}: acc={a*100:5.1f}%  f1={f*100:5.1f}%")
    print("-" * 60)
    print(f"  Accuracy media: {accs.mean()*100:.1f}%  (± {accs.std()*100:.1f})")
    print(f"  F1 macro media: {f1s.mean()*100:.1f}%  (± {f1s.std()*100:.1f})")
    print(f"  Rango accuracy: {accs.min()*100:.1f}% – {accs.max()*100:.1f}%")


if __name__ == "__main__":
    main()
