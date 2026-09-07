"""
Ensamble de modelos: promedia las probabilidades de EfficientNetB0 + EfficientNetB2
para exprimir algo más de exactitud que cualquiera por separado.

Idea: dos redes distintas cometen errores distintos; promediar sus probabilidades
suele dar una predicción más robusta (+1–3 puntos habituales).

Entrena los miembros que falten (a la resolución de config.IMG_SIZE, p. ej. 224),
evalúa cada uno por separado y luego el ensamble, todo sobre el MISMO test.

Uso:
    python -m src.ensemble                    # entrena lo que falte y evalúa (con TTA)
    python -m src.ensemble --retrain          # reentrena ambos miembros
    python -m src.ensemble --no-tta           # sin Test-Time Augmentation
    python -m src.ensemble --epochs 20 --ft-epochs 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import data as data_mod  # noqa: E402
from src import train as train_mod  # noqa: E402
from src.compare import _predict_tta  # noqa: E402


def _member_probs(model, x, tta):
    return _predict_tta(model, x) if tta else model.predict(x, verbose=0)


def _metrics(y_true, y_pred, class_names):
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   digits=3, zero_division=0)
    return acc, f1, report


def main():
    parser = argparse.ArgumentParser(description="Ensamble EfficientNetB0 + B2")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS_HEAD)
    parser.add_argument("--ft-epochs", type=int, default=config.EPOCHS_FINE_TUNE)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()
    tta = not args.no_tta

    tf.keras.utils.set_random_seed(config.SEED)
    train_ds, val_ds, test_ds, class_names = data_mod.load_datasets()
    config.CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2)
    )

    # 1) Asegura que cada miembro esté entrenado a la resolución actual
    members = {}
    for backbone in config.ENSEMBLE_MEMBERS:
        path = config.model_path(backbone)
        if not path.exists() or args.retrain:
            print(f"\n[{backbone}] entrenando a {config.IMG_SIZE}px...")
            train_mod.train_backbone(
                backbone, train_ds, val_ds, class_names,
                epochs=args.epochs, ft_epochs=args.ft_epochs,
            )
        else:
            print(f"[{backbone}] cargando modelo existente: {path.name}")
        members[backbone] = tf.keras.models.load_model(path)

    # 2) Recolecta probabilidades por miembro y del ensamble sobre el test
    y_true = []
    per_member_probs = {b: [] for b in members}
    for x, y in test_ds:
        y_true.extend(np.argmax(y.numpy(), axis=1))
        for b, m in members.items():
            per_member_probs[b].append(_member_probs(m, x, tta))
    y_true = np.array(y_true)
    per_member_probs = {b: np.concatenate(p, axis=0) for b, p in per_member_probs.items()}
    ensemble_probs = np.mean(list(per_member_probs.values()), axis=0)

    # 3) Métricas
    print("\n" + "=" * 60)
    print(f"ENSAMBLE (test, {'con TTA' if tta else 'sin TTA'})")
    print("=" * 60)
    for b, probs in per_member_probs.items():
        acc, f1, _ = _metrics(y_true, np.argmax(probs, axis=1), class_names)
        print(f"  {b:<16} accuracy={acc*100:5.1f}%   f1_macro={f1*100:5.1f}%")
    acc, f1, report = _metrics(y_true, np.argmax(ensemble_probs, axis=1), class_names)
    print("-" * 60)
    print(f"  {'ENSAMBLE B0+B2':<16} accuracy={acc*100:5.1f}%   f1_macro={f1*100:5.1f}%")
    print("\nReporte por clase (ensamble):")
    print(report)

    # 4) Registra el ensamble como opción de despliegue
    info = {
        "tipo": "ensamble",
        "miembros": config.ENSEMBLE_MEMBERS,
        "img_size": config.IMG_SIZE,
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1), 4),
        "tta": tta,
    }
    (config.MODELS_DIR / "ensemble.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2)
    )
    print(f"\nEnsamble registrado en {config.MODELS_DIR / 'ensemble.json'}")


if __name__ == "__main__":
    main()
