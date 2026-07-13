"""
Definición de los modelos: transfer learning con DOS arquitecturas comparables.

  - "mobilenetv2":   ligero y muy rápido en CPU (ideal para cámara en tiempo real)
  - "efficientnetb0": algo más pesado, suele dar mayor exactitud

Ambos comparten la misma "cabeza" (GlobalAveragePooling + Dropout + Dense) y el
mismo bloque de aumento de datos, para que la comparación sea justa: lo único que
cambia es el backbone.

Cada modelo incluye DENTRO de sí mismo el aumento de datos (solo en entrenamiento)
y su preprocesado específico, de modo que la inferencia (predict/cámara) siempre
pasa la imagen redimensionada en rango 0-255, sin recordar detalles por modelo:
  - MobileNetV2 espera [-1, 1]  -> se añade una capa Rescaling.
  - EfficientNetB0 ya normaliza internamente (include_preprocessing=True) -> nada.
"""
from __future__ import annotations

import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _augmentation_block():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.15),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.1, value_range=(0, 255)),
        ],
        name="aumento_datos",
    )


def _build_backbone(backbone: str, input_shape):
    """Devuelve (backbone, capa_de_preprocesado_o_None) según la arquitectura."""
    if backbone == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        preprocess = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="preproc")
        return base, preprocess

    if backbone == "efficientnetb0":
        # include_preprocessing=True (por defecto): el modelo normaliza el rango
        # 0-255 internamente, así que no hace falta capa de preprocesado extra.
        base = tf.keras.applications.EfficientNetB0(
            input_shape=input_shape, include_top=False, weights="imagenet"
        )
        return base, None

    raise ValueError(
        f"Backbone desconocido: {backbone}. Opciones: {config.AVAILABLE_BACKBONES}"
    )


def build_model(num_classes: int, backbone: str = config.DEFAULT_BACKBONE) -> tf.keras.Model:
    input_shape = (config.IMG_SIZE, config.IMG_SIZE, 3)
    inputs = layers.Input(shape=input_shape, name="imagen")

    x = _augmentation_block()(inputs)

    base, preprocess = _build_backbone(backbone, input_shape)
    if preprocess is not None:
        x = preprocess(x)
    base.trainable = False  # fase 1: congelado

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="salida")(x)

    return tf.keras.Model(inputs, outputs, name=f"cacao_{backbone}")


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Devuelve la sub-red backbone dentro del modelo (para fine-tuning).

    Es el submodelo anidado que NO es el bloque de aumento de datos. Se excluye
    'aumento_datos' explícitamente (y los Sequential) porque un Sequential
    también es subclase de tf.keras.Model y, si no, se devolvería por error.
    """
    for layer in model.layers:
        if (isinstance(layer, tf.keras.Model)
                and not isinstance(layer, tf.keras.Sequential)
                and layer.name != "aumento_datos"):
            return layer
    raise ValueError("No se encontró el backbone (submodelo) en el modelo.")


def enable_fine_tuning(model: tf.keras.Model, backbone: str = config.DEFAULT_BACKBONE):
    """Descongela las capas superiores del backbone para el fine-tuning.

    IMPORTANTE: se mantienen CONGELADAS las capas BatchNormalization. Si se
    descongelan, sus estadísticas móviles (media/varianza) se recalculan con un
    batch pequeño y muy aumentado, lo que desestabiliza el modelo y hace que el
    fine-tuning empeore el resultado en vez de mejorarlo. Congelarlas es la
    práctica recomendada para transfer learning.
    """
    base = get_backbone(model)
    base.trainable = True
    fine_tune_at = config.FINE_TUNE_AT.get(backbone, 100)
    for i, layer in enumerate(base.layers):
        if i < fine_tune_at or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model
