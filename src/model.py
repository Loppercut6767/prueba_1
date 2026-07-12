"""
Definición del modelo: transfer learning con MobileNetV2.

Se eligió MobileNetV2 porque es ligero y rápido en CPU, lo que lo hace ideal
para el objetivo final: correr en tiempo real sobre el vídeo de una cámara.

El modelo incluye DENTRO de sí mismo:
  - aumento de datos (solo activo en entrenamiento)
  - el reescalado a [-1, 1] que espera MobileNetV2

Así, tanto el script de predicción como el de la cámara solo tienen que pasar
la imagen redimensionada en rango 0-255, sin recordar pasos de preprocesado.
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


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 3), name="imagen")

    x = _augmentation_block()(inputs)
    # MobileNetV2 espera valores en [-1, 1]
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="reescalado")(x)

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    backbone.trainable = False  # fase 1: congelado

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="salida")(x)

    model = tf.keras.Model(inputs, outputs, name="cacao_morfologia")
    return model


def get_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Devuelve la sub-red MobileNetV2 dentro del modelo (para fine-tuning).

    Se localiza por tipo (submodelo funcional) y por prefijo de nombre, así el
    código sigue funcionando aunque Keras asigne un nombre autogenerado como
    'mobilenetv2_1.00_128'.
    """
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            return layer
    raise ValueError("No se encontró el backbone MobileNetV2 en el modelo.")


def enable_fine_tuning(model: tf.keras.Model, fine_tune_at: int = config.FINE_TUNE_AT):
    """Descongela las capas superiores del backbone para el fine-tuning.

    IMPORTANTE: se mantienen CONGELADAS las capas BatchNormalization. Si se
    descongelan, sus estadísticas móviles (media/varianza) se recalculan con un
    batch pequeño y muy aumentado, lo que desestabiliza el modelo y hace que el
    fine-tuning empeore el resultado en vez de mejorarlo. Congelarlas es la
    práctica recomendada para transfer learning con MobileNetV2.
    """
    backbone = get_backbone(model)
    backbone.trainable = True
    for i, layer in enumerate(backbone.layers):
        if i < fine_tune_at or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    return model
