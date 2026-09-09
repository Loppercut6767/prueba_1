"""
Configuración central del clasificador de morfología de granos de cacao.

Aquí se definen las rutas, los hiperparámetros y —muy importante— el mapeo
entre las carpetas del dataset público y los nombres de morfología en español
que se muestran al usuario y que usará la cámara en tiempo real.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent
# El dataset vive en la raíz del repo (data/), compartido con modelo_yolo/
REPO_ROOT = ROOT_DIR.parent
DATA_DIR = REPO_ROOT / "data" / "raw" / "Cocoa Beans"   # dataset etiquetado
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
# Puntero al modelo elegido para producción/cámara (lo escribe src/compare.py)
BEST_MODEL_PATH = MODELS_DIR / "best_model.json"

for _d in (MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Modelos disponibles para comparar
#
# El proyecto entrena y compara dos arquitecturas de transfer learning. La
# cámara usa la que gane la comparación (ver src/compare.py).
# --------------------------------------------------------------------------- #
# Modelos candidatos para la CÁMARA (los que compara src/compare.py)
AVAILABLE_BACKBONES = ["mobilenetv2", "efficientnetb0", "efficientnetb2"]
# Todos los backbones ENTRENABLES (incluye el mayor usado en el ensamble)
ALL_BACKBONES = ["mobilenetv2", "efficientnetb0", "efficientnetb2"]
# Miembros del ensamble (se promedian sus probabilidades en src/ensemble.py)
ENSEMBLE_MEMBERS = ["efficientnetb0", "efficientnetb2"]
DEFAULT_BACKBONE = "mobilenetv2"


def model_path(backbone: str):
    """Ruta del modelo entrenado para un backbone dado."""
    return MODELS_DIR / f"cacao_{backbone}.keras"


# Compatibilidad: MODEL_PATH sigue apuntando al backbone por defecto
MODEL_PATH = model_path(DEFAULT_BACKBONE)


def resolve_model_path(backbone: str | None = None):
    """Ruta del modelo a usar en inferencia (predicción / cámara).

    Prioridad:
      1) el backbone pedido explícitamente,
      2) el ganador de la comparación (models/best_model.json),
      3) el backbone por defecto.
    """
    import json

    if backbone:
        return model_path(backbone)
    if BEST_MODEL_PATH.exists():
        try:
            info = json.loads(BEST_MODEL_PATH.read_text())
            # Se reconstruye desde el 'backbone' (portable entre máquinas), no
            # desde una ruta absoluta guardada.
            p = model_path(info["backbone"])
            if p.exists():
                return p
        except (ValueError, KeyError):
            pass
    return MODEL_PATH

# --------------------------------------------------------------------------- #
# Parámetros de imagen / entrenamiento
# --------------------------------------------------------------------------- #
IMG_SIZE = 160          # 160: equilibrio precisión/velocidad, apto para Raspberry Pi 5
BATCH_SIZE = 32
SEED = 42

# Redimensionado preservando la proporción (padding) en vez de estirar a cuadrado.
# Evita deformar el grano, lo que ayuda a las clases basadas en forma
# (partido / fraccion / entero).
PAD_RESIZE = True

# Aumento de datos: se separa el geométrico (fuerte) del de color (SUAVE), porque
# el color es justo lo que distingue 'fermentado' de 'pizarroso'. Aumentar mucho
# el brillo/contraste borra esa señal y confunde ambas clases.
AUG_COLOR_STRENGTH = 0.06   # antes ~0.15; más bajo = conserva mejor el color

# Suavizado de etiquetas: regulariza y mejora la calibración/generalización
LABEL_SMOOTHING = 0.1

# Reparto del dataset (estratificado por clase)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Fase 1: entrenar solo la "cabeza" con el backbone congelado
EPOCHS_HEAD = 25
LR_HEAD = 1e-3

# Fase 2 (opcional): fine-tuning de las últimas capas del backbone
FINE_TUNE = True
EPOCHS_FINE_TUNE = 20    # más épocas: el fine-tuning con cosine decay necesita margen
LR_FINE_TUNE = 3e-5      # algo mayor que 1e-5 para que el fine-tuning mueva la aguja
# Capa a partir de la cual se descongela cada backbone durante el fine-tuning.
# Se descongela MÁS red que antes (número menor) para adaptar mejor al dominio cacao.
FINE_TUNE_AT = {
    "mobilenetv2": 80,
    "efficientnetb0": 100,
    "efficientnetb2": 120,
}

# --------------------------------------------------------------------------- #
# Mapeo carpeta_del_dataset -> etiqueta de morfología (español)
#
# El dataset público "Cocoa Beans" trae 6 categorías. Aquí las traducimos a la
# terminología de morfología/calidad usada en la práctica cacaotera. Si más
# adelante consigues imágenes de otras morfologías (p. ej. "germinado"), basta
# con crear una carpeta nueva dentro de data/raw/Cocoa Beans/ y añadirla aquí.
# --------------------------------------------------------------------------- #
CLASS_MAP_ES = {
    "Whole_Beans_Cocoa":   "entero",        # grano sano y completo
    "Broken_Beans_Cocoa":  "partido",       # grano roto / dañado mecánicamente
    "Bean_Fraction_Cocoa": "fraccion",      # fragmento de grano
    "Moldy_Cocoa":         "mohoso",        # dañado por hongos
    "Fermented_Cocoa":     "fermentado",    # bien fermentado (marrón)
    "Unfermented_Cocoa":   "pizarroso",     # sin fermentar (violáceo/gris pizarra)
}

# Descripción legible de cada morfología (para reportes y overlay de cámara)
CLASS_DESCRIPTIONS_ES = {
    "entero":     "Grano entero y sano",
    "partido":    "Grano partido o dañado",
    "fraccion":   "Fracción/fragmento de grano",
    "mohoso":     "Grano con moho (hongos)",
    "fermentado": "Grano correctamente fermentado",
    "pizarroso":  "Grano pizarroso (sin fermentar)",
}
