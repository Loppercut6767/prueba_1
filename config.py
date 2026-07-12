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
DATA_DIR = ROOT_DIR / "data" / "raw" / "Cocoa Beans"   # dataset etiquetado
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"

MODEL_PATH = MODELS_DIR / "cacao_mobilenetv2.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

for _d in (MODELS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Parámetros de imagen / entrenamiento
# --------------------------------------------------------------------------- #
IMG_SIZE = 128          # MobileNetV2 acepta 96/128/160/224; 128 es buen balance
BATCH_SIZE = 32
SEED = 42

# Reparto del dataset (estratificado por clase)
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Fase 1: entrenar solo la "cabeza" con el backbone congelado
EPOCHS_HEAD = 25
LR_HEAD = 1e-3

# Fase 2 (opcional): fine-tuning de las últimas capas del backbone
FINE_TUNE = True
EPOCHS_FINE_TUNE = 15
LR_FINE_TUNE = 1e-5
FINE_TUNE_AT = 100      # descongela desde esta capa de MobileNetV2 en adelante

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
