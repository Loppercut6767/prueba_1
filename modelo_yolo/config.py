"""
Configuración del proyecto YOLO (clasificación de morfología de cacao).

Usamos YOLO en modo CLASIFICACIÓN (no detección), porque el dataset tiene una
etiqueta por imagen (carpeta = clase) y no cajas delimitadoras. El modelo elegido
es el más ligero de la familia: 'yolo11n-cls' (n = nano, ~1.5 M parámetros), pensado
justo para bajo consumo de recursos.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT_DIR.parent
# Mismo dataset que modelo_actual (compartido en la raíz del repo)
RAW_DATA_DIR = REPO_ROOT / "data" / "raw" / "Cocoa Beans"
# Dataset en formato YOLO-clasificación (train/val/test por clase). Se genera.
DATASET_DIR = ROOT_DIR / "dataset"
RUNS_DIR = ROOT_DIR / "runs"

# Modelo base ligero y parámetros de entrenamiento
MODEL = "yolo11n-cls.pt"     # nano: mínimo consumo
IMG_SIZE = 224
EPOCHS = 60
BATCH = 32
SEED = 42

# Mismo reparto que modelo_actual para comparar de forma justa
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Mapeo carpeta_del_dataset -> etiqueta de morfología (español), igual que en
# modelo_actual, para que las clases coincidan entre ambos proyectos.
CLASS_MAP_ES = {
    "Whole_Beans_Cocoa":   "entero",
    "Broken_Beans_Cocoa":  "partido",
    "Bean_Fraction_Cocoa": "fraccion",
    "Moldy_Cocoa":         "mohoso",
    "Fermented_Cocoa":     "fermentado",
    "Unfermented_Cocoa":   "pizarroso",
}
