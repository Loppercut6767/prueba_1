# Clasificación de morfología de granos de cacao

Proyecto para determinar la **morfología / calidad de un grano de cacao** a partir
de una imagen (clases: `entero`, `partido`, `fraccion`, `mohoso`, `fermentado`,
`pizarroso`), con vistas a clasificar en tiempo real desde una cámara.

> **Recomendación para Raspberry Pi 5 (8 GB): `YOLO11n-cls` (semilla 3).** Es el
> más preciso (media 87.3 %, hasta 93.5 %), ~30–70× más rápido y ~8–30× más
> liviano (3.2 MB) que los modelos Keras. Detalle y tabla completa en
> [`COMPARACION_PI5.md`](COMPARACION_PI5.md).

Se comparan **dos enfoques**, cada uno en su carpeta, usando el **mismo dataset**:

| Carpeta         | Enfoque                                   | Modelo(s) |
|-----------------|-------------------------------------------|-----------|
| `modelo_actual/`| Transfer learning con Keras/TensorFlow    | MobileNetV2, EfficientNetB0, **EfficientNetB2** (desplegado) |
| `modelo_yolo/`  | Clasificación con YOLO (Ultralytics)      | **yolo11n-cls** (nano, bajo consumo) |

```
.
├── data/raw/Cocoa Beans/   # dataset etiquetado (614 img, 6 clases) — COMPARTIDO
├── modelo_actual/          # proyecto Keras/TensorFlow (ver su README)
└── modelo_yolo/            # proyecto YOLO (ver su README)
```

- El **dataset vive en `data/`** en la raíz y lo usan ambos proyectos, con el
  mismo reparto estratificado 70/15/15 (semilla 42) para que la comparación sea
  justa.
- **Entornos separados:** `modelo_actual` usa TensorFlow y `modelo_yolo` usa
  PyTorch/Ultralytics. Conviene un entorno virtual por carpeta (ambos frameworks
  pueden chocar en la versión de `numpy`). Cada carpeta trae su `requirements.txt`.

## Puesta en marcha rápida

```bash
# --- Modelo actual (Keras) ---
cd modelo_actual
pip install -r requirements.txt
python -m src.compare            # entrena/compara y elige el mejor
python -m src.camera             # cámara en tiempo real

# --- Modelo YOLO ---
cd ../modelo_yolo
pip install -r requirements.txt
python prepare_data.py           # arma el dataset en formato YOLO
python train_yolo.py             # entrena yolo11n-cls
python evaluate_yolo.py          # métricas sobre el test
```

Consulta el `README.md` de cada carpeta para el detalle de cada enfoque, sus
resultados y opciones.
