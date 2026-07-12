# Clasificador de morfología de granos de cacao

Modelo de visión por computador que determina la **morfología / calidad de un
grano de cacao** a partir de una imagen. Está pensado para entrenarse con un
dataset público etiquetado y luego funcionar **en tiempo real sobre el vídeo de
una cámara**, clasificando cada grano (entero, partido, mohoso, pizarroso, etc.).

## Índice
- [¿Qué morfologías detecta?](#qué-morfologías-detecta)
- [Dataset](#dataset)
- [Instalación](#instalación)
- [Uso rápido](#uso-rápido)
- [Cámara en tiempo real](#cámara-en-tiempo-real)
- [Cómo funciona](#cómo-funciona)
- [Ampliar a nuevas morfologías (p. ej. germinado)](#ampliar-a-nuevas-morfologías)
- [Estructura del proyecto](#estructura-del-proyecto)

## ¿Qué morfologías detecta?

El dataset público usado trae 6 categorías, que se mapean a la terminología
cacaotera en español (ver `config.py → CLASS_MAP_ES`):

| Carpeta del dataset   | Etiqueta (ES) | Descripción                          |
|-----------------------|---------------|--------------------------------------|
| `Whole_Beans_Cocoa`   | `entero`      | Grano entero y sano                  |
| `Broken_Beans_Cocoa`  | `partido`     | Grano partido o dañado               |
| `Bean_Fraction_Cocoa` | `fraccion`    | Fracción/fragmento de grano          |
| `Moldy_Cocoa`         | `mohoso`      | Grano con moho (hongos)              |
| `Fermented_Cocoa`     | `fermentado`  | Grano correctamente fermentado       |
| `Unfermented_Cocoa`   | `pizarroso`   | Grano sin fermentar (gris pizarra)   |

> **Nota:** el dataset público no incluye una clase "germinado". El código está
> preparado para añadir nuevas morfologías fácilmente (ver
> [sección de ampliación](#ampliar-a-nuevas-morfologías)).

## Dataset

Se usa el dataset público **Cocoa Beans** (614 imágenes, 6 clases balanceadas,
formato JPG). Ya viene incluido en `data/raw/Cocoa Beans/`.

- **Entrenamiento / validación / test**: el reparto es **aleatorio y
  estratificado** (70 % / 15 % / 15 %) con semilla fija para reproducibilidad.
  El conjunto de **test son imágenes aleatorias que el modelo nunca ve durante
  el entrenamiento**, tal como se pide para validar el funcionamiento.

Datasets públicos equivalentes/adicionales que puedes usar para ampliar:
- *Cocoa Beans* (Kaggle) — el aquí incluido.
- Buscar "cocoa bean quality dataset" / "cacao beans classification" en Kaggle
  o Roboflow Universe para clases adicionales (germinado, plano, múltiple…).

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate    # opcional
pip install -r requirements.txt
```

> Si tienes GPU, cambia `tensorflow-cpu` por `tensorflow` en `requirements.txt`.

## Uso rápido

```bash
# 1) Entrenar (cabeza + fine-tuning). Guarda models/cacao_mobilenetv2.keras
python -m src.train

# variantes
python -m src.train --no-fine-tune      # solo la cabeza (más rápido)
python -m src.train --epochs 30         # más épocas de la fase 1

# 2) Evaluar sobre el test (reporte + matriz de confusión en outputs/)
python -m src.evaluate

# 3) Predecir sobre imágenes
python -m src.predict --image ruta/a/grano.jpg      # una imagen
python -m src.predict --dir  ruta/a/carpeta          # una carpeta
python -m src.predict --random 12                    # 12 imágenes aleatorias
```

`--random` además genera `outputs/predicciones_aleatorias.png`, una cuadrícula
con la predicción y la etiqueta real de cada grano: la forma más rápida de
comprobar visualmente que el modelo funciona antes de conectar la cámara.

## Cámara en tiempo real

Objetivo final del proyecto. Toma el vídeo de una webcam y superpone la
morfología detectada en cada fotograma:

```bash
python -m src.camera                 # webcam por defecto (cámara 0)
python -m src.camera --source 1      # otra cámara
python -m src.camera --source video.mp4   # un archivo de vídeo
python -m src.camera --roi           # analiza solo un recuadro central
python -m src.camera --conf 0.6      # solo muestra etiqueta si confianza >= 60 %
```

Controles: pulsa **`q`** para salir. Requiere un entorno con cámara/pantalla
(no funciona en un servidor headless). Para colocar un grano en un punto fijo,
usa `--roi` y sitúa el grano dentro del recuadro verde.

## Cómo funciona

- **Arquitectura**: *transfer learning* con **MobileNetV2** (pesos de ImageNet).
  Se eligió por ser ligera y rápida en CPU → ideal para vídeo en tiempo real.
- El modelo lleva **integrados** el aumento de datos (solo en entrenamiento) y
  el reescalado a `[-1, 1]`. Por eso la inferencia (predict/cámara) solo tiene
  que pasar la imagen redimensionada en rango 0-255, sin recordar preprocesados.
- **Entrenamiento en 2 fases**:
  1. *Cabeza*: backbone congelado, se entrena solo el clasificador final.
  2. *Fine-tuning*: se descongelan las capas superiores del backbone con un
     *learning rate* muy bajo para afinar.
- **Regularización**: `Dropout`, aumento de datos, `EarlyStopping`,
  `ReduceLROnPlateau` y pesos por clase.

Los hiperparámetros están todos en `config.py`.

## Ampliar a nuevas morfologías

Para añadir una clase nueva (por ejemplo, **germinado**):

1. Crea la carpeta con las imágenes:
   `data/raw/Cocoa Beans/Germinated_Cocoa/*.jpg`
2. Añade el mapeo en `config.py`:
   ```python
   CLASS_MAP_ES = {
       ...,
       "Germinated_Cocoa": "germinado",
   }
   CLASS_DESCRIPTIONS_ES = {
       ...,
       "germinado": "Grano germinado",
   }
   ```
3. Reentrena: `python -m src.train`

El número de clases se detecta automáticamente; no hay que tocar el resto.

## Estructura del proyecto

```
.
├── config.py               # rutas, hiperparámetros y mapeo de clases (ES)
├── requirements.txt
├── data/raw/Cocoa Beans/   # dataset etiquetado (6 clases)
├── models/                 # modelo entrenado + class_names.json
├── outputs/                # gráficas, matriz de confusión, predicciones
└── src/
    ├── data.py             # carga y reparto estratificado del dataset
    ├── model.py            # MobileNetV2 (transfer learning)
    ├── train.py            # entrenamiento en 2 fases
    ├── evaluate.py         # reporte + matriz de confusión sobre el test
    ├── predict.py          # inferencia en imágenes / lote / aleatorias
    └── camera.py           # clasificación en tiempo real con cámara
```
