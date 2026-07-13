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
# 1) Entrenar un modelo (cabeza + fine-tuning)
python -m src.train                              # MobileNetV2 (por defecto)
python -m src.train --backbone efficientnetb0    # el otro modelo

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

## Comparación de modelos (¿cuál va a la cámara?)

El proyecto entrena y compara **dos arquitecturas** de transfer learning sobre
exactamente el mismo reparto de datos, para **justificar** qué modelo se despliega
en la cámara:

| Modelo           | Idea                                                        |
|------------------|-------------------------------------------------------------|
| `MobileNetV2`    | Muy ligero y rápido → pensado para tiempo real en CPU       |
| `EfficientNetB0` | Algo más pesado, normalmente más exacto                     |

```bash
# entrena (si falta) ambos modelos y los compara sobre el test
python -m src.compare

# reentrena ambos desde cero para una comparación 100 % reproducible
python -m src.compare --retrain

# priorizar velocidad en el desempate (útil si la cámara es el cuello de botella)
python -m src.compare --criterio latencia
```

`compare.py` mide, sobre el conjunto de **test**: exactitud, F1 macro, **latencia
(ms/imagen) y FPS en CPU**, tamaño en disco y nº de parámetros. Después:

- elige el ganador (por defecto, el de mayor exactitud) y escribe
  `models/best_model.json` — el **puntero que usan la cámara y la predicción**;
- guarda la gráfica comparativa en `outputs/comparacion_modelos.png`.

La comparación de exactitud **frente a** latencia es justo la justificación para
elegir el modelo de la cámara: no basta con acertar, tiene que ir fluido en vídeo.

### Resultado de la comparación

Reentrenando ambos modelos con el mismo reparto (429 train / 92 val / 93 test):

| Modelo             | Accuracy | F1 macro | Latencia (CPU) |   FPS | Tamaño |
|--------------------|:--------:|:--------:|:--------------:|:-----:|:------:|
| MobileNetV2        |  75.3 %  |  0.755   |    ~133 ms     |  ~7.5 | 9.7 MB |
| **EfficientNetB0** | **81.7 %** | **0.812** |   ~257 ms     |  ~3.9 | 43.3 MB |

![Comparación de modelos](outputs/comparacion_modelos.png)

**Modelo elegido para la cámara: `EfficientNetB0`**, por ser el de **mayor
exactitud** (+6.4 puntos de accuracy y +5.7 de F1 macro). Mejora sobre todo en
`mohoso` (0.75 → 0.94 F1) y `entero` (perfecto). El coste es que es ~2× más
lento y más pesado, pero a ~4 FPS sigue siendo apto para clasificar granos
colocados frente a la cámara.

> Si en tu hardware la fluidez del vídeo fuese crítica, MobileNetV2 (≈7.5 FPS)
> es la alternativa: ejecuta `python -m src.compare --criterio latencia` o
> `python -m src.camera --backbone mobilenetv2`. Las latencias son de esta CPU;
> en otro equipo cambian, pero la relación entre ambos modelos se mantiene.

## Cámara en tiempo real

Objetivo final del proyecto. Toma el vídeo de una webcam y superpone la
morfología detectada en cada fotograma. **Usa automáticamente el modelo ganador
de la comparación** (`models/best_model.json`); si no se ha ejecutado la
comparación, usa MobileNetV2 por defecto.

```bash
python -m src.camera                 # webcam, modelo ganador de la comparación
python -m src.camera --source 1      # otra cámara
python -m src.camera --source video.mp4   # un archivo de vídeo
python -m src.camera --roi           # analiza solo un recuadro central
python -m src.camera --conf 0.6      # solo muestra etiqueta si confianza >= 60 %
python -m src.camera --backbone mobilenetv2   # forzar un modelo concreto
```

Controles: pulsa **`q`** para salir. Requiere un entorno con cámara/pantalla
(no funciona en un servidor headless). Para colocar un grano en un punto fijo,
usa `--roi` y sitúa el grano dentro del recuadro verde.

## Resultados de referencia

Reporte del modelo desplegado (**EfficientNetB0**) sobre el conjunto de **test**
(imágenes aleatorias no vistas; 429 train / 92 val / 93 test):

```
              precision    recall  f1-score   support
      entero      1.000     1.000     1.000        16
  fermentado      0.579     0.688     0.629        16
    fraccion      0.933     0.933     0.933        15
      mohoso      0.938     0.938     0.938        16
     partido      0.875     0.467     0.609        15
   pizarroso      0.684     0.867     0.765        15
    accuracy                          0.817        93
```

**~82 % de accuracy** en 6 clases con solo 614 imágenes de 80–190 px. La mayor
confusión sigue siendo `fermentado ↔ pizarroso` (ambos granos enteros que se
distinguen por el color de fermentación) y algo de `partido → fraccion` — algo
esperable y mejorable con más datos y mayor resolución. El repo incluye **ambos
modelos** ya entrenados en `models/` para poder comparar y probar la cámara sin
reentrenar.

> Los números pueden variar ligeramente entre ejecuciones. Con `--random`, el
> script de predicción muestrea imágenes de **todo** el dataset (train incluido),
> por lo que su porcentaje no equivale al accuracy de test.

## Cómo funciona

- **Arquitectura**: *transfer learning* con dos backbones intercambiables,
  **MobileNetV2** y **EfficientNetB0** (pesos de ImageNet), que comparten la
  misma cabeza y bloque de aumento para que la comparación sea justa.
- El modelo lleva **integrados** el aumento de datos (solo en entrenamiento) y
  el preprocesado propio de cada arquitectura (MobileNetV2 reescala a `[-1, 1]`;
  EfficientNetB0 normaliza internamente). Por eso la inferencia (predict/cámara)
  solo pasa la imagen redimensionada en rango 0-255, sin recordar preprocesados.
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
├── models/                 # modelos entrenados + class_names.json + best_model.json
├── outputs/                # gráficas, matriz de confusión, comparación
└── src/
    ├── data.py             # carga y reparto estratificado del dataset
    ├── model.py            # backbones intercambiables (MobileNetV2 / EfficientNetB0)
    ├── train.py            # entrenamiento en 2 fases (--backbone)
    ├── compare.py          # compara los 2 modelos y elige el de la cámara
    ├── evaluate.py         # reporte + matriz de confusión sobre el test
    ├── predict.py          # inferencia en imágenes / lote / aleatorias
    └── camera.py           # clasificación en tiempo real con cámara
```
