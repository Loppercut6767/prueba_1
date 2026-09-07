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

El proyecto entrena y compara **tres arquitecturas** de transfer learning sobre
exactamente el mismo reparto de datos, para **justificar** qué modelo se despliega
en la cámara:

| Modelo           | Idea                                                        |
|------------------|-------------------------------------------------------------|
| `MobileNetV2`    | Muy ligero y rápido → máxima velocidad en CPU               |
| `EfficientNetB0` | Intermedio                                                  |
| `EfficientNetB2` | Más pesado, el más exacto → **el desplegado en la cámara**  |

```bash
# entrena (si falta) los modelos y los compara sobre el test
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

Tres modelos entrenados a **224 px** con el mismo reparto (429 train / 92 val /
93 test), evaluados con TTA:

| Modelo             | Accuracy | F1 macro | Latencia (CPU) |   FPS  | Tamaño |
|--------------------|:--------:|:--------:|:--------------:|:------:|:------:|
| MobileNetV2        |  74.2 %  |  0.744   |    ~161 ms     |  ~6.2  | 9.7 MB |
| EfficientNetB0     |  76.3 %  |  0.774   |    ~305 ms     |  ~3.3  | 47.6 MB |
| **EfficientNetB2** | **90.3 %** | **0.902** |  ~429 ms     |  ~2.3  | 92.4 MB |

![Comparación de modelos](outputs/comparacion_modelos.png)

**Modelo elegido para la cámara: `EfficientNetB2`**, por ser con diferencia el de
**mayor exactitud** (90.3 % en test, +14 pp sobre los demás). **Coste:** es el más
pesado y lento (~2.3 FPS en CPU, 92 MB), pero para clasificar granos colocados
uno a uno frente a la cámara (modo `--roi`) esa velocidad es suficiente.

> **Cifra honesta (validación cruzada 5-fold):** el 90.3 % de la tabla es de un
> único test de 93 imágenes y resultó **optimista**. El k-fold de EfficientNetB2
> (`python -m src.kfold --backbone efficientnetb2`) da la estimación fiable:
> **83.1 % ± 2.3 %** (rango 78.9–85.4 %). Es la mejor de las probadas (B0 quedó en
> 81.4 %), pero **aún no llega al 87 % robusto**: para eso falta, sobre todo, más
> datos. Detalle y plan en `docs/MEJORAS.md`.

> **Sobre el ensamble:** se probó un ensamble EfficientNetB0 + B2
> (`python -m src.ensemble`), pero **no mejoró**: al promediar un modelo fuerte
> (B2, 90 %) con uno flojo (B0, 76 %) el resultado baja (~82 %). Con estos datos,
> **B2 en solitario es la mejor opción**.

> Si la fluidez del vídeo fuese crítica, MobileNetV2 (~6 FPS) es la alternativa
> rápida: `python -m src.camera --backbone mobilenetv2`. Las latencias son de esta
> CPU; en otro equipo cambian, pero la relación entre modelos se mantiene.

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

Reporte del modelo desplegado (**EfficientNetB2**, 224 px + TTA) sobre el
conjunto de **test** (imágenes aleatorias no vistas; 429 train / 92 val / 93 test):

```
              precision    recall  f1-score   support
      entero      1.000     0.938     0.968        16
  fermentado      0.917     0.688     0.786        16
    fraccion      1.000     1.000     1.000        15
      mohoso      0.842     1.000     0.914        16
     partido      1.000     0.800     0.889        15
   pizarroso      0.750     1.000     0.857        15
    accuracy                          0.903        93
   macro avg      0.918     0.904     0.902        93
```

En este test único da **~90 % de accuracy**, pero es optimista: la cifra fiable
(validación cruzada 5-fold) es **83.1 % ± 2.3 %**. Aun así, la clase difícil
`fermentado` subió de F1 0.63 a **0.79** y `pizarroso` a **0.86** (la resolución
224 + backbone mayor sí capta mejor la diferencia de color). El repo incluye **los
tres modelos** ya entrenados en `models/` para comparar y probar la cámara sin
reentrenar. Ver `docs/MEJORAS.md` para el plan de mejora y cómo llegar a 87 %+.

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
├── docs/MEJORAS.md         # plan para subir la precisión (objetivo 87%+)
├── data/raw/Cocoa Beans/   # dataset etiquetado (6 clases)
├── models/                 # modelos entrenados + class_names.json + best_model.json
├── outputs/                # gráficas, matriz de confusión, comparación
└── src/
    ├── data.py             # carga y reparto estratificado del dataset
    ├── model.py            # backbones intercambiables (MobileNetV2 / EfficientNetB0)
    ├── train.py            # entrenamiento en 2 fases (--backbone)
    ├── compare.py          # compara los 2 modelos y elige el de la cámara
    ├── kfold.py            # validación cruzada (estimación fiable de accuracy)
    ├── evaluate.py         # reporte + matriz de confusión sobre el test
    ├── predict.py          # inferencia en imágenes / lote / aleatorias
    └── camera.py           # clasificación en tiempo real con cámara
```
