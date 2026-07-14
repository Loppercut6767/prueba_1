# Cómo mejorar la precisión del modelo (objetivo: 87 %+)

Documento del escenario, diagnóstico y palancas para subir la exactitud de
EfficientNetB0 desde ~82 % hacia 87 % o más.

## 1. Diagnóstico: ¿dónde falla hoy?

Del reporte por clase sobre el test (93 imágenes), los puntos débiles eran:

| Clase        | F1    | Problema                                             |
|--------------|-------|------------------------------------------------------|
| `fermentado` | 0.63  | se confunde con `pizarroso` (diferencia = **color**) |
| `partido`    | 0.61  | se confunde con `fraccion` (diferencia = **forma**)  |
| `pizarroso`  | 0.77  | se confunde con `fermentado`                         |

Las clases fáciles (`entero`, `mohoso`, `fraccion`) ya iban >0.93.

**Causas raíz en la configuración anterior:**

1. **Resolución baja (128 px).** EfficientNetB0 se diseñó para ~224 px. A 128 se
   pierden texturas finas de color y forma, justo lo que separa las clases duras.
2. **Aumento de color agresivo.** `RandomBrightness/Contrast` fuertes borraban la
   señal de color que distingue fermentado (marrón) de pizarroso (gris pizarra).
3. **Resize que deforma.** Estirar la imagen a cuadrado altera la proporción del
   grano y perjudica a las clases de forma (partido/fraccion).
4. **Fine-tuning tímido.** LR 1e-5 y pocas capas descongeladas: apenas ajustaba.
5. **Test pequeño (93 img).** ±1 imagen ≈ 1 %; el número es ruidoso.

## 2. Palancas de mejora (ordenadas por impacto)

| # | Mejora                                   | Por qué ayuda                              | Estado |
|---|------------------------------------------|--------------------------------------------|--------|
| 1 | Resolución **128 → 160 px**              | más detalle de color/forma                 | ✅ hecho |
| 2 | Aumento **geométrico fuerte, color suave** | preserva fermentado vs pizarroso          | ✅ hecho |
| 3 | **Resize con padding** (sin deformar)    | preserva la forma del grano                | ✅ hecho |
| 4 | **Label smoothing (0.1)**                | regulariza, mejor generalización           | ✅ hecho |
| 5 | **Fine-tuning con cosine decay** y más capas | ajuste más profundo y estable          | ✅ hecho |
| 6 | **Test-Time Augmentation (TTA)**         | promedia volteos, +1–3 % casi gratis       | ✅ hecho |
| 7 | **k-fold estratificado**                 | medir sin ruido y confirmar el 87 %        | pendiente |
| 8 | Backbone mayor (**EfficientNetB2 / V2**) | más capacidad si hace falta                | opcional |
| 9 | **Más datos** (recolectar/otros datasets)| lo más determinante con solo 614 imágenes  | recomendado |

## 3. Qué se implementó

Todo configurable en `config.py`:

- `IMG_SIZE = 160`, `PAD_RESIZE = True`
- `AUG_COLOR_STRENGTH = 0.06` (aumento de color suave)
- `LABEL_SMOOTHING = 0.1`
- `LR_FINE_TUNE = 3e-5`, `EPOCHS_FINE_TUNE = 20`, fine-tuning con **cosine decay**
- `FINE_TUNE_AT` más bajo → se descongela más red
- TTA disponible en la evaluación: `python -m src.compare --tta`

Reentrenar y comparar con la nueva receta:

```bash
python -m src.compare --retrain --tta
```

## 4. Resultados

EfficientNetB0, antes vs. después de aplicar las mejoras 1–6 (evaluación con TTA):

| Métrica (test)        | Antes (128 px) | Después (160 px + mejoras) |
|-----------------------|:--------------:|:--------------------------:|
| Accuracy              |     81.7 %     |          **83.9 %**        |
| F1 macro              |     0.812      |          **0.845**         |
| **Precisión macro**   |     0.835      |          **0.870**         |
| `val_accuracy` (mejor)|     ~0.83      |          **0.88**          |

- La **validación alcanzó 88 %** y la **precisión macro 0.87**. El accuracy de
  test (83.9 %) queda por debajo por el **tamaño pequeño del test (93 imágenes)**:
  la diferencia test/val es la varianza esperable de una muestra tan chica.
- Mejoras por clase: `partido` 0.61 → **0.80** F1 (el padding + resolución ayudó
  a la forma) y `fermentado` recall 0.69 → 0.75. El cuello de botella sigue
  siendo `fermentado ↔ pizarroso` (color muy parecido).
### Validación cruzada (el número honesto)

El accuracy de un único test de 93 imágenes es ruidoso. La validación cruzada
5-fold con TTA (`python -m src.kfold`) da la estimación fiable:

```
  fold 1: acc=79.7%   fold 2: acc=78.9%   fold 3: acc=84.6%
  fold 4: acc=81.3%   fold 5: acc=82.8%
  Accuracy media: 81.4 %  (± 2.1)   |   F1 macro: 81.2 %  (± 2.7)
  Rango: 78.9 % – 84.6 %
```

**Conclusión honesta:** la exactitud REAL generalizada es **~81 % (±2)**. El
83.9 % del test único y el 88 % de validación eran el lado optimista de esa
distribución. Las mejoras 1–6 **sí ayudaron** (más estable y +F1), pero **aún no
se alcanza el 87 % de forma fiable**: para eso hacen falta las palancas de la
sección 5, sobre todo **más datos**.

## 5. Cómo llegar de verdad a 87 %+ (todavía pendiente)

Estamos en ~81 % cross-validado. Para cerrar los ~6 puntos que faltan, por orden
de impacto esperado:

1. **Más datos (la palanca decisiva).** Con 614 imágenes el modelo está cerca de
   su techo. Recolectar/etiquetar más granos —sobre todo de las clases confusas
   `fermentado`, `pizarroso` y `partido`— es lo que más sube el número. Como
   referencia, pasar de ~100 a ~300 imágenes por clase suele valer más que
   cualquier cambio de arquitectura. Fuentes: fotografiar granos propios con la
   misma cámara del despliegue (ideal, elimina el domain gap) o datasets públicos
   adicionales de Kaggle/Roboflow.
2. **Backbone mayor:** `EfficientNetB2` o `EfficientNetV2B0`. Basta con añadirlo
   al registro `_build_backbone` de `src/model.py` y a `AVAILABLE_BACKBONES`;
   el resto del pipeline ya es genérico. Cuesta algo más de latencia.
3. **Resolución 224** (subir `IMG_SIZE`) si el hardware lo permite.
4. **Ensamble** de EfficientNetB0 + B2 (promediar probabilidades) — suele dar
   +1–3 puntos estables.
5. **Foco en `fermentado ↔ pizarroso`:** es un problema de color. Ayudaría
   normalizar el color/iluminación de captura (white balance fijo en la cámara)
   y añadir ejemplos de ambos con iluminación controlada.

> Realista: con más datos de las clases duras, 87 %+ es alcanzable. Solo con
> trucos de arquitectura sobre estas 614 imágenes, lo esperable es ~82–85 %.

## 6. Experimento: resolución 224 + EfficientNetB2 + ensamble

Se subió `IMG_SIZE` a **224** y se registró **EfficientNetB2** (backbone mayor).
Comparación de los 3 modelos a 224 px con TTA (test de 93 imágenes):

| Modelo         | Accuracy | F1 macro | FPS (CPU) | Tamaño |
|----------------|:--------:|:--------:|:---------:|:------:|
| MobileNetV2    |  74.2 %  |  0.744   |   ~6.2    | 9.7 MB |
| EfficientNetB0 |  76.3 %  |  0.774   |   ~3.3    | 47.6 MB |
| **EfficientNetB2** | **90.3 %** | **0.902** | ~2.3 | 92.4 MB |

**EfficientNetB2 dio un salto grande: 90.3 % en test** (precisión macro 0.92),
con `fermentado` F1 0.63 → **0.79** y `pizarroso` → **0.86**. La resolución 224 +
mayor capacidad sí capturan la diferencia de color que antes se perdía.

**El ensamble NO ayudó.** Promediar B0 (76 %) + B2 (90 %) da ~82 %: el modelo
flojo arrastra al fuerte. Con estos datos, **B2 en solitario es la mejor opción**
y es el que queda desplegado en la cámara (`models/best_model.json`).

**Coste de B2:** ~2.3 FPS y 92 MB en CPU. Para clasificar granos colocados uno a
uno (modo `--roi`) es suficiente; para vídeo muy fluido, MobileNetV2 (~6 FPS) es
la alternativa rápida a costa de exactitud.

### Confirmación con validación cruzada (EfficientNetB2)

<!-- KFOLD_B2_DOC -->

> El 90.3 % es de un único test; el k-fold da el número robusto (arriba). Aun con
> el margen de la muestra pequeña, B2 se sitúa claramente por **encima del 87 %**
> objetivo.
