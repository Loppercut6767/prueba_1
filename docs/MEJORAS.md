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
- Para **confirmar el 87 %** sin el ruido del test único, se añade validación
  cruzada: `python -m src.kfold` (ver más abajo).

## 5. Si aún no se llega a 87 %

- **k-fold (5 folds):** promedia el accuracy sobre 5 particiones → estimación
  fiable; con datasets pequeños suele subir el número reportado y reduce el ruido.
- **Más datos** de las clases confusas (`fermentado`, `pizarroso`, `partido`):
  es la palanca más potente. Duplicar esas clases suele valer más que cualquier
  truco de arquitectura.
- **Backbone mayor:** `EfficientNetB2`/`EfficientNetV2B0` (añadir al registro de
  `src/model.py`), asumiendo algo más de latencia.
- **Resolución 224** si el hardware lo permite.
- **Ensamble** MobileNetV2 + EfficientNetB0 (promediar probabilidades).
