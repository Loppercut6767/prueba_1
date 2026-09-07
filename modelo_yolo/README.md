# Modelo YOLO — clasificación de morfología de cacao

Enfoque alternativo usando **YOLO en modo clasificación** (Ultralytics), pensado
para **bajo consumo de recursos**. Usa el mismo dataset y el mismo reparto que
`modelo_actual/`, para comparar de forma justa.

## ¿Por qué YOLO *clasificación* y no detección?

El dataset tiene **una etiqueta por imagen** (cada carpeta es una clase) y **no
tiene cajas delimitadoras**. La detección de objetos necesita cajas anotadas, que
aquí no existen. Por eso el encaje correcto es **YOLO-cls** (clasificación de
imagen completa), no YOLO de detección.

## Modelo elegido: `yolo11n-cls` (nano)

El más pequeño de la familia: **~1.5 M parámetros, 3.3 GFLOPs**. Es la opción de
**menor consumo** (mucho más liviano que EfficientNetB2, de ~7.8 M). Ideal si el
objetivo es correr en hardware modesto o en tiempo real.

## Uso

```bash
# entorno virtual propio (recomendado; separado del de modelo_actual)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python prepare_data.py     # arma dataset/ (train/val/test) desde ../data
python train_yolo.py       # entrena yolo11n-cls (CPU); pesos en runs/.../best.pt
python evaluate_yolo.py    # accuracy, F1 y matriz de confusión sobre el test
python predict_yolo.py --image ruta/a/grano.jpg
```

Parámetros (modelo base, épocas, imgsz, batch) en `config.py`.

## Estructura

```
modelo_yolo/
├── config.py          # rutas, clase ES, hiperparámetros
├── prepare_data.py    # ../data -> dataset/ en formato YOLO-cls (mismo split)
├── train_yolo.py      # entrena yolo11n-cls
├── evaluate_yolo.py   # métricas sobre el test (comparables con modelo_actual)
├── predict_yolo.py    # inferencia en imágenes
├── dataset/           # generado (no versionado)
└── runs/              # salidas de entrenamiento (no versionado)
```

## Resultados (primera corrida)

`yolo11n-cls`, 60 épocas, imgsz 224, sobre el test (93 imágenes, mismo split que
`modelo_actual`):

```
              precision    recall  f1-score   support
      entero      1.000     1.000     1.000        16
  fermentado      0.909     0.625     0.741        16
    fraccion      1.000     0.933     0.966        15
      mohoso      0.750     0.938     0.833        16
     partido      0.824     0.933     0.875        15
   pizarroso      0.933     0.933     0.933        15
    accuracy                          0.892        93
   macro avg      0.903     0.894     0.891        93
```

**Accuracy 89.2 % / F1 macro 0.891.** Comparación en el MISMO test:

| Modelo                  | Accuracy | F1 macro | Parámetros | Nota |
|-------------------------|:--------:|:--------:|:----------:|------|
| EfficientNetB2 (actual) |  90.3 %  |  0.902   |   ~7.8 M   | test único |
| **YOLO11n-cls**         | **89.2 %** | **0.891** | **~1.5 M** | test único |

**Conclusión:** YOLO nano prácticamente **iguala a EfficientNetB2 con ~5× menos
parámetros** → mucho más ligero, justo lo buscado para bajo consumo. En la primera
corrida ya maneja bien `partido` (0.875) y `pizarroso` (0.933); el punto flojo
sigue siendo `fermentado` (recall 0.63, se confunde con `mohoso`/`pizarroso`).

El peso entrenado se versiona en `cacao_yolo11n.pt` (~3 MB) para usarlo sin
reentrenar.

> **Ojo (cifra honesta):** 89.2 % es de un único test de 93 imágenes y es
> optimista/ruidoso, igual que el 90.3 % de EfficientNetB2 (cuyo k-fold real fue
> 83.1 %). Para comparar de forma fiable hay que hacer k-fold también de YOLO.

> El reparto de test es el mismo (93 imágenes, mismos conteos por clase) que en
> `modelo_actual`, así que las cifras son directamente comparables. Recuerda que
> un test de 93 imágenes es ruidoso; para una estimación fiable conviene k-fold.
