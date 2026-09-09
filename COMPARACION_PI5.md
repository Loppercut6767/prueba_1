# Comparación de modelos y recomendación para Raspberry Pi 5 (8 GB)

Comparación de los **4 modelos** entrenados sobre el mismo dataset (614 imágenes,
6 clases de morfología), mismo reparto estratificado 70/15/15, a **160 px**
(resolución elegida por equilibrio precisión/velocidad, apta para el Pi 5).

> Sobre el dataset **CoCoaSpec** propuesto: es hiperespectral + fisicoquímico
> (fermentación/humedad continua), 23 GB, y **no encaja** con la clasificación de
> morfología en 6 clases. No se usó; se mantuvo el dataset actual.

## Metodología (honesta)

- **Barrido de semillas (5 semillas: 0–4)** por modelo, con el **reparto fijo**
  (mismo test para todos) variando solo la semilla de entrenamiento.
- Se reporta la **media ± desviación** (lo robusto). Elegir la semilla con mejor
  *test* sería sobreajustar al test; por eso la **semilla recomendada** es la de
  **mejor validación**.
- Latencia/tamaño medidos en la CPU de desarrollo (x86); en el Pi 5 (ARM
  Cortex-A76) serán más lentos, pero **la relación entre modelos se mantiene**.

## Resultados (test, imgsz 160)

| Modelo           | Accuracy media ± std | Rango (min–max) | Seed recom. (val) → test | Params | Peso | Latencia | FPS  |
|------------------|:--------------------:|:---------------:|:------------------------:|:------:|:----:|:--------:|:----:|
| MobileNetV2      |    78.5 % ± 2.3      |  75.3 – 80.6 %  |        0 → 80.6 %        | 2.3 M  | 26 MB| ~129 ms  | 7.8  |
| EfficientNetB0   |    78.9 % ± 1.6      |  77.4 – 81.7 %  |        0 → 77.4 %        | 4.1 M  | 48 MB| ~232 ms  | 4.3  |
| EfficientNetB2   |    82.6 % ± 1.1      |  80.6 – 83.9 %  |        2 → 82.8 %        | 7.8 M  | 92 MB| ~332 ms  | 3.0  |
| **YOLO11n-cls**  |  **87.3 % ± 5.2**    |  79.6 – 93.5 %  |      **3 → 93.5 %**      |**1.5 M**|**3.2 MB**|**~5 ms**|**~200**|

(Datos: `modelo_actual/outputs/seed_sweep.json`, `modelo_actual/outputs/finalize.json`,
`modelo_yolo/seed_sweep_yolo.json`.)

## Recomendación para el Raspberry Pi 5: **YOLO11n-cls (semilla 3)**

Es la mejor opción **con diferencia** para el objetivo:

- **Más preciso**: mayor accuracy media (87.3 %) y su semilla recomendada llega a
  **93.5 %** (validación 93.5 %, así que no es cherry-picking del test).
- **Más rápido**: ~5 ms/img frente a 129–332 ms de los Keras → **~30–70× más
  rápido** en CPU. En el Pi 5 será la única realmente fluida para vídeo.
- **Más liviano**: 1.5 M parámetros y **3.2 MB** de peso (vs 26–92 MB) → mínimo
  uso de RAM/almacenamiento, ideal para el Pi.

Peso desplegado: `modelo_yolo/cacao_yolo11n.pt`.

### Matices honestos

- El barrido de YOLO tiene **alta varianza** (± 5.2): es sensible a la semilla.
  La semilla 3 es un buen punto (confirmado por validación), pero para producción
  conviene **fijar la semilla 3** y, si se reentrena con más datos, volver a
  barrer semillas.
- Todos los números salen de un test de **93 imágenes** (ruidoso). El salto de
  calidad definitivo vendrá de **más datos**, no de más ajustes.
- Entre los Keras, el mejor por accuracy media robusta es **EfficientNetB2**
  (82.6 %), pero es el más pesado/lento: mala opción para el Pi frente a YOLO.

## Cómo reproducir

```bash
# Keras (3 modelos)
cd modelo_actual
python -m src.seed_sweep --seeds 0 1 2 3 4      # barrido -> outputs/seed_sweep.json
python -m src.finalize --deploy efficientnetb2  # reentrena con seed recomendada + benchmark

# YOLO
cd ../modelo_yolo
python prepare_data.py
python seed_sweep_yolo.py --seeds 0 1 2 3 4 --imgsz 160   # -> seed_sweep_yolo.json
python evaluate_yolo.py --weights runs/sweep_s3/weights/best.pt
```
