# Отчет по ДЗ 2 (Бирюков Никита)

1) Конвертация нейросети ResNet-18 в ONNX.
2) Теоретический анализ ее производительности (Arithmetic Intensity).
3) Высокопроизводительный сервер инференса Nvidia Triton с GPU.

## Step 1: Конвертация модели

В качестве базовой модели выбрана **ResNet-18** (`microsoft/resnet-18`).

**Реализация:**
1.  **Wrapper:** Создал класс-обертку, добавляющий к выходу модели линейную проекцию в размерность `(Batch, 32)`.
2.  **Export:** Экспортировал в формат ONNX с динамической осью батча (`opset_version=17`). Использовал стандартный экспортер PyTorch (БЕЗ dynamo), иначе несовмещались имена входов.
3.  **IO Names:** Вход: `INPUT_IMAGE`, Выход: `OUTPUT_PROJECTION`.

**Валидация:**
Сравнение выходов PyTorch и ONNX Runtime показало максимальный дифф `~2e-06`, что норм для FP32.

---

## Step 2: Анализ архитектуры и FLOPs

С помощью `thop` посчитал вычислительной сложности.
*   **Total GFLOPs (Batch=1):** `3.65`

### Арифметическая интенсивность (Arithmetic Intensity)

Анализ сверточного слоя блока ResNet (Conv2d 3x3, 64->64 канала, карта 56x56):

**Характеристики Nvidia A10:**
*   FP32 Performance: ~31.2 TFLOPS
*   Memory Bandwidth: ~600 GB/s
*   **Ridge Point (Порог):** $31200 / 600 \approx 52$ FLOPs/Byte.

Если AI слоя > 52, он ограничен вычислениями (Compute Bound).

**Результаты расчетов:**

| Batch | GFLOPs | Memory (MB) | AI (Ops/Byte) | Limiter (A10) |
|-------|--------|-------------|---------------|---------------|
| 4     | 0.92   | 6.57        | 140.77        | **Compute**   |
| 32    | 7.40   | 51.53       | 143.59        | **Compute**   |
| 128   | 29.60  | 205.67      | 143.90        | **Compute**   |
| 512   | 118.38 | 822.23      | 143.97        | **Compute**   |

**Вывод:** Сверточные слои ResNet являются Compute Bound (AI ~144), то есть GPU утилизируется эффективно.

---

## Step 3: Настройка Nvidia Triton

`model_repository` содержит:

1.  **`preprocess` (Python Backend):**
    *   Нормализация ImageNet (mean/std) и перестановка каналов (HWC -> CHW).
    *   Время выполнения: ~1.3 мс.
2.  **`resnet_onnx` (ONNX Runtime, GPU):**
    *   В конфигурацию добавлен `instance_group { kind: KIND_GPU }`.
    *   Выполняет инференс на Nvidia GPU.
    *   Время выполнения: ~1.6 мс.
3.  **`ensemble`:** Объединяет шаги в единый пайплайн.

**Sanity Check:**
Успешно прошел проверку:
```text
final shape: (2, 32) (what we want: [2, 32])
sanity check passed
```
