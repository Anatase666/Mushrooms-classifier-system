"""
Простое веб-приложение для распознавания грибов по фотографии.

Запуск:
    python app.py

После запуска откроется локальный адрес (обычно http://127.0.0.1:7860) —
откройте его в браузере, перетащите туда фото гриба и получите результат.

Требует обученную модель: models/mushroom_efficientnet_b1.h5
и models/class_indices.json (создаются скриптом src/train.py).
"""

import json
import os
import sys

import gradio as gr
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from config import CLASS_INDICES_PATH, IMG_SIZE, MODEL_PATH  # noqa: E402

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_INDICES_PATH):
    sys.exit(
        "Не найдена обученная модель.\n"
        f"Ожидались файлы:\n  {MODEL_PATH}\n  {CLASS_INDICES_PATH}\n\n"
        "Сначала выполните:\n"
        "  python src/prepare_data.py\n"
        "  python src/train.py"
    )

print("Загружаю модель...")
model = load_model(MODEL_PATH)
with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
    idx_to_class = {int(k): v for k, v in json.load(f).items()}
print("Модель загружена, классы:", list(idx_to_class.values()))


def classify(img):
    if img is None:
        return {}
    img = img.resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img).astype("float32")
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]
    return {idx_to_class[i]: float(preds[i]) for i in range(len(preds))}


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="pil", label="Загрузите фото гриба"),
    outputs=gr.Label(num_top_classes=5, label="Вероятные виды"),
    title="Классификатор лесных грибов",
    description=(
        "Загрузите фотографию гриба, и модель предложит наиболее вероятные виды. "
       
    ),
    examples=None,
)

if __name__ == "__main__":
    demo.launch()
