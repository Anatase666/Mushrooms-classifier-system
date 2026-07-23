"""
Распознавание гриба на одной фотографии с помощью обученной модели.

Использование:
    python src/predict.py путь/к/фото.jpg
    python src/predict.py путь/к/фото.jpg --top 5

Требует наличия обученной модели (models/mushroom_efficientnet_b1.h5)
и файла с маппингом классов (models/class_indices.json) — оба создаются
скриптом train.py.
"""

import argparse
import json
import os
import sys

import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image

from config import CLASS_INDICES_PATH, IMG_SIZE, MODEL_PATH


def load_class_names():
    if not os.path.exists(CLASS_INDICES_PATH):
        sys.exit(
            f"Не найден файл {CLASS_INDICES_PATH}. "
            "Сначала обучите модель командой: python src/train.py"
        )
    with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    # ключи в json всегда строки, приводим обратно к int
    return {int(k): v for k, v in idx_to_class.items()}


def predict_image(model_path, img_path, top_k=3):
    if not os.path.exists(model_path):
        sys.exit(
            f"Не найден файл модели {model_path}. "
            "Сначала обучите модель командой: python src/train.py"
        )
    if not os.path.exists(img_path):
        sys.exit(f"Файл изображения не найден: {img_path}")

    idx_to_class = load_class_names()
    model = load_model(model_path)

    img = keras_image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = keras_image.img_to_array(img)
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]
    top_indices = preds.argsort()[-top_k:][::-1]

    print(f"\nРезультат для файла: {img_path}\n")
    for i, idx in enumerate(top_indices, start=1):
        class_name = idx_to_class.get(idx, f"class_{idx}")
        print(f"{i}. {class_name:30s} — {preds[idx] * 100:.2f}%")

    return idx_to_class[top_indices[0]], float(preds[top_indices[0]])


def main():
    parser = argparse.ArgumentParser(description="Классификация гриба по фотографии")
    parser.add_argument("image", help="Путь к файлу изображения (jpg/png)")
    parser.add_argument(
        "--top", type=int, default=3, help="Сколько наиболее вероятных классов показать"
    )
    parser.add_argument(
        "--model", default=MODEL_PATH, help="Путь к файлу модели (.h5)"
    )
    args = parser.parse_args()

    predict_image(args.model, args.image, top_k=args.top)


if __name__ == "__main__":
    main()
