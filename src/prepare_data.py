"""
Скачивает датасет грибов с Kaggle и делит его на train/val (80/20)
с сохранением пропорций классов.

Запуск:
    python src/prepare_data.py

Перед запуском нужно один раз настроить доступ к Kaggle API:
см. README.md, раздел "Настройка Kaggle API".
"""

import os
import shutil
from pathlib import Path

import kagglehub
from sklearn.model_selection import train_test_split

from config import KAGGLE_DATASET, SPLIT_DIR, TRAIN_DIR, VAL_DIR


def main():
    print(f"Скачиваю датасет '{KAGGLE_DATASET}' с Kaggle...")
    data_dir = kagglehub.dataset_download(KAGGLE_DATASET)
    data_dir = os.path.join(data_dir, "Mushrooms")

    all_images, all_labels = [], []
    for class_dir in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        for img_file in os.listdir(class_path):
            if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_images.append(os.path.join(class_path, img_file))
                all_labels.append(class_dir)

    print(f"Всего изображений: {len(all_images)}")
    print(f"Всего классов: {len(set(all_labels))}")

    train_imgs, val_imgs = train_test_split(
        all_images, test_size=0.2, random_state=42, stratify=all_labels
    )
    print(f"Train: {len(train_imgs)} изображений")
    print(f"Val:   {len(val_imgs)} изображений")

    if os.path.exists(SPLIT_DIR):
        shutil.rmtree(SPLIT_DIR)

    for split_name, split_imgs in [("train", train_imgs), ("val", val_imgs)]:
        split_dir = os.path.join(SPLIT_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for img_path in split_imgs:
            class_name = Path(img_path).parent.name
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy2(img_path, os.path.join(class_dir, os.path.basename(img_path)))

    print(f"\nГотово. Датасет разбит и сохранён в '{SPLIT_DIR}'")
    print(f"  Train: {TRAIN_DIR}")
    print(f"  Val:   {VAL_DIR}")


if __name__ == "__main__":
    main()
