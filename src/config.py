"""
Общие настройки проекта.
Меняйте параметры здесь, а не в разных файлах — так они не разъедутся
между обучением (train.py) и распознаванием (predict.py / app.py).
"""

import os

# --- Данные -----------------------------------------------------------
KAGGLE_DATASET = "maysee/mushrooms-classification-common-genuss-images"
SPLIT_DIR = "mushrooms_split"          # куда кладём train/ и val/ после разбиения
TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "val")

# --- Модель / обучение --------------------------------------------------
IMG_SIZE = 240
BATCH_SIZE = 16
EPOCHS_STAGE1 = 10          # обучение только "головы" (базовая сеть заморожена)
EPOCHS_STAGE2 = 20          # fine-tuning (размораживаем верхние слои базовой сети)
FINE_TUNE_AT = 100          # индекс слоя EfficientNetB1, начиная с которого размораживаем

# --- Куда сохраняем результаты -----------------------------------------
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "mushroom_efficientnet_b1.h5")
BEST_CHECKPOINT_PATH = os.path.join(MODELS_DIR, "best_efficientnetb1.h5")
CLASS_INDICES_PATH = os.path.join(MODELS_DIR, "class_indices.json")
