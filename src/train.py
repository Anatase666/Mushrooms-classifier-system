"""
Обучение классификатора грибов (EfficientNetB1, transfer learning + fine-tuning).

Запуск:
    python src/train.py

Перед запуском нужно выполнить src/prepare_data.py (см. README.md).
Результат: models/mushroom_efficientnet_b1.h5 и models/class_indices.json —
эти два файла нужны для последующего распознавания (predict.py / app.py).
"""

import json
import os
import random

import numpy as np
import pandas as pd
from collections import Counter
from PIL import ImageFile
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import callbacks, layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    BATCH_SIZE,
    BEST_CHECKPOINT_PATH,
    CLASS_INDICES_PATH,
    EPOCHS_STAGE1,
    EPOCHS_STAGE2,
    FINE_TUNE_AT,
    IMG_SIZE,
    MODEL_PATH,
    MODELS_DIR,
    TRAIN_DIR,
    VAL_DIR,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True  # не прерывать обучение из-за битых файлов


def build_generators():
    """Балансировка классов oversampling'ом + аугментация train, только препроцессинг для val."""
    classes = sorted(os.listdir(TRAIN_DIR))
    all_image_paths, all_labels = [], []
    for cls in classes:
        cls_folder = os.path.join(TRAIN_DIR, cls)
        files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder)]
        all_image_paths.extend(files)
        all_labels.extend([cls] * len(files))

    count_per_class = Counter(all_labels)
    max_count = max(count_per_class.values())

    balanced_paths, balanced_labels = [], []
    for cls in classes:
        cls_paths = [p for p, l in zip(all_image_paths, all_labels) if l == cls]
        oversampled = np.random.choice(cls_paths, size=max_count, replace=True)
        balanced_paths.extend(oversampled)
        balanced_labels.extend([cls] * max_count)

    combined = list(zip(balanced_paths, balanced_labels))
    random.shuffle(combined)
    balanced_paths, balanced_labels = zip(*combined)
    print("Распределение по классам после балансировки:", Counter(balanced_labels))

    train_df = pd.DataFrame({"filename": balanced_paths, "class": balanced_labels})

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="reflect",
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="filename",
        y_col="class",
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_generator = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_generator, val_generator, balanced_labels


def build_model(num_classes):
    base_model = EfficientNetB1(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs), base_model


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_generator, val_generator, balanced_labels = build_generators()
    num_classes = len(train_generator.class_indices)
    print("Индексы классов:", train_generator.class_indices)

    # Сохраняем маппинг индекс -> название класса, он понадобится для predict.py / app.py
    idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
    with open(CLASS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)
    print(f"Маппинг классов сохранён в {CLASS_INDICES_PATH}")

    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    checkpoint_cb = callbacks.ModelCheckpoint(
        BEST_CHECKPOINT_PATH, save_best_only=True, monitor="val_loss", mode="min"
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    )
    earlystop = callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=1
    )

    labels = [train_generator.class_indices[c] for c in balanced_labels]
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(labels), y=labels
    )
    class_weights = dict(enumerate(class_weights))
    print("Веса классов:", class_weights)

    # Этап 1: обучаем только "голову", база заморожена
    print("\n=== Этап 1: обучение головы модели ===")
    history1 = model.fit(
        train_generator,
        epochs=EPOCHS_STAGE1,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, reduce_lr, earlystop],
    )

    # Этап 2: fine-tuning верхних слоёв базовой сети
    print("\n=== Этап 2: fine-tuning ===")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_generator,
        epochs=EPOCHS_STAGE2,
        validation_data=val_generator,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, reduce_lr, earlystop],
        initial_epoch=history1.epoch[-1] if hasattr(history1, "epoch") else 0,
    )

    model.save(MODEL_PATH)
    print(f"\nМодель сохранена в {MODEL_PATH}")

    val_loss, val_acc = model.evaluate(val_generator)
    print(f"Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")


if __name__ == "__main__":
    main()
