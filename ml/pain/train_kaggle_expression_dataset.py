import os
import sys
import ssl
import argparse
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

DEFAULT_PAIN_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../saved_models/pain_inception.keras"))

def train_on_kaggle_expression_dataset(dataset_dir: str, output_path: str = DEFAULT_PAIN_SAVE_PATH, epochs: int = 12, batch_size: int = 16):
    """
    Trains InceptionV3 Multi-Class Facial Expression & Emotion Classification model
    supporting categories: `happy`, `neutral`, `pain` (or `grimace`/`distress`), `sad`.
    """
    print(f"==================================================")
    print(f" Kaggle Multi-Class Facial Expression Model Training ")
    print(f" Dataset Path: {dataset_dir}")
    print(f"==================================================")

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    import tensorflow as tf
    from tensorflow.keras.applications import InceptionV3
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    print("[INFO] Building InceptionV3 Multi-Class Classifier...")
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(4, activation='softmax')(x)  # 4 classes: happy, neutral, pain, sad

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    print(f"\n[INFO] Class Indices: {train_gen.class_indices}")
    print(f"[INFO] Starting InceptionV3 training for {epochs} epochs...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"\n[SUCCESS] Saved multi-class facial expression model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Class Facial Expression Model on Kaggle Datasets")
    parser.add_argument("--dataset_dir", required=True, help="Path to facial expression image dataset folder")
    parser.add_argument("--output_path", default=DEFAULT_PAIN_SAVE_PATH)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_on_kaggle_expression_dataset(args.dataset_dir, args.output_path, args.epochs, args.batch_size)
