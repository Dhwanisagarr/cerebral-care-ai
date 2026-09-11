import os
import sys
import ssl
import argparse
import numpy as np

# Workaround for macOS Python SSL certificate verification when fetching Keras weights
ssl._create_default_https_context = ssl._create_unverified_context

DEFAULT_PAIN_SAVE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../saved_models/pain_inception.keras"))

def train_on_kaggle_pain_dataset(dataset_dir: str, output_path: str = DEFAULT_PAIN_SAVE_PATH, epochs: int = 10, batch_size: int = 16):
    """
    Trains InceptionV3 Transfer Learning model on Kaggle Facial Pain datasets.
    Dataset structure can be:
    - Folders: `dataset_dir/pain/` and `dataset_dir/no_pain/` (or `normal/`)
    - Subfolders by subject ID or pain severity grade.
    """
    print(f"==================================================")
    print(f" Kaggle Facial Pain Dataset Transfer Learning ")
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

    print("[INFO] Building InceptionV3 Deep Neural Network Architecture...")
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  # Freeze base layers for initial transfer learning

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

    # Data Augmentation & Normalization
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        dataset_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    print(f"\n[INFO] Starting InceptionV3 training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"\n[SUCCESS] Saved trained InceptionV3 model to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train InceptionV3 Pain Detection Model on Kaggle Datasets")
    parser.add_argument("--dataset_dir", required=True, help="Path to facial pain image dataset folder")
    parser.add_argument("--output_path", default=DEFAULT_PAIN_SAVE_PATH)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    train_on_kaggle_pain_dataset(args.dataset_dir, args.output_path, args.epochs, args.batch_size)
