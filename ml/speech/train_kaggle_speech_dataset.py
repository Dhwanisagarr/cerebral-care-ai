import os
import sys
import glob
import argparse
import joblib
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
from app.audio.mfcc import extract_mfcc_features
from app.audio.preprocessor import apply_bandpass_filter, apply_pre_emphasis, apply_gain_normalization

def load_and_preprocess_audio(file_path: str, sr: int = 16000) -> np.ndarray:
    """
    Loads audio file, applies bandpass filter, pre-emphasis, and gain normalization,
    and returns enhanced audio array for acoustic feature extraction.
    """
    y, orig_sr = librosa.load(file_path, sr=sr, mono=True)
    y_filt = apply_bandpass_filter(y, sr=sr)
    y_preem = apply_pre_emphasis(y_filt)
    y_norm = apply_gain_normalization(y_preem)
    return y_norm

def train_on_kaggle_dataset(dataset_dir: str, output_model_path: str, output_scaler_path: str, n_neighbors: int = 5):
    """
    Trains KNN speech clarity model on Kaggle dysarthric speech datasets.
    Dataset structure can be:
    - Folders: `dataset_dir/clear/` and `dataset_dir/impaired/` (or `dysarthric/`)
    - Subfolders by speaker ID or speech severity rating.
    """
    print(f"==================================================")
    print(f" Kaggle Speech Dataset Training & Feature Extraction ")
    print(f" Dataset Path: {dataset_dir}")
    print(f"==================================================")

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    X, y = [], []
    categories = {"clear": "Clear", "normal": "Clear", "impaired": "Impaired", "dysarthric": "Impaired", "cp": "Impaired"}

    for root, dirs, files in os.walk(dataset_dir):
        folder_name = os.path.basename(root).lower()
        matched_label = None
        for cat_key, cat_val in categories.items():
            if cat_key in folder_name:
                matched_label = cat_val
                break

        if matched_label:
            audio_files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.m4a', '.flac'))]
            print(f"[INFO] Processing {len(audio_files)} audio files in '{root}' (Class: {matched_label})...")
            for fname in audio_files:
                fpath = os.path.join(root, fname)
                try:
                    enhanced_signal = load_and_preprocess_audio(fpath)
                    feats = extract_mfcc_features(enhanced_signal, sr=16000)
                    X.append(feats)
                    y.append(matched_label)
                except Exception as e:
                    print(f"  [WARN] Failed processing {fname}: {e}")

    X = np.array(X)
    y = np.array(y)

    if len(X) < 10:
        print(f"[ERROR] Insufficient audio files extracted ({len(X)} samples found). Ensure folder structure contains 'clear' and 'impaired'/'dysarthric' subfolders.")
        sys.exit(1)

    print(f"\n[INFO] Total dataset loaded: {len(X)} samples across classes: {np.unique(y, return_counts=True)}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[INFO] Training K-Nearest Neighbors Classifier (n_neighbors={n_neighbors})...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='euclidean')
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Model Training Evaluation Results ===")
    print(f"Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(knn, output_model_path)
    joblib.dump(scaler, output_scaler_path)
    print(f"\n[SUCCESS] Trained model saved to: {output_model_path}")
    print(f"[SUCCESS] Feature scaler saved to: {output_scaler_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Speech Clarity KNN Model on Kaggle Dysarthric Speech Datasets")
    parser.add_argument("--dataset_dir", required=True, help="Path to raw audio dataset folder")
    parser.add_argument("--model_output", default=os.path.join(os.path.dirname(__file__), "../saved_models/speech_knn.joblib"))
    parser.add_argument("--scaler_output", default=os.path.join(os.path.dirname(__file__), "../saved_models/speech_scaler.joblib"))
    parser.add_argument("--n_neighbors", type=int, default=5)
    args = parser.parse_args()

    train_on_kaggle_dataset(args.dataset_dir, args.model_output, args.scaler_output, args.n_neighbors)
