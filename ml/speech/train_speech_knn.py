import os
import sys
import glob
import joblib
import numpy as np
import librosa
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend')))
from app.audio.mfcc import extract_mfcc_features

def generate_synthetic_speech_dataset(n_samples_per_class=60):
    """
    Generates 86-dimensional acoustic feature vectors simulating Clear vs Impaired dysarthric speech
    acoustic characteristics (spectral centroid, jitter, ZCR, RMS stability).
    """
    print("[INFO] Generating sample feature dataset (86 dims) for KNN training demonstration...")
    np.random.seed(42)

    # Clear speech: high formant stability, low spectral centroid variance, stable RMS
    X_clear = np.random.normal(loc=15.0, scale=2.5, size=(n_samples_per_class, 86))
    X_clear[:, 78] = np.random.normal(loc=1800.0, scale=150.0, size=n_samples_per_class) # Centroid mean
    X_clear[:, 79] = np.random.normal(loc=120.0, scale=20.0, size=n_samples_per_class)   # Centroid std (low jitter)
    X_clear[:, 82] = np.random.normal(loc=0.08, scale=0.01, size=n_samples_per_class)    # ZCR mean
    X_clear[:, 83] = np.random.normal(loc=0.02, scale=0.005, size=n_samples_per_class)   # ZCR std
    y_clear = ["Clear"] * n_samples_per_class

    # Impaired speech (dysarthric): high spectral jitter, elevated centroid variance, motor instability
    X_impaired = np.random.normal(loc=45.0, scale=6.0, size=(n_samples_per_class, 86))
    X_impaired[:, 78] = np.random.normal(loc=2400.0, scale=350.0, size=n_samples_per_class)
    X_impaired[:, 79] = np.random.normal(loc=450.0, scale=80.0, size=n_samples_per_class)   # Centroid std (high jitter)
    X_impaired[:, 82] = np.random.normal(loc=0.18, scale=0.04, size=n_samples_per_class)
    X_impaired[:, 83] = np.random.normal(loc=0.08, scale=0.02, size=n_samples_per_class)   # ZCR std
    y_impaired = ["Impaired"] * n_samples_per_class

    X = np.vstack([X_clear, X_impaired])
    y = np.array(y_clear + y_impaired)
    return X, y

DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../saved_models/speech_knn.joblib"))
DEFAULT_SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../saved_models/speech_scaler.joblib"))

def train_knn_model(dataset_dir=None, n_neighbors=5, model_output=DEFAULT_MODEL_PATH, scaler_output=DEFAULT_SCALER_PATH):
    if dataset_dir and os.path.exists(dataset_dir):
        print(f"[INFO] Loading audio dataset from {dataset_dir}...")
        X, y = [], []
        for label in ["clear", "impaired"]:
            folder = os.path.join(dataset_dir, label)
            if not os.path.exists(folder):
                continue
            audio_files = (
                glob.glob(os.path.join(folder, "*.wav")) +
                glob.glob(os.path.join(folder, "*.mp3")) +
                glob.glob(os.path.join(folder, "*.m4a"))
            )
            for f in audio_files:
                try:
                    signal, sr = librosa.load(f, sr=16000, mono=True)
                    feats = extract_mfcc_features(signal, sr=sr)
                    X.append(feats)
                    y.append("Clear" if label == "clear" else "Impaired")
                except Exception as e:
                    print(f"[WARN] Could not process {f}: {e}")
        X = np.array(X)
        y = np.array(y)
    else:
        print("[INFO] Dataset directory not provided or not found. Training on synthetic sample dataset.")
        X, y = generate_synthetic_speech_dataset()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[INFO] Training K-Nearest Neighbors (n_neighbors={n_neighbors})...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance')
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print("\n=== Speech Clarity KNN Model Evaluation Results ===")
    print(f"Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs(os.path.dirname(model_output), exist_ok=True)
    joblib.dump(knn, model_output)
    joblib.dump(scaler, scaler_output)
    print(f"\n[SUCCESS] Saved trained Speech KNN model to {model_output}")
    print(f"[SUCCESS] Saved StandardScaler to {scaler_output}")

if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else None
    train_knn_model(dataset_dir=dataset_path)
