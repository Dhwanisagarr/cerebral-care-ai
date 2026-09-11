import os
import sys
import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def evaluate_saved_model(model_path="../saved_models/speech_knn.joblib"):
    if not os.path.exists(model_path):
        print(f"[ERROR] Trained model file not found at {model_path}. Train model first.")
        return

    knn = joblib.load(model_path)
    print(f"[INFO] Evaluating model loaded from {model_path}...")

    # Generate synthetic test evaluation benchmark
    np.random.seed(99)
    X_test_clear = np.random.normal(loc=12.0, scale=3.0, size=(20, 78))
    X_test_impaired = np.random.normal(loc=65.0, scale=8.0, size=(20, 78))
    X_test = np.vstack([X_test_clear, X_test_impaired])
    y_test = np.array(["Clear"] * 20 + ["Impaired"] * 20)

    y_pred = knn.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Impaired")
    rec = recall_score(y_test, y_pred, pos_label="Impaired")
    f1 = f1_score(y_test, y_pred, pos_label="Impaired")
    cm = confusion_matrix(y_test, y_pred, labels=["Clear", "Impaired"])

    print("\n==========================================")
    print("      SPEECH CLARITY EVALUATION REPORT    ")
    print("==========================================")
    print(f"Accuracy    : {acc * 100:.2f}%")
    print(f"Precision   : {prec * 100:.2f}%")
    print(f"Recall      : {rec * 100:.2f}%")
    print(f"F1 Score    : {f1 * 100:.2f}%")
    print("\nConfusion Matrix [Clear, Impaired]:")
    print(cm)
    print("==========================================\n")

if __name__ == "__main__":
    evaluate_saved_model()
