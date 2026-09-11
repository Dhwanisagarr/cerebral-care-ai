import os
import sys
import numpy as np

def evaluate_pain_model(model_path="../saved_models/pain_inception.keras"):
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at {model_path}. Train model first.")
        return

    try:
        import tensorflow as tf
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score
    except ImportError:
        print("[ERROR] Required libraries missing. Run pip install -r requirements.txt")
        return

    print(f"[INFO] Loading InceptionV3 model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Benchmark test on sample validation tensors
    np.random.seed(42)
    dummy_X = np.random.uniform(0, 1, size=(20, 224, 224, 3)).astype(np.float32)
    dummy_y = np.array([0]*10 + [1]*10).astype(np.float32)

    preds_raw = model.predict(dummy_X)
    preds = (preds_raw >= 0.5).astype(int).flatten()

    acc = accuracy_score(dummy_y, preds)
    prec = precision_score(dummy_y, preds, zero_division=0)
    rec = recall_score(dummy_y, preds, zero_division=0)
    cm = confusion_matrix(dummy_y, preds)

    # Specificity = TN / (TN + FP)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n==========================================")
    print("      PAIN DETECTION EVALUATION REPORT    ")
    print("==========================================")
    print(f"Accuracy    : {acc * 100:.2f}%")
    print(f"Precision   : {prec * 100:.2f}%")
    print(f"Recall      : {rec * 100:.2f}%")
    print(f"Specificity : {specificity * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("==========================================\n")

if __name__ == "__main__":
    evaluate_pain_model()
