import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple
from app.config import settings

class SpeechKNNModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self._load_model()

    def _load_model(self):
        model_path = settings.SPEECH_MODEL_PATH
        scaler_path = os.path.join(os.path.dirname(model_path), "speech_scaler.joblib")

        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                self.is_loaded = True
                print(f"[INFO] Loaded Speech KNN model from {model_path}")
            except Exception as e:
                print(f"[WARN] Failed to load Speech KNN model: {e}. Switching to DEMO MODE.")
                self.is_loaded = False
        else:
            print(f"[INFO] Speech model file not found at {model_path}. Operating in DEMO MODE.")
            self.is_loaded = False

    def predict(self, feature_vector: np.ndarray) -> Tuple[str, float, Dict[str, Any], bool]:
        """
        Predicts speech clarity from an 86-dimensional acoustic feature vector.
        Returns (clarity_label, clarity_score_pct, acoustic_metrics, is_demo_mode).
        """
        features = feature_vector.reshape(1, -1)

        # Calculate acoustic breakdown metrics from feature vector components
        # (Feature layout: 0..77 = MFCCs/Deltas, 78..79 = Spectral Centroid, 80..81 = Rolloff, 82..83 = ZCR, 84..85 = RMS)
        cent_std = float(features[0, 79]) if features.shape[1] >= 80 else 100.0
        zcr_std = float(features[0, 83]) if features.shape[1] >= 84 else 0.05
        rms_std = float(features[0, 85]) if features.shape[1] >= 86 else 0.02

        # Vocal stability index derived from spectral jitter & energy variance
        vocal_stability_pct = round(max(5.0, min(98.0, 100.0 - (cent_std / 15.0 + zcr_std * 200.0 + rms_std * 300.0))), 1)

        if self.is_loaded and self.model is not None:
            try:
                if self.scaler is not None and features.shape[1] == getattr(self.scaler, 'n_features_in_', features.shape[1]):
                    proc_features = self.scaler.transform(features)
                else:
                    proc_features = features

                # Predict probability if supported, else kneighbors distance
                if hasattr(self.model, "predict_proba"):
                    probs = self.model.predict_proba(proc_features)[0]
                    classes = list(self.model.classes_)
                    clear_idx = classes.index("Clear") if "Clear" in classes else 0
                    clarity_prob = float(probs[clear_idx])
                else:
                    pred_label = str(self.model.predict(proc_features)[0])
                    clarity_prob = 0.85 if pred_label == "Clear" else 0.35

                clarity_score = round(clarity_prob * 100.0, 1)

                if clarity_score >= 70.0:
                    label = "Clear Speech"
                elif clarity_score >= 45.0:
                    label = "Mild Dysarthria (Moderate Clarity)"
                else:
                    label = "Impaired Speech (Severe Dysarthria)"

                acoustic_metrics = {
                    "vocal_stability_score": vocal_stability_pct,
                    "formant_definition": "High" if vocal_stability_pct > 75.0 else ("Moderate" if vocal_stability_pct > 45.0 else "Low"),
                    "spectral_variability": round(cent_std, 2)
                }

                return label, clarity_score, acoustic_metrics, False

            except Exception as e:
                print(f"[ERROR] Speech KNN inference failed: {e}. Falling back to acoustic analysis.")

        # Fallback / Acoustic evaluation mode
        clarity_score = vocal_stability_pct
        if clarity_score >= 65.0:
            label = "Clear Speech"
        elif clarity_score >= 40.0:
            label = "Mild Dysarthria (Moderate Clarity)"
        else:
            label = "Impaired Speech (Severe Dysarthria)"

        acoustic_metrics = {
            "vocal_stability_score": vocal_stability_pct,
            "formant_definition": "High" if vocal_stability_pct > 70.0 else ("Moderate" if vocal_stability_pct > 40.0 else "Low"),
            "spectral_variability": round(cent_std, 2)
        }

        return label, clarity_score, acoustic_metrics, True

speech_knn_instance = SpeechKNNModel()
