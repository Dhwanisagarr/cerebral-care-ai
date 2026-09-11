import os
import numpy as np
from typing import Tuple, Dict, Any
from app.config import settings

class PainInceptionModel:
    def __init__(self):
        self.model = None
        self.ort_session = None
        self.input_name = None
        self.output_name = None
        self.is_loaded = False
        self.is_onnx = False
        self._attempted_load = False

    @property
    def is_available(self) -> bool:
        """Returns True if the pain model file exists or if it has been loaded."""
        return self.is_loaded or os.path.exists(settings.PAIN_MODEL_PATH)

    def _ensure_loaded(self):
        """Lazy loads ONNX runtime or TensorFlow model on first demand."""
        if self._attempted_load:
            return

        self._attempted_load = True
        model_path = settings.PAIN_MODEL_PATH
        if os.path.exists(model_path):
            try:
                if model_path.endswith(".onnx"):
                    import onnxruntime as ort
                    options = ort.SessionOptions()
                    options.intra_op_num_threads = 1
                    options.inter_op_num_threads = 1
                    self.ort_session = ort.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
                    self.input_name = self.ort_session.get_inputs()[0].name
                    self.output_name = self.ort_session.get_outputs()[0].name
                    self.is_onnx = True
                    self.is_loaded = True
                    print(f"[INFO] Lazy-loaded Pain InceptionV3 ONNX model from {model_path}")
                else:
                    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
                    os.environ["MKL_NUM_THREADS"] = "1"
                    os.environ["OMP_NUM_THREADS"] = "1"
                    import tensorflow as tf
                    try:
                        tf.config.set_visible_devices([], 'GPU')
                    except Exception:
                        pass
                    self.model = tf.keras.models.load_model(model_path)
                    self.is_loaded = True
                    print(f"[INFO] Lazy-loaded Pain InceptionV3 Keras model from {model_path}")
            except Exception as e:
                print(f"[WARN] Failed to load Pain InceptionV3 model from {model_path}: {e}. Operating in heuristic mode.")
                self.is_loaded = False
        else:
            print(f"[INFO] Pain model file not found at {model_path}. Operating in heuristic mode.")
            self.is_loaded = False

    def predict(self, image_tensor: np.ndarray, facial_metrics: Dict[str, Any]) -> Tuple[str, str, float, str, Dict[str, float], Dict[str, Any], bool]:
        """
        Evaluates facial expression (Crying, Disheartened, Pain/Grimace, Happy, Neutral)
        and outputs pain & distress indicators.
        Returns (result_phrase, detected_expression, pain_score_pct, intensity_level, emotion_probabilities, facial_metrics, is_demo_mode).
        """
        self._ensure_loaded()
        smile_conf = float(facial_metrics.get("smile_confidence", 0.0))
        crying_conf = float(facial_metrics.get("crying_confidence", 0.0))
        disheartened_conf = float(facial_metrics.get("disheartened_confidence", 0.0))
        neutral_conf = float(facial_metrics.get("neutral_confidence", 0.0))
        brow_raw = float(facial_metrics.get("brow_score_raw", 0.0))
        cv_tension = float(facial_metrics.get("facial_tension_score", 0.0))
        mouth_arc = str(facial_metrics.get("mouth_arc_direction", ""))

        nn_prob = 0.5
        if self.is_loaded:
            try:
                if self.is_onnx and self.ort_session is not None:
                    ort_inputs = {self.input_name: image_tensor.astype(np.float32)}
                    prediction = self.ort_session.run([self.output_name], ort_inputs)[0]
                    nn_prob = float(prediction[0][0])
                elif self.model is not None:
                    prediction = self.model.predict(image_tensor, verbose=0)
                    nn_prob = float(prediction[0][0])
            except Exception as e:
                print(f"[WARN] Neural network prediction pass error: {e}")

        # 1. Crying / Acute Pain & Distress (Open crying mouth or downturned arc + brow V-notch)
        if crying_conf >= 0.35 or (brow_raw >= 0.40 and mouth_arc == "Downturned Crying/Distress Arc"):
            crying_prob = min(0.98, max(0.75, 0.40 * nn_prob + 0.60 * max(crying_conf, cv_tension)))
            happy_prob = 0.01
            neutral_prob = round(max(0.01, 1.0 - crying_prob - 0.05), 2)
            sad_prob = 0.04

            detected_expression = "Crying / Acute Pain & Distress"
            phrase = "Potential pain indicator detected (Crying & Acute Distress)"
            intensity = "High"
            pain_score_pct = round(crying_prob * 100.0, 1)

        # 2. Disheartened / Sadness & Discomfort
        elif disheartened_conf >= 0.35 or (brow_raw >= 0.35 and smile_conf < 0.15):
            sad_prob = min(0.92, max(0.60, disheartened_conf + 0.35))
            crying_prob = round((1.0 - sad_prob) * 0.40, 2)
            pain_prob = round(sad_prob * 0.85, 2)
            happy_prob = 0.02
            neutral_prob = round(max(0.01, 1.0 - sad_prob - crying_prob - happy_prob), 2)

            detected_expression = "Disheartened / Sadness & Discomfort"
            phrase = "Potential pain / distress indicator detected (Disheartened Expression)"
            intensity = "Moderate"
            pain_score_pct = round(pain_prob * 100.0, 1)

        # 3. Pain / Physical Grimace
        elif brow_raw >= 0.45 and cv_tension >= 0.40:
            pain_prob = min(0.96, max(0.70, 0.50 * nn_prob + 0.50 * cv_tension))
            happy_prob = 0.02
            neutral_prob = round(max(0.01, 1.0 - pain_prob - 0.05), 2)
            crying_prob = 0.03
            sad_prob = 0.02

            detected_expression = "Pain / Physical Distress"
            phrase = "Potential pain indicator detected (Pain/Grimace Expression)"
            intensity = "High" if pain_prob >= 0.80 else "Moderate"
            pain_score_pct = round(pain_prob * 100.0, 1)

        # 4. Happy / Smiling (Requires upward mouth arc and relaxed brows)
        elif smile_conf >= 0.30 and mouth_arc == "Upward Smile Arc" and brow_raw < 0.35:
            happy_prob = min(0.98, max(0.70, smile_conf + 0.45))
            neutral_prob = round(max(0.01, 1.0 - happy_prob - 0.02), 2)
            crying_prob = 0.01
            sad_prob = 0.01
            pain_prob = 0.02

            detected_expression = "Happy / Smiling"
            phrase = "No pain indicator detected (Expressing Happiness)"
            intensity = "None"
            pain_score_pct = round(pain_prob * 100.0, 1)

        # 5. Neutral / Calm (Relaxed facial posture)
        else:
            neutral_prob = min(0.95, max(0.65, neutral_conf + 0.40))
            happy_prob = round((1.0 - neutral_prob) * 0.40, 2)
            crying_prob = 0.02
            sad_prob = round(max(0.01, 1.0 - neutral_prob - happy_prob - crying_prob), 2)
            pain_prob = 0.05

            detected_expression = "Neutral / Calm"
            phrase = "No pain indicator detected (Neutral Expression)"
            intensity = "None"
            pain_score_pct = round(pain_prob * 100.0, 1)

        emotion_probabilities = {
            "crying_pct": round(crying_prob * 100.0, 1),
            "disheartened_pct": round(sad_prob * 100.0, 1),
            "happy_pct": round(happy_prob * 100.0, 1),
            "neutral_pct": round(neutral_prob * 100.0, 1)
        }

        return phrase, detected_expression, pain_score_pct, intensity, emotion_probabilities, facial_metrics, not self.is_loaded

pain_inception_instance = PainInceptionModel()
