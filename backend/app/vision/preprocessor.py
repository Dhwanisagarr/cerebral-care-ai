from PIL import Image
import numpy as np
import cv2
import io
from typing import Tuple, Dict, Any

IMAGE_SIZE = (224, 224)

# Load OpenCV frontal face & eye cascade classifiers
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def extract_opencv_facial_metrics(img_np: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes facial image using computer vision techniques to evaluate:
    - Smile Arc vs Downturned Crying Mouth Arc
    - Inner Brow V-Notch Elevation & Furrowing (AU1 + AU4)
    - Eye Tear Squeeze & Contraction (AU6 / AU7)
    - Crying / Distress / Disheartened Expression Scores
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    face_detected = len(faces) > 0

    smile_score = 0.0
    crying_score = 0.0
    disheartened_score = 0.0
    neutral_score = 0.0
    brow_furrow_score = 0.0
    brow_v_notch_score = 0.0
    eye_contraction_score = 0.0
    mouth_grimace_score = 0.0
    mouth_arc_direction = "Relaxed Neutral Arc"

    if face_detected:
        x, y, w, h = faces[0]
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_hsv = hsv[y:y+h, x:x+w]

        # 1. Eyebrow & Inner Forehead Region (Top 35% of face)
        brow_roi = face_roi_gray[0:int(h * 0.35), :]
        if brow_roi.size > 0:
            sobel_v = cv2.Sobel(brow_roi, cv2.CV_64F, 0, 1, ksize=3)
            brow_var = float(np.mean(np.abs(sobel_v)))
            brow_furrow_score = min(1.0, max(0.0, (brow_var - 10.0) / 30.0))

            # Inner brow elevation V-notch (AU1 + AU4)
            center_w = brow_roi.shape[1] // 2
            inner_brow_roi = brow_roi[:, max(0, center_w-20):min(brow_roi.shape[1], center_w+20)]
            if inner_brow_roi.size > 0:
                inner_v = cv2.Sobel(inner_brow_roi, cv2.CV_64F, 0, 1, ksize=3)
                brow_v_notch_score = min(1.0, max(0.0, float(np.mean(np.abs(inner_v))) / 25.0))

        # 2. Eye Squeeze / Tear Contraction (Upper 20% to 50%)
        eye_roi = face_roi_gray[int(h * 0.20):int(h * 0.50), :]
        if eye_roi.size > 0:
            eyes = EYE_CASCADE.detectMultiScale(eye_roi)
            if len(eyes) >= 2:
                eye_contraction_score = 0.10
            elif len(eyes) == 1:
                eye_contraction_score = 0.35
            else:
                eye_contraction_score = 0.70  # Eye squeeze during crying or severe pain

        # 3. Mouth Arc & Geometry Analysis (Bottom 40% of face)
        mouth_roi_gray = face_roi_gray[int(h * 0.60):h, :]
        mouth_roi_hsv = face_roi_hsv[int(h * 0.60):h, :]

        if mouth_roi_gray.size > 0:
            val_channel = mouth_roi_hsv[:, :, 2]
            sobel_h = cv2.Sobel(mouth_roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            mouth_grad = float(np.mean(np.abs(sobel_h)))

            center_h, center_w = val_channel.shape[0] // 2, val_channel.shape[1] // 2
            center_bright = float(np.mean(val_channel[max(0, center_h-10):center_h+10, max(0, center_w-15):center_w+15]))
            outer_bright = float(np.mean(val_channel))
            bright_diff = center_bright - outer_bright

            # Measure 2D vertical lip corner curvature vs lip center
            corners_y = float((np.mean(mouth_roi_gray[0:10, 0:15]) + np.mean(mouth_roi_gray[0:10, -15:])) / 2.0)
            center_lip_y = float(np.mean(mouth_roi_gray[center_h-5:center_h+5, center_w-10:center_w+10]))

            # Upward Smile Arc vs Downturned Crying Arc
            if center_lip_y > corners_y and bright_diff > 12.0 and brow_furrow_score < 0.35:
                mouth_arc_direction = "Upward Smile Arc"
                smile_score = min(1.0, max(0.35, (bright_diff / 45.0) + 0.30))
            elif center_lip_y <= corners_y or brow_furrow_score >= 0.40:
                mouth_arc_direction = "Downturned Crying/Distress Arc"
                smile_score = 0.0
                if bright_diff > 10.0 or mouth_grad > 15.0:
                    # Open crying mouth or tight downturned grimace
                    crying_score = min(1.0, max(0.40, (mouth_grad / 30.0) + (brow_furrow_score * 0.4)))
                    disheartened_score = min(1.0, max(0.35, brow_furrow_score * 0.7))

            if brow_furrow_score > 0.45 or crying_score > 0.40:
                mouth_grimace_score = min(1.0, max(0.35, (mouth_grad - 8.0) / 25.0))

        if brow_furrow_score < 0.25 and smile_score < 0.25 and crying_score < 0.25:
            neutral_score = min(1.0, max(0.50, 1.0 - (brow_furrow_score + smile_score)))
        else:
            neutral_score = 0.15
    else:
        # Full-frame anatomical region partitioning fallback
        h_total, w_total = gray.shape[:2]
        top_brow_roi = gray[0:int(h_total * 0.35), :]
        bottom_mouth_roi = gray[int(h_total * 0.60):h_total, :]

        sobel_v = cv2.Sobel(top_brow_roi, cv2.CV_64F, 0, 1, ksize=3)
        sobel_h = cv2.Sobel(bottom_mouth_roi, cv2.CV_64F, 1, 0, ksize=3)

        brow_grad = float(np.mean(np.abs(sobel_v)))
        mouth_grad = float(np.mean(np.abs(sobel_h)))

        if brow_grad > 18.0:
            # Elevated brow furrowing (Action Unit 4 crying / pain grimace)
            brow_furrow_score = min(1.0, max(0.50, (brow_grad - 12.0) / 22.0))
            crying_score = min(1.0, max(0.45, (mouth_grad / 30.0) + 0.30))
            mouth_grimace_score = min(1.0, max(0.35, (mouth_grad - 10.0) / 25.0))
            eye_contraction_score = 0.65
            mouth_arc_direction = "Downturned Crying/Distress Arc"
            smile_score = 0.0
            neutral_score = 0.10
        else:
            neutral_score = 0.85
            smile_score = 0.05
            brow_furrow_score = 0.05
            mouth_grimace_score = 0.05

    composite_tension = round(0.40 * brow_furrow_score + 0.30 * eye_contraction_score + 0.30 * max(crying_score, mouth_grimace_score), 2)

    return {
        "face_detected": face_detected,
        "smile_confidence": round(smile_score, 2),
        "crying_confidence": round(crying_score, 2),
        "disheartened_confidence": round(disheartened_score, 2),
        "neutral_confidence": round(neutral_score, 2),
        "mouth_arc_direction": mouth_arc_direction,
        "brow_furrow_intensity": "High (V-Notch)" if brow_v_notch_score > 0.50 else ("Moderate" if brow_furrow_score > 0.35 else "Low (Relaxed)"),
        "eye_contraction_intensity": "High" if eye_contraction_score > 0.60 else ("Moderate" if eye_contraction_score > 0.35 else "Low (Open)"),
        "mouth_state": "Open Crying / Downturned Arc" if crying_score > 0.40 else ("Smiling / Upward Arc" if smile_score > 0.35 else "Relaxed / Neutral"),
        "facial_tension_score": composite_tension,
        "brow_score_raw": round(brow_furrow_score, 2),
        "crying_score_raw": round(crying_score, 2),
        "mouth_score_raw": round(mouth_grimace_score, 2)
    }

def preprocess_image_file(image_bytes: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Loads image from bytes, resizes to (224, 224) RGB, normalizes pixel array,
    runs OpenCV facial expression analysis, and returns (image_tensor, facial_metrics).
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img_np = np.array(img)

        # OpenCV Facial Expression Feature Extraction
        facial_metrics = extract_opencv_facial_metrics(img_np)

        # Prepare normalized tensor for InceptionV3 input
        img_resized = img.resize(IMAGE_SIZE)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_tensor = np.expand_dims(img_array, axis=0)

        return img_tensor, facial_metrics
    except Exception as e:
        raise ValueError(f"Failed to process image file. Ensure valid JPG or PNG format. Error: {str(e)}")
