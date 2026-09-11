# Assistive AI for Speech Clarity & Facial Pain Detection

> **Assistive AI Platform**: Full-stack web application for acoustic speech clarity evaluation and non-verbal facial expression analysis (*“Enhancing Speech Clarity and Pain Detection in Individuals with Cerebral Palsy”*).

---

## 📌 Project Overview

Individuals with cerebral palsy often encounter severe communication barriers due to speech articulation challenges (dysarthria) and motor impairments that prevent non-verbal expression of pain. 

This project implements an accessible, AI-assisted web application featuring two specialized tools:
1. **Speech Clarity Assessment**: Analyzes acoustic speech signals using **Mel-Frequency Cepstral Coefficients (MFCCs)** and **K-Nearest Neighbors (KNN)** classification to evaluate intelligibility and generate automated speech transcriptions.
2. **Facial Pain Indicator Detection**: Analyzes facial images using an **InceptionV3** deep learning architecture to identify potential facial expressions associated with physical pain or distress.

---

## ⚠️ Medical & Ethical Disclaimer

> **Research & Demonstration Purpose Only**:
> - This application is built for research and decision-support purposes only. It is **not** a medical diagnostic device and does not claim medical diagnostic accuracy.
> - Pain detection outputs state: *"Potential pain indicator detected"* or *"No pain indicator detected"*. It never makes absolute diagnostic claims.

---

## 🛠️ Architecture & Tech Stack

```
/Cerebral Paly
├── frontend/             # Next.js 14 (App Router, TypeScript, Tailwind CSS, WCAG AA Accessibility)
├── backend/              # Python FastAPI REST Server (Librosa, SoundFile, NoiseReduce, SpeechRecognition)
├── ml/                   # Machine Learning Training & Evaluation Pipelines
│   ├── speech/           # Audio feature extraction & KNN model scripts
│   ├── pain/             # InceptionV3 transfer learning training & evaluation scripts
│   └── saved_models/     # Serialized binary model weights (.joblib, .keras)
├── scripts/              # Local environment & runner scripts
└── README.md
```

### Technologies Used:
- **Frontend**: Next.js 14, React 18, TypeScript, Tailwind CSS, Web Audio API MediaRecorder, Web Camera Stream API.
- **Backend**: Python 3.11, FastAPI, Uvicorn, Librosa (audio processing), NoiseReduce, SpeechRecognition (Google Web Speech API), Pillow, Pydantic.
- **Machine Learning**: Scikit-Learn (KNN Classifier), TensorFlow/Keras (InceptionV3), NumPy, Joblib.

---

## 🚀 Key Features & Accessibility

- **Accessible UI**: Large touch targets (minimum 48px × 48px), high contrast color palette, screen reader ARIA labels, keyboard focus indicators (`focus-visible:ring-4`).
- **Dual Audio Input**: Upload existing audio files (`.wav`, `.mp3`, `.m4a`, `.mp4`) or record voice directly using the browser microphone.
- **Dual Visual Input**: Upload facial images (`.jpg`, `.png`, `.webp`) or capture live photos using the device camera.
- **Transparent Demo Mode**: If pre-trained model weights are missing during backend initialization, the application cleanly operates in **Demo Mode**, displaying clear badges on both backend logs and frontend UI.

---

## ⚙️ Installation & Running Locally

### 1. Prerequisites
- Node.js v20+ and `npm`
- Python 3.11+

### 2. Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
The API server will run at `http://localhost:8000`. Swagger API documentation is available at `http://localhost:8000/docs`.

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install

# Start Next.js development server
npm run dev
```
Open `http://localhost:3000` in your web browser.

---

## 🧠 Machine Learning Training & Evaluation

The `/ml` directory contains offline training and evaluation scripts separate from backend API execution.

### Train & Evaluate Speech KNN Model:
```bash
# Train KNN classifier using Librosa MFCCs
python ml/speech/train_speech_knn.py

# Evaluate trained model (Accuracy, Precision, Recall, Confusion Matrix)
python ml/speech/evaluate_speech.py
```

### Train & Evaluate Pain InceptionV3 Model:
```bash
# Train InceptionV3 transfer learning model on image dataset
python ml/pain/train_pain_inception.py

# Evaluate InceptionV3 model (Accuracy, Precision, Recall, Specificity)
python ml/pain/evaluate_pain.py
```


