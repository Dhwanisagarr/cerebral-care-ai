# Cerebral Care AI

Cerebral Care AI is an AI-based assistive application for speech clarity analysis and facial pain indicator detection for individuals with cerebral palsy.

**Repository:** [https://github.com/Dhwanisagarr/cerebral-care-ai](https://github.com/Dhwanisagarr/cerebral-care-ai)

---

## Overview

Individuals with cerebral palsy often experience motor impairments that affect speech articulation (dysarthria) and non-verbal communication, making it difficult to express physical discomfort or pain. 

**Cerebral Care AI** provides an accessible web application combining signal processing, acoustic feature extraction, and deep learning vision models to assist caregivers, speech therapists, and individuals with cerebral palsy.

- **Speech Clarity Analysis**: Enhances audio signals, extracts acoustic features, classifies speech clarity, and generates automated speech transcriptions.
- **Pain Indicator Detection**: Processes facial images or camera captures to analyze facial expression geometry and detect potential non-verbal indicators of physical discomfort or pain.

---

## Features

### Speech Clarity Analysis
- **Audio Upload**: Upload `.wav`, `.mp3`, `.m4a`, `.flac`, or `.ogg` audio recordings.
- **Browser Recording**: Live microphone capture directly within the web interface.
- **Audio Preprocessing**: Butterworth bandpass filtering (80Hz–7500Hz), spectral noise reduction, pre-emphasis formant boost, and RMS normalization.
- **MFCC Extraction**: 86-dimensional acoustic feature vector (13 MFCCs + $\Delta$ + $\Delta^2$, Spectral Centroid, Rolloff, ZCR, RMS stability).
- **KNN Classification**: K-Nearest Neighbors model trained to evaluate speech clarity score (0–100%) and multi-tier clarity levels.
- **Transcription**: OpenAI Whisper AI speech recognition with acoustic phoneme fallback.
- **Result Display**: Clear breakdown of speech clarity score, dysarthria classification, transcription, and acoustic metrics.

### Pain Indicator Detection
- **Facial Image Upload**: Upload `.jpg`, `.jpeg`, `.png`, or `.webp` facial photographs.
- **Camera Capture**: Integrated web camera snapshot capture.
- **Image Preprocessing**: Face detection, landmark extraction (mouth curvature, inner brow elevation $AU1+AU4$, eye squeeze), and RGB normalization (224×224×3).
- **InceptionV3-Based Classification**: Deep Convolutional Neural Network multi-class facial expression classifier.
- **Pain / No Pain Indicator Result**: Continuous pain confidence index (0–100%), expression probability breakdown (Crying, Disheartened, Pain, Happy, Neutral), and potential pain indicator status.

---

## How It Works

### Speech Analysis Pipeline

```text
Audio Upload / Recording
        ↓
Audio Preprocessing (Filtering & Noise Reduction)
        ↓
MFCC & Acoustic Feature Extraction (86D)
        ↓
KNN Speech Classification
        ↓
Speech Clarity Result Score (%)
        ↓
Speech Transcription (Whisper AI)
```

### Pain Indicator Pipeline

```text
Image Upload / Camera Capture
        ↓
Image Preprocessing & Landmark Extraction
        ↓
InceptionV3 Neural Network Analysis
        ↓
Multi-Class Expression & Pain Classification
        ↓
Potential Pain Indicator Result (%)
```

---

## Tech Stack

### Frontend
- **Framework**: Next.js 14 (App Router)
- **UI Library**: React 18, TypeScript
- **Styling**: Vanilla CSS / Tailwind CSS design system (WCAG AA accessible)
- **Media Handling**: Web Audio API, MediaRecorder API, HTML5 Canvas Camera Stream

### Backend
- **Server**: Python 3.11, FastAPI, Uvicorn
- **Audio Processing**: Librosa, SoundFile, NoiseReduce, SciPy, PyTorch
- **Vision & ML**: TensorFlow / Keras (InceptionV3), OpenCV (`cv2`), scikit-learn (`Joblib`)
- **Speech Recognition**: OpenAI Whisper, SpeechRecognition

---

## Project Structure

```text
cerebral-care-ai/
├── frontend/             # Next.js 14 + React + TypeScript web interface
│   ├── app/              # App Router pages (speech, pain, about)
│   ├── components/       # UI components & medical disclaimer banner
│   └── lib/              # API client and TypeScript interfaces
├── backend/              # Python FastAPI REST Server
│   ├── app/              # Audio & vision preprocessing, models, REST API endpoints
│   └── tests/            # Pytest suite for API endpoints
├── ml/                   # Machine Learning pipelines & scripts
│   ├── speech/           # Speech feature extraction & KNN model training scripts
│   ├── pain/             # InceptionV3 training & evaluation scripts
│   └── saved_models/     # Model weights (.joblib, .keras)
├── scripts/              # Development and helper scripts
├── .gitignore            # Git ignore rules
└── README.md             # Project documentation
```

---

## Running Locally

### Prerequisites
- Node.js 18+ and `npm`
- Python 3.10+ and `pip`
- FFmpeg (for audio format conversions)

### 1. Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
*API Swagger Documentation will be available at `http://localhost:8000/docs`.*

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*Web Application will be available at `http://localhost:3000`.*

---

## Machine Learning

The repository contains full training and evaluation scripts for both ML pipelines under `ml/`:

- **Speech Model**: `ml/speech/train_speech_knn.py` extracts 86-dimensional acoustic features and fits a `StandardScaler` + `KNeighborsClassifier`. A CLI utility `train_kaggle_speech_dataset.py` is provided for training on raw audio dataset directories.
- **Pain & Expression Model**: `ml/pain/train_pain_inception.py` builds an InceptionV3 transfer learning model with custom classification layers. A CLI utility `train_kaggle_expression_dataset.py` is provided for training multi-class facial expression datasets.
- **Evaluation**: `ml/pain/evaluate_pain.py` generates classification reports and evaluation metrics.

Pre-trained model artifacts are loaded by the FastAPI server from `ml/saved_models/`. If model weights are absent, the application gracefully provides heuristic signal fallbacks.

---

## Screenshots

*Screenshots will be added after the production UI is finalized.*

---

## Future Improvements

- **Larger Datasets**: Expand speech models with larger dysarthric speech corpora (e.g., TORGO database).
- **Advanced Speech Encoders**: Integrate self-supervised speech representations (Wav2Vec 2.0 / HuBERT) for fine-grained dysarthria severity scoring.
- **Enhanced Facial Models**: Incorporate 3D facial mesh landmark dynamics (MediaPipe / AU Action Units) for continuous micro-expression analysis.
- **Real-Time Video Stream**: Live webcam video stream processing for continuous pain monitoring.
- **Multimodal Fusion**: Jointly model synchronized audio-visual streams for multimodal caregiving insights.

---

## Project Background

The original project was an academic capstone project focused on assistive technology for individuals with cerebral palsy. This repository is a modern full-stack web reimplementation based on the original project specifications and report documentation, rebuilt to provide a accessible web interface and RESTful ML API.

---

## Disclaimer

This application is a research and portfolio project. It is not intended to diagnose medical conditions, determine pain severity, or replace professional medical assessment.
