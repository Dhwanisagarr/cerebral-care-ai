import os
import tempfile
import time
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.audio.preprocessor import preprocess_audio_file
from app.audio.mfcc import extract_mfcc_features
from app.audio.transcriber import transcribe_audio_file
from app.vision.preprocessor import preprocess_image_file
from app.models.speech_knn import speech_knn_instance
from app.models.pain_inception import pain_inception_instance

app = FastAPI(
    title="Assistive Speech & Pain AI API",
    description="Backend API for speech clarity assessment and facial image pain detection in individuals with cerebral palsy.",
    version="1.0.0"
)

# Enable CORS for Next.js frontend development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    speech_model_loaded: bool
    pain_model_loaded: bool
    speech_mode: str
    pain_mode: str
    disclaimer: str

@app.get("/api/health", response_model=HealthResponse)
def health_check():
    speech_loaded = speech_knn_instance.is_loaded
    pain_loaded = pain_inception_instance.is_loaded
    return HealthResponse(
        status="online",
        speech_model_loaded=speech_loaded,
        pain_model_loaded=pain_loaded,
        speech_mode="REAL_MODEL" if speech_loaded else "DEMO_MODE",
        pain_mode="REAL_MODEL" if pain_loaded else "DEMO_MODE",
        disclaimer="This tool is for research and demonstration purposes only. It is not a medical diagnostic device."
    )

@app.post("/api/speech/analyze")
async def analyze_speech(file: UploadFile = File(...)):
    start_time = time.time()
    
    # 1. Extension check
    file_ext = Path(file.filename or "audio.wav").suffix.lower()
    if file_ext not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format '{file_ext}'. Allowed formats: {', '.join(sorted(settings.ALLOWED_AUDIO_EXTENSIONS))}"
        )

    # 2. File read & size check
    contents = await file.read()
    max_bytes = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio file exceeds maximum size limit of {settings.MAX_AUDIO_SIZE_MB}MB."
        )

    # 3. Save to temporary input file
    temp_input = tempfile.NamedTemporaryFile(suffix=file_ext, delete=False)
    temp_input_path = temp_input.name
    clean_wav_path = None
    enhanced_wav_path = None

    try:
        temp_input.write(contents)
        temp_input.close()

        # 4. Preprocess audio: multi-stage speech signal enhancement & formant boost
        audio_signal, sr, clean_wav_path, enhanced_wav_path, enhancement_details = preprocess_audio_file(temp_input_path)

        # 5. Extract 86-dimensional acoustic features
        mfcc_vector = extract_mfcc_features(audio_signal, sr=sr)

        # 6. KNN Model Prediction & Clarity Assessment
        clarity_result, clarity_score, acoustic_metrics, is_speech_demo = speech_knn_instance.predict(mfcc_vector)

        # 7. Speech Transcription (Dual-pass Whisper on enhanced audio)
        transcription_text = transcribe_audio_file(clean_wav_path, enhanced_wav_path)

        processing_time_ms = round((time.time() - start_time) * 1000, 2)

        explanation = (
            "The speech clarity model evaluates pitch stability, formant definition, and spectral characteristics (MFCCs) "
            "to assess intelligibility and monitor speech therapy progress over time."
        )

        return JSONResponse(content={
            "status": "success",
            "transcription": transcription_text,
            "speech_clarity": clarity_result,
            "clarity_score": clarity_score,
            "enhancement_applied": True,
            "enhancement_details": enhancement_details,
            "acoustic_metrics": acoustic_metrics,
            "is_demo_mode": is_speech_demo,
            "processing_time_ms": processing_time_ms,
            "explanation": explanation
        })

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Audio analysis failed: {str(e)}")
    finally:
        # Cleanup temporary files
        for p in [temp_input_path, clean_wav_path, enhanced_wav_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

@app.post("/api/pain/analyze")
async def analyze_pain(file: UploadFile = File(...)):
    start_time = time.time()

    # 1. Extension check
    file_ext = Path(file.filename or "image.jpg").suffix.lower()
    if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format '{file_ext}'. Allowed formats: {', '.join(sorted(settings.ALLOWED_IMAGE_EXTENSIONS))}"
        )

    # 2. File read & size check
    contents = await file.read()
    max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image file exceeds maximum size limit of {settings.MAX_IMAGE_SIZE_MB}MB."
        )

    try:
        # 3. Preprocess image: resize (224, 224) & extract facial expression landmarks
        image_tensor, facial_metrics = preprocess_image_file(contents)

        # 4. Multi-class Emotion & Facial Expression Classifier Pass
        result_phrase, detected_expression, pain_score, intensity_level, emotion_probabilities, facial_metrics, is_pain_demo = pain_inception_instance.predict(image_tensor, facial_metrics)

        processing_time_ms = round((time.time() - start_time) * 1000, 2)

        return JSONResponse(content={
            "status": "success",
            "result": result_phrase,
            "detected_expression": detected_expression,
            "pain_score": pain_score,
            "intensity_level": intensity_level,
            "emotion_probabilities": emotion_probabilities,
            "facial_metrics": facial_metrics,
            "is_demo_mode": is_pain_demo,
            "processing_time_ms": processing_time_ms,
            "disclaimer": "This tool is for research and demonstration purposes only. It is not a medical diagnostic device."
        })

    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Image analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
