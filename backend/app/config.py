import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BASE_DIR.parent

def _resolve_speech_model_path() -> str:
    if os.getenv("SPEECH_MODEL_PATH"):
        return os.getenv("SPEECH_MODEL_PATH")
    p1 = ROOT_DIR / "ml" / "saved_models" / "speech_knn.joblib"
    if p1.exists():
        return str(p1)
    p2 = BASE_DIR / "ml" / "saved_models" / "speech_knn.joblib"
    return str(p2)

def _resolve_pain_model_path() -> str:
    if os.getenv("PAIN_MODEL_PATH"):
        return os.getenv("PAIN_MODEL_PATH")
    p_onnx1 = ROOT_DIR / "ml" / "saved_models" / "pain_inception.onnx"
    if p_onnx1.exists():
        return str(p_onnx1)
    p_onnx2 = BASE_DIR / "ml" / "saved_models" / "pain_inception.onnx"
    if p_onnx2.exists():
        return str(p_onnx2)
    p1 = ROOT_DIR / "ml" / "saved_models" / "pain_inception.keras"
    if p1.exists():
        return str(p1)
    p2 = BASE_DIR / "ml" / "saved_models" / "pain_inception.keras"
    return str(p2)

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    MAX_AUDIO_SIZE_MB: int = 4
    MAX_IMAGE_SIZE_MB: int = 4
    SPEECH_MODEL_PATH: str = _resolve_speech_model_path()
    PAIN_MODEL_PATH: str = _resolve_pain_model_path()
    ALLOWED_AUDIO_EXTENSIONS: set = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4"}
    ALLOWED_IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
