import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    MAX_AUDIO_SIZE_MB: int = 10
    MAX_IMAGE_SIZE_MB: int = 5
    SPEECH_MODEL_PATH: str = str(BASE_DIR.parent / "ml" / "saved_models" / "speech_knn.joblib")
    PAIN_MODEL_PATH: str = str(BASE_DIR.parent / "ml" / "saved_models" / "pain_inception.keras")
    ALLOWED_AUDIO_EXTENSIONS: set = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mp4"}
    ALLOWED_IMAGE_EXTENSIONS: set = {".jpg", ".jpeg", ".png", ".webp"}

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
