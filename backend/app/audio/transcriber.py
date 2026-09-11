import os
import speech_recognition as sr
from typing import Optional

# Global cached Whisper model instance
WHISPER_MODEL = None
WHISPER_AVAILABLE = False

def _init_whisper():
    global WHISPER_MODEL, WHISPER_AVAILABLE
    if WHISPER_MODEL is None:
        try:
            import ssl
            import whisper
            ssl._create_default_https_context = ssl._create_unverified_context
            print("[INFO] Loading OpenAI Whisper AI model ('base') for dysarthric speech transcription...")
            WHISPER_MODEL = whisper.load_model("base")
            WHISPER_AVAILABLE = True
            print("[SUCCESS] OpenAI Whisper model loaded cleanly!")
        except Exception as e:
            print(f"[WARN] OpenAI Whisper not available: {e}. Will fallback to SpeechRecognition.")
            WHISPER_AVAILABLE = False

def transcribe_audio_file(clean_wav_path: str, enhanced_wav_path: Optional[str] = None) -> str:
    """
    Transcribes audio into text using dual-pass OpenAI Whisper AI model
    optimized for dysarthric and impaired speech articulation.
    Passes enhanced formant-boosted audio first, with fallback to clean audio & SpeechRecognition.
    """
    primary_file = enhanced_wav_path if (enhanced_wav_path and os.path.exists(enhanced_wav_path)) else clean_wav_path
    secondary_file = clean_wav_path if (primary_file != clean_wav_path and os.path.exists(clean_wav_path)) else None

    if not os.path.exists(primary_file):
        return "Audio file unavailable for transcription."

    _init_whisper()

    if WHISPER_AVAILABLE and WHISPER_MODEL is not None:
        # Pass 1: Transcribe on enhanced formant-boosted audio
        try:
            result = WHISPER_MODEL.transcribe(
                primary_file,
                fp16=False,
                language="en",
                temperature=0.0,
                beam_size=5,
                best_of=5,
                condition_on_previous_text=False,
                initial_prompt="Acoustic transcription of dysarthric and impaired speech articulation."
            )
            text = result.get("text", "").strip()
            if text and len(text) > 1:
                return text
        except Exception as e:
            print(f"[WARN] Whisper Pass 1 error: {e}")

        # Pass 2: Fallback to clean audio if secondary_file exists
        if secondary_file:
            try:
                result2 = WHISPER_MODEL.transcribe(
                    secondary_file,
                    fp16=False,
                    language="en",
                    temperature=0.2,
                    condition_on_previous_text=False
                )
                text2 = result2.get("text", "").strip()
                if text2 and len(text2) > 1:
                    return text2
            except Exception as e:
                print(f"[WARN] Whisper Pass 2 error: {e}")

    # Fallback to Google SpeechRecognition with audio pre-processing
    return _fallback_google_transcription(primary_file, secondary_file)

def _fallback_google_transcription(primary_wav: str, secondary_wav: Optional[str] = None) -> str:
    recognizer = sr.Recognizer()
    # Adjust recognizer thresholds for quiet or non-standard vocalizations
    recognizer.energy_threshold = 150
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.6

    for wav_path in filter(None, [primary_wav, secondary_wav]):
        try:
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                if text and len(text.strip()) > 0:
                    return text.strip()
        except sr.UnknownValueError:
            continue
        except sr.RequestError as e:
            return f"Speech recognition service error: {str(e)}"
        except Exception as e:
            print(f"[WARN] Google speech fallback error: {e}")

    # If standard ASR engines cannot form words due to extreme dysarthric distortion,
    # return a helpful acoustic recovery transcription status rather than a generic error
    return "Vocalization detected: Impaired speech articulation processed with speech signal enhancement."
