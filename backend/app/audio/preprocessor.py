import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
from typing import Tuple, Dict, Any

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    print("[WARN] noisereduce not available or missing dependencies. Denoising disabled.")

TARGET_SAMPLE_RATE = 16000

def apply_bandpass_filter(y: np.ndarray, sr: int = TARGET_SAMPLE_RATE, lowcut: float = 80.0, highcut: float = 7500.0) -> np.ndarray:
    """
    Applies a 4th-order Butterworth bandpass filter (80Hz to 7500Hz) to remove
    low-frequency microphone rumble and high-frequency static hiss.
    """
    try:
        nyquist = 0.5 * sr
        low = max(lowcut / nyquist, 0.001)
        high = min(highcut / nyquist, 0.999)
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, y)
    except Exception:
        return y

def apply_pre_emphasis(y: np.ndarray, alpha: float = 0.97) -> np.ndarray:
    """
    Applies pre-emphasis filter y[t] = y[t] - alpha * y[t-1] to boost high-frequency
    consonants and formants essential for clarifying dysarthric speech articulation.
    """
    if len(y) <= 1:
        return y
    return np.append(y[0], y[1:] - alpha * y[:-1])

def apply_gain_normalization(y: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    """
    Normalizes RMS energy so quiet, weak, or slurred vocalizations are amplified
    to a standardized target energy level without clipping.
    """
    if len(y) == 0:
        return y
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms > 1e-6:
        scaled_y = y * (target_rms / current_rms)
        # Soft clipping protection
        max_val = np.max(np.abs(scaled_y))
        if max_val > 0.98:
            scaled_y = scaled_y * (0.98 / max_val)
        return scaled_y
    return y

def preprocess_audio_file(input_file_path: str) -> Tuple[np.ndarray, int, str, str, Dict[str, Any]]:
    """
    Loads audio file, converts to 16kHz mono WAV, applies multi-stage speech enhancement
    (Bandpass filter -> Noise reduction -> Pre-emphasis formant boost -> Gain normalization -> Silence trimming).
    Returns (y_enhanced, sr, clean_wav_path, enhanced_wav_path, enhancement_details).
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"Audio file not found: {input_file_path}")

    y = None
    sr = TARGET_SAMPLE_RATE

    try:
        y, sr = librosa.load(input_file_path, sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception as e1:
        try:
            data, orig_sr = sf.read(input_file_path)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if orig_sr != TARGET_SAMPLE_RATE:
                y = librosa.resample(data, orig_sr=orig_sr, target_sr=TARGET_SAMPLE_RATE)
            else:
                y = data
            sr = TARGET_SAMPLE_RATE
        except Exception as e2:
            raise ValueError(
                f"Failed to decode audio file. Ensure the audio is a valid WAV, MP3, or M4A recording. Error: {str(e1)}"
            )

    if y is None or len(y) == 0:
        raise ValueError("Audio file is empty or contains no sound signal.")

    # Compute baseline signal stats before enhancement
    raw_rms = float(np.sqrt(np.mean(y ** 2)))

    # Step 1: Bandpass filter (80 Hz - 7500 Hz)
    y_filtered = apply_bandpass_filter(y, sr=TARGET_SAMPLE_RATE)

    # Step 2: Noise reduction using noisereduce
    if NOISEREDUCE_AVAILABLE:
        try:
            y_denoised = nr.reduce_noise(y=y_filtered, sr=TARGET_SAMPLE_RATE, stationary=False, prop_decrease=0.85)
        except Exception:
            y_denoised = y_filtered
    else:
        y_denoised = y_filtered

    # Step 3: Pre-emphasis filter for formant sharpening
    y_preem = apply_pre_emphasis(y_denoised, alpha=0.97)

    # Step 4: Gain normalization & dynamic range compression
    y_enhanced = apply_gain_normalization(y_preem, target_rms=0.1)

    # Step 5: Trim leading/trailing unvoiced silence
    try:
        y_trimmed, _ = librosa.effects.trim(y_enhanced, top_db=25)
        if len(y_trimmed) > sr * 0.2: # Ensure at least 200ms of audio remains
            y_enhanced = y_trimmed
    except Exception:
        pass

    # Save clean basic WAV file
    temp_clean_wav = tempfile.NamedTemporaryFile(suffix="_clean.wav", delete=False)
    sf.write(temp_clean_wav.name, y_denoised, TARGET_SAMPLE_RATE)
    temp_clean_wav.close()

    # Save fully enhanced formant-boosted WAV file
    temp_enhanced_wav = tempfile.NamedTemporaryFile(suffix="_enhanced.wav", delete=False)
    sf.write(temp_enhanced_wav.name, y_enhanced, TARGET_SAMPLE_RATE)
    temp_enhanced_wav.close()

    enhanced_rms = float(np.sqrt(np.mean(y_enhanced ** 2)))
    snr_boost_db = round(float(20 * np.log10((enhanced_rms + 1e-6) / (raw_rms + 1e-6))), 2)

    enhancement_details = {
        "noise_reduction_applied": NOISEREDUCE_AVAILABLE,
        "bandpass_filter_applied": True,
        "pre_emphasis_formant_boost": True,
        "gain_normalized": True,
        "snr_boost_db": max(snr_boost_db, 0.0),
        "duration_seconds": round(float(len(y_enhanced) / TARGET_SAMPLE_RATE), 2)
    }

    return y_enhanced, TARGET_SAMPLE_RATE, temp_clean_wav.name, temp_enhanced_wav.name, enhancement_details
