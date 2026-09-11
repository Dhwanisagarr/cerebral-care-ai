import numpy as np
import librosa

def extract_mfcc_features(y: np.ndarray, sr: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """
    Extracts comprehensive acoustic features from audio signal y:
    - 13 MFCCs + deltas + delta-deltas (mean & std)
    - Spectral Centroid, Spectral Rolloff, Zero Crossing Rate (ZCR), RMS Energy (mean & std)
    Yields an 86-dimensional feature vector measuring speech intelligibility and dysarthria.
    """
    if len(y) == 0:
        raise ValueError("Cannot extract acoustic features from empty audio array.")

    # 1. 13 MFCCs & Deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_mean, mfcc_std = np.mean(mfcc, axis=1), np.std(mfcc, axis=1)
    delta_mean, delta_std = np.mean(mfcc_delta, axis=1), np.std(mfcc_delta, axis=1)
    delta2_mean, delta2_std = np.mean(mfcc_delta2, axis=1), np.std(mfcc_delta2, axis=1)

    # 2. Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean, cent_std = np.mean(spec_cent), np.std(spec_cent)

    # 3. Spectral Rolloff
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    roll_mean, roll_std = np.mean(spec_roll), np.std(spec_roll)

    # 4. Zero Crossing Rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean, zcr_std = np.mean(zcr), np.std(zcr)

    # 5. RMS Energy
    rms = librosa.feature.rms(y=y)
    rms_mean, rms_std = np.mean(rms), np.std(rms)

    # Combine into 86-dimensional feature vector
    feature_vector = np.hstack([
        mfcc_mean, mfcc_std,
        delta_mean, delta_std,
        delta2_mean, delta2_std,
        np.array([cent_mean, cent_std, roll_mean, roll_std, zcr_mean, zcr_std, rms_mean, rms_std])
    ])

    return feature_vector
