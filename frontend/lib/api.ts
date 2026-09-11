const getApiBaseUrl = (): string => {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL.replace(/\/$/, '');
  }
  if (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    return 'https://cerebral-care-ai.onrender.com';
  }
  return 'http://localhost:8000';
};

const API_BASE_URL = getApiBaseUrl();

export interface HealthStatus {
  status: string;
  speech_model_loaded: boolean;
  pain_model_loaded: boolean;
  speech_mode: 'REAL_MODEL' | 'DEMO_MODE';
  pain_mode: 'REAL_MODEL' | 'DEMO_MODE';
  disclaimer: string;
}

export interface SpeechAnalysisResponse {
  status: string;
  transcription: string;
  speech_clarity: string;
  clarity_score?: number;
  enhancement_applied?: boolean;
  enhancement_details?: {
    noise_reduction_applied: boolean;
    bandpass_filter_applied: boolean;
    pre_emphasis_formant_boost: boolean;
    gain_normalized: boolean;
    snr_boost_db: number;
    duration_seconds: number;
  };
  acoustic_metrics?: {
    vocal_stability_score: number;
    formant_definition: string;
    spectral_variability: number;
  };
  is_demo_mode: boolean;
  processing_time_ms: number;
  explanation: string;
}

export interface PainAnalysisResponse {
  status: string;
  result: string;
  detected_expression?: string;
  pain_score?: number;
  intensity_level?: 'High' | 'Moderate' | 'None' | string;
  emotion_probabilities?: {
    crying_pct?: number;
    disheartened_pct?: number;
    happy_pct?: number;
    neutral_pct?: number;
    pain_pct?: number;
    sad_pct?: number;
  };
  facial_metrics?: {
    face_detected: boolean;
    smile_confidence?: number;
    crying_confidence?: number;
    neutral_confidence?: number;
    mouth_arc_direction?: string;
    brow_furrow_intensity: string;
    eye_contraction_intensity: string;
    mouth_state?: string;
    facial_tension_score: number;
  };
  is_demo_mode: boolean;
  processing_time_ms: number;
  disclaimer: string;
}

export async function checkBackendHealth(): Promise<HealthStatus | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, { cache: 'no-store' });
    if (!res.ok) return null;
    return await res.json();
  } catch (error) {
    console.warn('Backend healthcheck failed:', error);
    return null;
  }
}

export async function analyzeSpeechAudio(file: File | Blob): Promise<SpeechAnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file, file instanceof File ? file.name : 'recorded_speech.wav');

  const res = await fetch(`${API_BASE_URL}/api/speech/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Speech analysis request failed.' }));
    throw new Error(errorData.detail || `Server error (${res.status})`);
  }

  return await res.json();
}

export async function analyzePainImage(file: File | Blob): Promise<PainAnalysisResponse> {
  const formData = new FormData();
  formData.append('file', file, file instanceof File ? file.name : 'facial_capture.jpg');

  const res = await fetch(`${API_BASE_URL}/api/pain/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ detail: 'Pain analysis request failed.' }));
    throw new Error(errorData.detail || `Server error (${res.status})`);
  }

  return await res.json();
}
