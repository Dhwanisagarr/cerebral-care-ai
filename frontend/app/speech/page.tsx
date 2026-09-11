'use client';

import React, { useState, useEffect } from 'react';
import AudioRecorder from '@/components/AudioRecorder';
import DemoBadge from '@/components/DemoBadge';
import { analyzeSpeechAudio, checkBackendHealth, SpeechAnalysisResponse } from '@/lib/api';

export default function SpeechPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<SpeechAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<any>(null);

  useEffect(() => {
    checkBackendHealth().then((status) => setBackendStatus(status));
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setAudioPreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleAudioRecorded = (file: File) => {
    setSelectedFile(file);
    setAudioPreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await analyzeSpeechAudio(selectedFile);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Speech analysis request failed. Ensure backend API is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const isDemo = result ? result.is_demo_mode : backendStatus ? backendStatus.speech_mode === 'DEMO_MODE' : true;

  return (
    <div className="flex flex-col gap-10 max-w-4xl mx-auto py-2">
      {/* Header */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Speech Clarity Assessment
          </h1>
          <DemoBadge isDemoMode={isDemo} modelName="KNN Speech Classifier" />
        </div>
        <p className="text-slate-600 text-base sm:text-lg">
          Upload an audio file or record your voice to generate a transcript and evaluate speech clarity features (MFCC + KNN).
        </p>
      </section>

      {/* Input Selection: Upload or Record */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Option 1: File Upload */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs flex flex-col justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-slate-900 mb-1 flex items-center gap-2">
              <svg className="w-5 h-5 text-sky-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
              Upload Audio File
            </h2>
            <p className="text-xs text-slate-500 mb-4">
              Supported Formats: <strong>WAV, MP3, M4A, MP4</strong> (Max 10MB)
            </p>

            <label className="block w-full">
              <span className="sr-only">Choose audio file</span>
              <input
                type="file"
                accept=".wav,.mp3,.m4a,.flac,.ogg,.mp4"
                onChange={handleFileChange}
                disabled={isLoading}
                className="block w-full text-sm text-slate-500
                  file:mr-4 file:py-2.5 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-semibold
                  file:bg-sky-50 file:text-sky-700
                  hover:file:bg-sky-100
                  cursor-pointer focus-visible:outline-hidden"
              />
            </label>
          </div>

          {selectedFile && (
            <div className="text-xs text-slate-600 bg-slate-50 p-2.5 rounded-lg border border-slate-200 truncate">
              Selected: <strong>{selectedFile.name}</strong> ({(selectedFile.size / 1024).toFixed(1)} KB)
            </div>
          )}
        </div>

        {/* Option 2: Live Browser Microphone */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs flex flex-col gap-3">
          <h2 className="text-lg font-bold text-slate-900 flex items-center gap-2">
            <svg className="w-5 h-5 text-sky-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
            Record Voice Live
          </h2>
          <AudioRecorder onAudioRecorded={handleAudioRecorded} disabled={isLoading} />
        </div>
      </div>

      {/* Audio Playback Preview & Submit Button */}
      {selectedFile && (
        <form onSubmit={handleSubmit} className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs flex flex-col gap-6">
          <div className="flex flex-col gap-2">
            <label className="text-sm font-semibold text-slate-700">Audio Preview:</label>
            {audioPreviewUrl && (
              <audio controls src={audioPreviewUrl} className="w-full rounded-xl shadow-2xs" />
            )}
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="accessible-btn bg-sky-600 hover:bg-sky-700 text-white font-bold text-lg py-4 px-8 rounded-xl shadow-md transition-all flex items-center justify-center gap-3 disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <svg className="animate-spin h-6 w-6 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Processing Audio Pipeline...</span>
              </>
            ) : (
              <span>Analyze Speech Clarity</span>
            )}
          </button>
        </form>
      )}

      {/* Error Message */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-800 p-4 rounded-xl text-sm" role="alert">
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Process Flow Visualization */}
      <section className="bg-slate-100/70 border border-slate-200 rounded-2xl p-6">
        <h3 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">
          Pipeline Processing Workflow
        </h3>
        <div className="flex flex-wrap items-center gap-2 text-xs font-semibold text-slate-700">
          <span className="bg-white px-3 py-1.5 rounded-md border border-slate-200 shadow-2xs">Audio Input</span>
          <span>→</span>
          <span className="bg-white px-3 py-1.5 rounded-md border border-slate-200 shadow-2xs">Bandpass Filter (80-7500Hz)</span>
          <span>→</span>
          <span className="bg-white px-3 py-1.5 rounded-md border border-slate-200 shadow-2xs">Spectral Denoising</span>
          <span>→</span>
          <span className="bg-white px-3 py-1.5 rounded-md border border-slate-200 shadow-2xs">Formant Pre-Emphasis Boost</span>
          <span>→</span>
          <span className="bg-white px-3 py-1.5 rounded-md border border-slate-200 shadow-2xs">Gain Normalization</span>
          <span>→</span>
          <span className="bg-white px-3 py-1.5 rounded-md border border-slate-200 shadow-2xs">86D Acoustic Feature Extraction</span>
          <span>→</span>
          <span className="bg-white px-3 py-1.5 rounded-md border border-slate-200 shadow-2xs">KNN Classifier</span>
          <span>→</span>
          <span className="bg-white px-3 py-1.5 rounded-md border border-slate-200 shadow-2xs">Dual-Pass Whisper AI</span>
        </div>
      </section>

      {/* Results Section */}
      {result && (
        <section aria-label="Analysis Results" className="bg-white border-2 border-sky-500 rounded-2xl p-8 shadow-lg flex flex-col gap-6">
          <div className="flex items-center justify-between border-b border-slate-100 pb-4 flex-wrap gap-2">
            <div className="flex items-center gap-3 flex-wrap">
              <h2 className="text-2xl font-bold text-slate-900">Analysis Results</h2>
              {result.enhancement_applied && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold bg-emerald-100 text-emerald-800 border border-emerald-300">
                  ⚡ Speech Signal Enhancement Active
                </span>
              )}
            </div>
            <span className="text-xs font-mono text-slate-500">
              Response Time: {result.processing_time_ms} ms
            </span>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Transcription Box */}
            <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 flex flex-col gap-2">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider flex items-center justify-between">
                <span>Enhanced Speech Transcription</span>
                <span className="text-[10px] text-sky-700 font-mono">Whisper AI Dual-Pass</span>
              </span>
              <p className="text-lg font-semibold text-slate-900 italic leading-relaxed">
                "{result.transcription}"
              </p>
            </div>

            {/* Speech Clarity Box */}
            <div className="bg-slate-50 p-5 rounded-xl border border-slate-200 flex flex-col justify-between gap-3">
              <div>
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                  Speech Clarity Assessment
                </span>
                <div className="flex items-center gap-3 mt-2 flex-wrap">
                  <span className={`text-xl font-extrabold px-3 py-1 rounded-lg ${
                    result.speech_clarity.includes('Clear') 
                      ? 'bg-emerald-100 text-emerald-800 border border-emerald-300'
                      : result.speech_clarity.includes('Mild')
                      ? 'bg-amber-100 text-amber-900 border border-amber-300'
                      : 'bg-rose-100 text-rose-900 border border-rose-300'
                  }`}>
                    {result.speech_clarity}
                  </span>
                  {result.clarity_score !== undefined && (
                    <span className="text-sm font-bold text-slate-700">
                      Score: <span className="text-sky-700 font-extrabold">{result.clarity_score}%</span>
                    </span>
                  )}
                </div>
              </div>

              {/* Acoustic Metrics */}
              {result.acoustic_metrics && (
                <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-slate-200">
                  <div>
                    <span className="text-slate-500">Vocal Stability:</span>{' '}
                    <strong className="text-slate-800">{result.acoustic_metrics.vocal_stability_score}%</strong>
                  </div>
                  <div>
                    <span className="text-slate-500">Formant Definition:</span>{' '}
                    <strong className="text-slate-800">{result.acoustic_metrics.formant_definition}</strong>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Signal Enhancement Metrics Details */}
          {result.enhancement_details && (
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-600">
              <span className="font-bold text-slate-700">Applied Enhancements:</span>
              <span className="bg-sky-50 px-2.5 py-1 rounded-md border border-sky-200 text-sky-800 font-medium">Bandpass Filter (80-7500Hz)</span>
              <span className="bg-sky-50 px-2.5 py-1 rounded-md border border-sky-200 text-sky-800 font-medium">Spectral Noise Reduction</span>
              <span className="bg-sky-50 px-2.5 py-1 rounded-md border border-sky-200 text-sky-800 font-medium">Formant Pre-Emphasis Boost</span>
              <span className="bg-sky-50 px-2.5 py-1 rounded-md border border-sky-200 text-sky-800 font-medium">RMS Gain Normalization</span>
              {result.enhancement_details.snr_boost_db > 0 && (
                <span className="bg-emerald-50 px-2.5 py-1 rounded-md border border-emerald-200 text-emerald-800 font-bold">
                  +{result.enhancement_details.snr_boost_db} dB SNR Boost
                </span>
              )}
            </div>
          )}

          {/* Explanation */}
          <div className="bg-sky-50 text-sky-900 p-4 rounded-xl border border-sky-200 text-sm">
            <strong>Understanding this result:</strong> {result.explanation}
          </div>
        </section>
      )}
    </div>
  );
}
