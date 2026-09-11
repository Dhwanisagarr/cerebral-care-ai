'use client';

import React, { useState, useEffect } from 'react';
import CameraCaptureModal from '@/components/CameraCaptureModal';
import DemoBadge from '@/components/DemoBadge';
import { analyzePainImage, checkBackendHealth, PainAnalysisResponse } from '@/lib/api';

export default function PainPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PainAnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [backendStatus, setBackendStatus] = useState<any>(null);

  useEffect(() => {
    checkBackendHealth().then((status) => setBackendStatus(status));
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setImagePreviewUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleImageCaptured = (file: File) => {
    setSelectedFile(file);
    setImagePreviewUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;

    setIsLoading(true);
    setError(null);

    try {
      const data = await analyzePainImage(selectedFile);
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'Pain image analysis failed. Ensure backend API is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const isDemo = result ? result.is_demo_mode : backendStatus ? backendStatus.pain_mode === 'DEMO_MODE' : true;

  return (
    <div className="flex flex-col gap-10 max-w-4xl mx-auto py-2">
      {/* Header */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
            Facial Image Pain Detection
          </h1>
          <DemoBadge isDemoMode={isDemo} modelName="InceptionV3 Deep Classifier" />
        </div>
        <p className="text-slate-600 text-base sm:text-lg">
          Analyze a facial photo for potential pain indicators using deep convolutional neural features (InceptionV3 architecture).
        </p>
      </section>

      {/* Input Methods: File Upload or Camera */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Upload Card */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs flex flex-col justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-slate-900 mb-1 flex items-center gap-2">
              <svg className="w-5 h-5 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
              Upload Facial Image
            </h2>
            <p className="text-xs text-slate-500 mb-4">
              Supported Formats: <strong>JPG, PNG, WebP</strong> (Max 5MB)
            </p>

            <label className="block w-full">
              <span className="sr-only">Choose image file</span>
              <input
                type="file"
                accept=".jpg,.jpeg,.png,.webp"
                onChange={handleFileChange}
                disabled={isLoading}
                className="block w-full text-sm text-slate-500
                  file:mr-4 file:py-2.5 file:px-4
                  file:rounded-lg file:border-0
                  file:text-sm file:font-semibold
                  file:bg-teal-50 file:text-teal-700
                  hover:file:bg-teal-100
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

        {/* Camera Option */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs flex flex-col justify-between gap-4">
          <div>
            <h2 className="text-lg font-bold text-slate-900 mb-1 flex items-center gap-2">
              <svg className="w-5 h-5 text-teal-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              Use Web Camera
            </h2>
            <p className="text-xs text-slate-500 mb-4">
              Capture a photo directly from your device's web camera.
            </p>
          </div>

          <button
            type="button"
            onClick={() => setIsCameraOpen(true)}
            disabled={isLoading}
            className="accessible-btn bg-teal-600 hover:bg-teal-700 text-white font-semibold px-6 py-3 rounded-xl shadow-md transition-all flex items-center justify-center gap-2"
          >
            Open Camera
          </button>
        </div>
      </div>

      <CameraCaptureModal
        isOpen={isCameraOpen}
        onClose={() => setIsCameraOpen(false)}
        onImageCaptured={handleImageCaptured}
      />

      {/* Image Preview & Submit Form */}
      {selectedFile && imagePreviewUrl && (
        <form onSubmit={handleSubmit} className="bg-white border border-slate-200 rounded-2xl p-6 shadow-xs flex flex-col gap-6">
          <div className="flex flex-col items-center gap-3">
            <label className="text-sm font-semibold text-slate-700 self-start">Image Preview (Resized to 224×224 for InceptionV3):</label>
            <div className="relative w-64 h-64 border-2 border-dashed border-teal-400 rounded-2xl overflow-hidden bg-slate-100 shadow-inner flex items-center justify-center">
              <img
                src={imagePreviewUrl}
                alt="Selected facial preview"
                className="w-full h-full object-cover"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="accessible-btn bg-teal-600 hover:bg-teal-700 text-white font-bold text-lg py-4 px-8 rounded-xl shadow-md transition-all flex items-center justify-center gap-3 disabled:opacity-50"
          >
            {isLoading ? (
              <>
                <svg className="animate-spin h-6 w-6 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                <span>Analyzing Facial Image...</span>
              </>
            ) : (
              <span>Analyze Facial Pain Indicator</span>
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

      {/* Results Section */}
      {result && (
        <section aria-label="Pain Analysis Results" className="bg-white border-2 border-teal-500 rounded-2xl p-8 shadow-lg flex flex-col gap-6">
          <div className="flex items-center justify-between border-b border-slate-100 pb-4 flex-wrap gap-2">
            <div className="flex items-center gap-3 flex-wrap">
              <h2 className="text-2xl font-bold text-slate-900">Analysis Output</h2>
              {result.detected_expression && (
                <span className={`inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-black shadow-xs ${
                  result.detected_expression.includes('Crying') || result.detected_expression.includes('Pain')
                    ? 'bg-rose-100 text-rose-900 border border-rose-300'
                    : result.detected_expression.includes('Disheartened')
                    ? 'bg-amber-100 text-amber-900 border border-amber-300'
                    : result.detected_expression.includes('Happy')
                    ? 'bg-emerald-100 text-emerald-900 border border-emerald-300'
                    : 'bg-sky-100 text-sky-900 border border-sky-300'
                }`}>
                  {result.detected_expression.includes('Crying') && '😭 '}
                  {result.detected_expression.includes('Disheartened') && '😔 '}
                  {result.detected_expression.includes('Happy') && '😊 '}
                  {result.detected_expression.includes('Neutral') && '😐 '}
                  {result.detected_expression.includes('Pain') && '😖 '}
                  Detected Expression: {result.detected_expression}
                </span>
              )}
            </div>
            <span className="text-xs font-mono text-slate-500">
              Response Time: {result.processing_time_ms} ms
            </span>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            {/* Classification Card */}
            <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 flex flex-col justify-between gap-4">
              <div>
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                  Pain / Distress Indicator Assessment
                </span>
                <div className="flex items-center gap-3 mt-3 flex-wrap">
                  <span className={`text-xl sm:text-2xl font-black px-5 py-2.5 rounded-xl border ${
                    result.intensity_level === 'High'
                      ? 'bg-rose-100 text-rose-900 border-rose-300'
                      : result.intensity_level === 'Moderate'
                      ? 'bg-amber-100 text-amber-900 border-amber-300'
                      : 'bg-emerald-100 text-emerald-900 border-emerald-300'
                  }`}>
                    {result.result}
                  </span>
                </div>
              </div>

              {result.pain_score !== undefined && (
                <div className="flex items-center justify-between pt-3 border-t border-slate-200 text-sm font-bold text-slate-700">
                  <span>Pain / Distress Index:</span>
                  <span className={`font-extrabold text-lg ${result.pain_score > 40 ? 'text-rose-700' : 'text-emerald-700'}`}>
                    {result.pain_score}%
                  </span>
                </div>
              )}
            </div>

            {/* Emotion Probabilities Breakdown */}
            {result.emotion_probabilities && (
              <div className="bg-slate-50 p-6 rounded-xl border border-slate-200 flex flex-col justify-between gap-3">
                <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">
                  Multi-Class Emotion & Distress Probabilities
                </span>
                
                <div className="flex flex-col gap-2 text-xs pt-1">
                  {result.emotion_probabilities.crying_pct !== undefined && (
                    <div>
                      <div className="flex justify-between font-bold text-slate-700 mb-1">
                        <span>😭 Crying / Acute Distress:</span>
                        <span>{result.emotion_probabilities.crying_pct}%</span>
                      </div>
                      <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                        <div className="bg-rose-600 h-2 rounded-full transition-all" style={{ width: `${result.emotion_probabilities.crying_pct}%` }} />
                      </div>
                    </div>
                  )}

                  {result.emotion_probabilities.disheartened_pct !== undefined && (
                    <div>
                      <div className="flex justify-between font-bold text-slate-700 mb-1">
                        <span>😔 Disheartened / Sadness:</span>
                        <span>{result.emotion_probabilities.disheartened_pct}%</span>
                      </div>
                      <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                        <div className="bg-amber-500 h-2 rounded-full transition-all" style={{ width: `${result.emotion_probabilities.disheartened_pct}%` }} />
                      </div>
                    </div>
                  )}

                  <div>
                    <div className="flex justify-between font-bold text-slate-700 mb-1">
                      <span>😊 Happy / Smiling:</span>
                      <span>{result.emotion_probabilities.happy_pct}%</span>
                    </div>
                    <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                      <div className="bg-emerald-500 h-2 rounded-full transition-all" style={{ width: `${result.emotion_probabilities.happy_pct}%` }} />
                    </div>
                  </div>

                  <div>
                    <div className="flex justify-between font-bold text-slate-700 mb-1">
                      <span>😐 Neutral / Calm:</span>
                      <span>{result.emotion_probabilities.neutral_pct}%</span>
                    </div>
                    <div className="w-full bg-slate-200 h-2 rounded-full overflow-hidden">
                      <div className="bg-sky-500 h-2 rounded-full transition-all" style={{ width: `${result.emotion_probabilities.neutral_pct}%` }} />
                    </div>
                  </div>
                </div>

                {result.facial_metrics && (
                  <div className="text-[11px] text-slate-600 pt-2 border-t border-slate-200 flex justify-between flex-wrap gap-1">
                    <span>Mouth Arc: <strong>{result.facial_metrics.mouth_arc_direction || result.facial_metrics.mouth_state}</strong></span>
                    <span>Brow Position: <strong>{result.facial_metrics.brow_furrow_intensity}</strong></span>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="bg-amber-50 text-amber-900 p-4 rounded-xl border border-amber-200 text-sm">
            <strong>Ethical & Safety Notice:</strong> {result.disclaimer}
          </div>
        </section>
      )}
    </div>
  );
}
