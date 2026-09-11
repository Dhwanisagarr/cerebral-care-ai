import React from 'react';

export default function AboutPage() {
  return (
    <div className="flex flex-col gap-10 max-w-4xl mx-auto py-2">
      {/* Title */}
      <section className="flex flex-col gap-3">
        <span className="text-xs font-bold uppercase tracking-wider text-sky-700">
          Platform Architecture & Technical Overview
        </span>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Enhancing Speech Clarity and Pain Detection in Individuals with Cerebral Palsy
        </h1>
        <p className="text-slate-600 text-lg">
          Assistive AI Speech Analysis & Facial Pain Detection Platform.
        </p>
      </section>

      {/* Project Background */}
      <section className="bg-white border border-slate-200 rounded-2xl p-8 shadow-xs flex flex-col gap-4">
        <h2 className="text-2xl font-bold text-slate-900">Platform Overview & Objectives</h2>
        <p className="text-slate-700 leading-relaxed text-base">
          Individuals with cerebral palsy often face significant motor impairments affecting speech articulation (dysarthria) and non-verbal pain communication. This application provides assistive AI tools designed to evaluate acoustic speech features and facial expressions to support caregivers, speech therapists, and individuals.
        </p>
        <p className="text-slate-700 leading-relaxed text-base">
          The primary objective is to deliver accurate speech transcription with enhanced noise filtering and non-invasive facial emotion/pain indicator detection using deep learning models.
        </p>
      </section>

      {/* Methodology & Machine Learning */}
      <section className="bg-white border border-slate-200 rounded-2xl p-8 shadow-xs flex flex-col gap-6">
        <h2 className="text-2xl font-bold text-slate-900">Technical Architecture & ML Pipelines</h2>

        <div className="grid md:grid-cols-2 gap-6">
          <div className="bg-sky-50 border border-sky-200 rounded-xl p-6 flex flex-col gap-3">
            <h3 className="text-xl font-bold text-sky-950">1. Speech Clarity Pipeline</h3>
            <ul className="list-disc list-inside text-sm text-sky-900 flex flex-col gap-2">
              <li><strong>Audio Preprocessing:</strong> 16kHz Butterworth bandpass filter, spectral noise reduction, and pre-emphasis formant boost.</li>
              <li><strong>Feature Extraction:</strong> 86-dimensional acoustic feature vector (MFCCs, Deltas, Delta-Deltas, Spectral Centroid/Rolloff, ZCR, RMS stability) via <code className="bg-sky-100 px-1 py-0.5 rounded">librosa</code>.</li>
              <li><strong>Classifier:</strong> K-Nearest Neighbors (KNN) model trained for speech clarity classification.</li>
              <li><strong>Transcription:</strong> OpenAI Whisper AI speech-to-text engine with acoustic phoneme fallback.</li>
            </ul>
          </div>

          <div className="bg-teal-50 border border-teal-200 rounded-xl p-6 flex flex-col gap-3">
            <h3 className="text-xl font-bold text-teal-950">2. Facial Pain & Expression Pipeline</h3>
            <ul className="list-disc list-inside text-sm text-teal-900 flex flex-col gap-2">
              <li><strong>Image Preprocessing:</strong> OpenCV landmark 2D curvature (lip arc, inner brow V-notch elevation $AU1+AU4$, eye squeeze).</li>
              <li><strong>Model Architecture:</strong> InceptionV3 Deep Neural Network with multi-class facial expression classification head.</li>
              <li><strong>Classification:</strong> 5-class expression breakdown (Crying, Disheartened, Pain, Happy, Neutral) with continuous pain index.</li>
              <li><strong>Ethical Safety:</strong> Non-diagnostic assistive indicator phrasing.</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
