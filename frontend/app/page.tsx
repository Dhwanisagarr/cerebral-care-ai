import React from 'react';
import Link from 'next/link';

export default function HomePage() {
  return (
    <div className="flex flex-col gap-12 py-4">
      {/* Hero Section */}
      <section className="text-center max-w-3xl mx-auto flex flex-col items-center gap-6">
        <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full text-sm font-semibold bg-sky-100 text-sky-800 border border-sky-200">
          <span className="w-2 h-2 rounded-full bg-sky-600" aria-hidden="true" />
          Assistive Technology Platform
        </span>
        
        <h1 className="text-4xl sm:text-5xl font-extrabold text-slate-900 leading-tight tracking-tight">
          Cerebral Care AI — Speech & Pain Analysis
        </h1>
        
        <p className="text-lg sm:text-xl text-slate-600 font-normal leading-relaxed">
          Explore speech clarity and facial pain indicators through two AI-assisted analysis tools designed to aid communication and caregiving for individuals with cerebral palsy.
        </p>
      </section>

      {/* Feature Cards Grid */}
      <section aria-label="Core Feature Navigation" className="grid md:grid-cols-2 gap-8 max-w-5xl mx-auto w-full">
        {/* Speech Clarity Card */}
        <div className="accessible-card p-8 flex flex-col justify-between gap-6 border-t-4 border-t-sky-600">
          <div className="flex flex-col gap-4">
            <div className="w-14 h-14 rounded-2xl bg-sky-100 text-sky-700 flex items-center justify-center">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
            </div>
            
            <h2 className="text-2xl font-bold text-slate-900">
              Speech Clarity Assessment
            </h2>
            
            <p className="text-slate-600 text-base leading-relaxed">
              Analyze speech and receive a transcription and clarity assessment using Mel-Frequency Cepstral Coefficients (MFCCs) and K-Nearest Neighbors (KNN) classification.
            </p>
          </div>

          <Link
            href="/speech"
            className="accessible-btn bg-sky-600 hover:bg-sky-700 text-white text-base font-semibold px-6 py-3 rounded-xl shadow-md transition-all flex items-center justify-between group"
          >
            <span>Launch Speech Tool</span>
            <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </Link>
        </div>

        {/* Pain Detection Card */}
        <div className="accessible-card p-8 flex flex-col justify-between gap-6 border-t-4 border-t-teal-600">
          <div className="flex flex-col gap-4">
            <div className="w-14 h-14 rounded-2xl bg-teal-100 text-teal-700 flex items-center justify-center">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            
            <h2 className="text-2xl font-bold text-slate-900">
              Pain Indicator Detection
            </h2>
            
            <p className="text-slate-600 text-base leading-relaxed">
              Analyze a facial image for potential pain indicators leveraging deep neural network feature extraction (InceptionV3) for non-verbal and minimally communicative individuals.
            </p>
          </div>

          <Link
            href="/pain"
            className="accessible-btn bg-teal-600 hover:bg-teal-700 text-white text-base font-semibold px-6 py-3 rounded-xl shadow-md transition-all flex items-center justify-between group"
          >
            <span>Launch Pain Detector</span>
            <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
            </svg>
          </Link>
        </div>
      </section>

    </div>
  );
}
