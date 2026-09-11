import React from 'react';

export default function Footer() {
  return (
    <footer className="bg-white border-t border-slate-200 mt-20 py-10 text-slate-600">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 text-center sm:text-left flex flex-col sm:flex-row items-center justify-between gap-4">
        <div>
          <p className="font-semibold text-slate-800">
            Enhancing Speech Clarity and Pain Detection in Individuals with Cerebral Palsy
          </p>
          <p className="text-xs text-slate-500 mt-1">
            Assistive AI platform for speech clarity assessment and facial expression analysis.
          </p>
        </div>
        <div className="text-xs text-slate-400">
          Built with Next.js, TypeScript, FastAPI & Librosa
        </div>
      </div>
    </footer>
  );
}
