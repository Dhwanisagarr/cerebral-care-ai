import React from 'react';

export default function MedicalDisclaimerBanner() {
  return (
    <div 
      role="region" 
      aria-label="Medical Disclaimer"
      className="bg-amber-50 border-b border-amber-200 text-amber-900 px-4 py-3 text-sm text-center font-medium shadow-xs"
    >
      <div className="max-w-6xl mx-auto flex items-center justify-center gap-2 flex-wrap">
        <svg 
          className="w-5 h-5 text-amber-700 flex-shrink-0" 
          fill="none" 
          stroke="currentColor" 
          viewBox="0 0 24 24" 
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
        </svg>
        <span>
          <strong>Research & Decision Support Purpose Only:</strong> This application is an assistive AI decision-support tool. It is <em>not</em> a medical diagnostic device and does not offer clinical medical diagnoses.
        </span>
      </div>
    </div>
  );
}
