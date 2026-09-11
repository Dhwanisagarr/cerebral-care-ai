import React from 'react';

interface DemoBadgeProps {
  isDemoMode: boolean;
  modelName?: string;
}

export default function DemoBadge({ isDemoMode, modelName }: DemoBadgeProps) {
  if (!isDemoMode) {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-emerald-100 text-emerald-800 border border-emerald-300">
        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" aria-hidden="true" />
        Live Model Active {modelName ? `(${modelName})` : ''}
      </span>
    );
  }

  return (
    <div 
      role="status" 
      aria-label="Demo Mode Indicator"
      className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold bg-amber-100 text-amber-900 border border-amber-300 shadow-2xs"
    >
      <svg className="w-4 h-4 text-amber-700" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
      <span>Demo Mode: Model inference is currently simulated.</span>
    </div>
  );
}
