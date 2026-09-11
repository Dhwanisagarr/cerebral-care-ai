'use client';

import React, { useState, useRef, useEffect } from 'react';
import { audioBufferToWav } from '@/lib/wavEncoder';

interface AudioRecorderProps {
  onAudioRecorded: (file: File) => void;
  disabled?: boolean;
}

export default function AudioRecorder({ onAudioRecorded, disabled = false }: AudioRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startRecording = async () => {
    setErrorMsg(null);
    setAudioUrl(null);
    audioChunksRef.current = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        try {
          const rawBlob = new Blob(audioChunksRef.current);
          const arrayBuffer = await rawBlob.arrayBuffer();

          // Use Web Audio API to decode recorded bytes to PCM AudioBuffer
          const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
          const decodedBuffer = await audioCtx.decodeAudioData(arrayBuffer);

          // Convert decoded AudioBuffer into a true 16-bit PCM WAV Blob
          const wavBlob = audioBufferToWav(decodedBuffer);
          const url = URL.createObjectURL(wavBlob);
          setAudioUrl(url);

          const recordedWavFile = new File([wavBlob], 'recorded_speech.wav', { type: 'audio/wav' });
          onAudioRecorded(recordedWavFile);
          
          audioCtx.close();
        } catch (err: any) {
          console.error('Audio processing/decoding error:', err);
          setErrorMsg('Failed to process recorded audio. Please try again.');
        }

        // Stop all audio track streams
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start(100);
      setIsRecording(true);
      setRecordingTime(0);

      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      console.error('Microphone permission or access error:', err);
      setErrorMsg('Microphone access denied or unavailable. Please check browser permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-slate-50 border border-slate-200 rounded-xl p-5 text-center flex flex-col items-center gap-4">
      <div className="flex items-center gap-3 flex-wrap justify-center">
        {!isRecording ? (
          <button
            type="button"
            onClick={startRecording}
            disabled={disabled}
            aria-label="Start audio recording"
            className="accessible-btn bg-sky-600 hover:bg-sky-700 text-white px-6 py-3 rounded-lg shadow-md font-semibold text-base flex items-center gap-2 disabled:opacity-50"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
            Record Audio
          </button>
        ) : (
          <button
            type="button"
            onClick={stopRecording}
            aria-label="Stop audio recording"
            className="accessible-btn bg-red-600 hover:bg-red-700 text-white px-6 py-3 rounded-lg shadow-md font-semibold text-base flex items-center gap-2 animate-pulse"
          >
            <span className="w-3 h-3 rounded-full bg-white animate-ping" />
            Stop Recording ({formatTime(recordingTime)})
          </button>
        )}
      </div>

      {isRecording && (
        <p className="text-sm font-medium text-red-600 flex items-center gap-2" role="status" aria-live="polite">
          <span className="w-2.5 h-2.5 rounded-full bg-red-600 animate-ping" />
          Recording in progress... speak clearly into your microphone.
        </p>
      )}

      {errorMsg && (
        <p className="text-sm text-red-600 bg-red-50 p-3 rounded-lg border border-red-200" role="alert">
          {errorMsg}
        </p>
      )}

      {audioUrl && !isRecording && (
        <div className="w-full max-w-md mt-2 flex flex-col gap-2">
          <label className="text-xs font-semibold text-slate-600 block text-left">
            Recording Playback Preview (Converted PCM WAV):
          </label>
          <audio controls src={audioUrl} className="w-full rounded-lg shadow-2xs" />
        </div>
      )}
    </div>
  );
}
