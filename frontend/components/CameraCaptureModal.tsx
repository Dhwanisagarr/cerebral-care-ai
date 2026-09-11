'use client';

import React, { useRef, useState, useEffect } from 'react';

interface CameraCaptureModalProps {
  isOpen: boolean;
  onClose: () => void;
  onImageCaptured: (file: File) => void;
}

export default function CameraCaptureModal({ isOpen, onClose, onImageCaptured }: CameraCaptureModalProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      startCamera();
    } else {
      stopCamera();
    }
    return () => {
      stopCamera();
    };
  }, [isOpen]);

  const startCamera = async () => {
    setErrorMsg(null);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
      });
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err: any) {
      console.error('Camera access error:', err);
      setErrorMsg('Unable to access camera. Please check camera permissions in your browser.');
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
  };

  const capturePhoto = () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;

    const context = canvas.getContext('2d');
    if (context) {
      context.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob) {
          const capturedFile = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
          onImageCaptured(capturedFile);
          stopCamera();
          onClose();
        }
      }, 'image/jpeg', 0.9);
    }
  };

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/60 backdrop-blur-xs p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="camera-modal-title"
    >
      <div className="bg-white rounded-2xl p-6 max-w-lg w-full shadow-2xl flex flex-col items-center gap-4">
        <div className="w-full flex justify-between items-center border-b border-slate-100 pb-3">
          <h2 id="camera-modal-title" className="text-xl font-bold text-slate-800">
            Take Facial Photo
          </h2>
          <button
            type="button"
            onClick={() => { stopCamera(); onClose(); }}
            aria-label="Close camera modal"
            className="text-slate-400 hover:text-slate-600 p-2 rounded-lg"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {errorMsg ? (
          <div className="bg-red-50 text-red-700 p-4 rounded-xl text-sm w-full border border-red-200" role="alert">
            {errorMsg}
          </div>
        ) : (
          <div className="relative w-full aspect-video bg-black rounded-xl overflow-hidden shadow-inner flex items-center justify-center">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover"
            />
            <canvas ref={canvasRef} className="hidden" />
          </div>
        )}

        <div className="flex justify-end gap-3 w-full pt-2">
          <button
            type="button"
            onClick={() => { stopCamera(); onClose(); }}
            className="accessible-btn bg-slate-100 hover:bg-slate-200 text-slate-700 px-5 py-2.5 rounded-lg font-semibold text-sm"
          >
            Cancel
          </button>
          {!errorMsg && (
            <button
              type="button"
              onClick={capturePhoto}
              className="accessible-btn bg-teal-600 hover:bg-teal-700 text-white px-6 py-2.5 rounded-lg font-semibold text-sm shadow-md"
            >
              Capture Photo
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
