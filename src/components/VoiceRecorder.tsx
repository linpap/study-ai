'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

interface VoiceRecorderProps {
  onTranscript: (text: string) => void;
  disabled?: boolean;
  maxDuration?: number; // maximum recording duration in seconds
}

export default function VoiceRecorder({ onTranscript, disabled, maxDuration = 30 }: VoiceRecorderProps) {
  const [isListening, setIsListening] = useState(false);
  const [isSupported, setIsSupported] = useState(true);
  const [elapsedTime, setElapsedTime] = useState(0);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const shouldRestartRef = useRef(false);
  const startTimeRef = useRef<number | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const restartTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const stopRecording = useCallback(() => {
    shouldRestartRef.current = false;
    if (restartTimeoutRef.current) {
      clearTimeout(restartTimeoutRef.current);
      restartTimeoutRef.current = null;
    }
    if (recognitionRef.current) {
      try {
        recognitionRef.current.stop();
      } catch {
        // Ignore stop errors
      }
    }
    setIsListening(false);
    setElapsedTime(0);
    startTimeRef.current = null;
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional feature detection on mount
        setIsSupported(false);
        return;
      }

      const recognition = new SpeechRecognition();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = 'en-US';
      recognition.maxAlternatives = 1;

      recognition.onresult = (event) => {
        let finalTranscript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript;
          }
        }
        if (finalTranscript) {
          onTranscript(finalTranscript);
        }
      };

      recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        // Don't stop on these errors, they're recoverable
        if (event.error === 'no-speech' || event.error === 'aborted' || event.error === 'network') {
          // Will auto-restart via onend handler
          return;
        }
        // For other errors, stop recording
        stopRecording();
      };

      recognition.onend = () => {
        // Auto-restart if we should still be listening
        if (shouldRestartRef.current) {
          // Small delay before restarting to avoid rapid start/stop cycles
          restartTimeoutRef.current = setTimeout(() => {
            if (shouldRestartRef.current && recognitionRef.current) {
              try {
                recognitionRef.current.start();
              } catch {
                // If start fails, try again after a longer delay
                restartTimeoutRef.current = setTimeout(() => {
                  if (shouldRestartRef.current && recognitionRef.current) {
                    try {
                      recognitionRef.current.start();
                    } catch {
                      stopRecording();
                    }
                  }
                }, 500);
              }
            }
          }, 100);
        } else {
          setIsListening(false);
        }
      };

      recognitionRef.current = recognition;
    }

    return () => {
      shouldRestartRef.current = false;
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {
          // Ignore
        }
      }
    };
  }, [onTranscript, stopRecording]);

  const startRecording = useCallback(() => {
    if (!recognitionRef.current) return;

    shouldRestartRef.current = true;
    startTimeRef.current = Date.now();
    setElapsedTime(0);

    // Start timer to track elapsed time and auto-stop at max duration
    timerRef.current = setInterval(() => {
      if (startTimeRef.current) {
        const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
        setElapsedTime(elapsed);

        // Auto-stop after max duration
        if (elapsed >= maxDuration) {
          stopRecording();
        }
      }
    }, 1000);

    try {
      recognitionRef.current.start();
      setIsListening(true);
    } catch (e) {
      console.error('Failed to start recognition:', e);
      stopRecording();
    }
  }, [maxDuration, stopRecording]);

  const toggleListening = () => {
    if (isListening) {
      // Allow stopping anytime
      stopRecording();
    } else {
      startRecording();
    }
  };

  if (!isSupported) {
    return (
      <p className="text-sm text-gray-500">
        Voice input not supported in this browser. Try Chrome or Edge.
      </p>
    );
  }

  const timeLeft = maxDuration - elapsedTime;

  return (
    <button
      onClick={toggleListening}
      disabled={disabled}
      className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
        isListening
          ? 'bg-red-500 text-white animate-pulse'
          : 'bg-gray-100 hover:bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      title={isListening ? 'Click to stop recording' : 'Click to start voice input'}
    >
      {isListening ? (
        <>
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
          Stop ({timeLeft}s)
        </>
      ) : (
        <>
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z"/>
            <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
          </svg>
          Voice Input
        </>
      )}
    </button>
  );
}
