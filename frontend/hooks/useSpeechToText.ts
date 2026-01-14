'use client';

import { useState, useRef, useCallback, useEffect } from 'react';

interface UseSpeechToTextReturn {
    startListening: () => void;
    stopListening: () => void;
    isListening: boolean;
    transcript: string;
    interimTranscript: string;
    audioLevel: number; // 0-100 for animation
    isSupported: boolean;
    error: string | null;
    resetTranscript: () => void;
}

interface UseSpeechToTextOptions {
    continuous?: boolean;
    interimResults?: boolean;
    language?: string;
    onResult?: (transcript: string) => void;
    onError?: (error: string) => void;
}

// SpeechRecognition types for browsers
interface ISpeechRecognition extends EventTarget {
    continuous: boolean;
    interimResults: boolean;
    lang: string;
    start: () => void;
    stop: () => void;
    abort: () => void;
    onstart: (() => void) | null;
    onend: (() => void) | null;
    onresult: ((event: ISpeechRecognitionEvent) => void) | null;
    onerror: ((event: ISpeechRecognitionErrorEvent) => void) | null;
}

interface ISpeechRecognitionEvent {
    resultIndex: number;
    results: ISpeechRecognitionResultList;
}

interface ISpeechRecognitionResultList {
    length: number;
    item: (index: number) => ISpeechRecognitionResult;
    [index: number]: ISpeechRecognitionResult;
}

interface ISpeechRecognitionResult {
    isFinal: boolean;
    length: number;
    item: (index: number) => ISpeechRecognitionAlternative;
    [index: number]: ISpeechRecognitionAlternative;
}

interface ISpeechRecognitionAlternative {
    transcript: string;
    confidence: number;
}

interface ISpeechRecognitionErrorEvent {
    error: string;
    message?: string;
}

// Extend Window interface for SpeechRecognition
declare global {
    interface Window {
        SpeechRecognition?: new () => ISpeechRecognition;
        webkitSpeechRecognition?: new () => ISpeechRecognition;
    }
}

export function useSpeechToText(options: UseSpeechToTextOptions = {}): UseSpeechToTextReturn {
    const {
        continuous = true,
        interimResults = true,
        language = 'en-US',
        onResult,
        onError,
    } = options;

    const [isListening, setIsListening] = useState(false);
    const [transcript, setTranscript] = useState('');
    const [interimTranscript, setInterimTranscript] = useState('');
    const [audioLevel, setAudioLevel] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [isSupported, setIsSupported] = useState(false);

    const recognitionRef = useRef<ISpeechRecognition | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);
    const analyserRef = useRef<AnalyserNode | null>(null);
    const mediaStreamRef = useRef<MediaStream | null>(null);
    const animationFrameRef = useRef<number | null>(null);

    // Check browser support
    useEffect(() => {
        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
        setIsSupported(!!SpeechRecognitionAPI);
    }, []);

    // Audio level analysis for visualization
    const startAudioAnalysis = useCallback(async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaStreamRef.current = stream;

            audioContextRef.current = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
            analyserRef.current = audioContextRef.current.createAnalyser();
            analyserRef.current.fftSize = 256;

            const source = audioContextRef.current.createMediaStreamSource(stream);
            source.connect(analyserRef.current);

            const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);

            const updateLevel = () => {
                if (!analyserRef.current) return;

                analyserRef.current.getByteFrequencyData(dataArray);

                // Calculate average volume
                const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length;
                const normalizedLevel = Math.min(100, (average / 128) * 100);

                setAudioLevel(normalizedLevel);
                animationFrameRef.current = requestAnimationFrame(updateLevel);
            };

            updateLevel();
        } catch (err) {
            console.error('Failed to start audio analysis:', err);
        }
    }, []);

    const stopAudioAnalysis = useCallback(() => {
        if (animationFrameRef.current) {
            cancelAnimationFrame(animationFrameRef.current);
            animationFrameRef.current = null;
        }
        if (mediaStreamRef.current) {
            mediaStreamRef.current.getTracks().forEach(track => track.stop());
            mediaStreamRef.current = null;
        }
        if (audioContextRef.current) {
            audioContextRef.current.close();
            audioContextRef.current = null;
        }
        analyserRef.current = null;
        setAudioLevel(0);
    }, []);

    // Start listening
    const startListening = useCallback(() => {
        const SpeechRecognitionAPI = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognitionAPI) {
            const errorMsg = 'Speech recognition is not supported in this browser';
            setError(errorMsg);
            onError?.(errorMsg);
            return;
        }

        setError(null);
        setTranscript('');
        setInterimTranscript('');

        const recognition = new SpeechRecognitionAPI();
        recognition.continuous = continuous;
        recognition.interimResults = interimResults;
        recognition.lang = language;

        recognition.onstart = () => {
            setIsListening(true);
            startAudioAnalysis();
        };

        recognition.onresult = (event: ISpeechRecognitionEvent) => {
            let finalTranscript = '';
            let interim = '';

            for (let i = event.resultIndex; i < event.results.length; i++) {
                const result = event.results[i];
                if (result.isFinal) {
                    finalTranscript += result[0].transcript;
                } else {
                    interim += result[0].transcript;
                }
            }

            if (finalTranscript) {
                setTranscript(prev => prev + finalTranscript);
                onResult?.(finalTranscript);
            }
            setInterimTranscript(interim);
        };

        recognition.onerror = (event: ISpeechRecognitionErrorEvent) => {
            const errorMsg = `Speech recognition error: ${event.error}`;
            setError(errorMsg);
            onError?.(errorMsg);
            setIsListening(false);
            stopAudioAnalysis();
        };

        recognition.onend = () => {
            setIsListening(false);
            stopAudioAnalysis();

            // Auto restart if continuous mode and not manually stopped
            if (continuous && recognitionRef.current === recognition) {
                try {
                    recognition.start();
                } catch (e) {
                    // Already started or stopped
                }
            }
        };

        recognitionRef.current = recognition;

        try {
            recognition.start();
        } catch (err) {
            const errorMsg = 'Failed to start speech recognition';
            setError(errorMsg);
            onError?.(errorMsg);
        }
    }, [continuous, interimResults, language, onResult, onError, startAudioAnalysis, stopAudioAnalysis]);

    // Stop listening
    const stopListening = useCallback(() => {
        if (recognitionRef.current) {
            recognitionRef.current.stop();
            recognitionRef.current = null;
        }
        setIsListening(false);
        stopAudioAnalysis();
    }, [stopAudioAnalysis]);

    // Reset transcript
    const resetTranscript = useCallback(() => {
        setTranscript('');
        setInterimTranscript('');
    }, []);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (recognitionRef.current) {
                recognitionRef.current.stop();
            }
            stopAudioAnalysis();
        };
    }, [stopAudioAnalysis]);

    return {
        startListening,
        stopListening,
        isListening,
        transcript,
        interimTranscript,
        audioLevel,
        isSupported,
        error,
        resetTranscript,
    };
}
