'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import type { TTSWorkerMessage, TTSWorkerResponse } from '../workers/tts-worker';

// Voice type from kokoro-js
type KokoroVoice =
    | 'af_heart' | 'af_bella' | 'af_nicole' | 'af_sarah' | 'af_sky'
    | 'af_alloy' | 'af_aoede' | 'af_jessica' | 'af_kore' | 'af_nova'
    | 'af_river' | 'af_shimmer'
    | 'am_adam' | 'am_michael' | 'am_echo' | 'am_eric' | 'am_fenrir'
    | 'am_liam' | 'am_onyx' | 'am_puck'
    | 'bf_emma' | 'bf_isabella' | 'bf_alice' | 'bf_lily'
    | 'bm_george' | 'bm_lewis' | 'bm_daniel' | 'bm_fable';

interface UseTTSOptions {
    voice?: KokoroVoice;
    speed?: number;
}

interface UseTTSReturn {
    speak: (text: string) => Promise<void>;
    stop: () => void;
    isSpeaking: boolean;
    isLoading: boolean;
    isGenerating: boolean;
    loadProgress: number;
    currentVoice: KokoroVoice;
    setVoice: (voice: KokoroVoice) => void;
    availableVoices: KokoroVoice[];
    error: string | null;
}

// Available Kokoro voices
const KOKORO_VOICES: KokoroVoice[] = [
    'af_heart',    // American Female - Heart (default)
    'af_bella',    // American Female - Bella
    'af_nicole',   // American Female - Nicole
    'af_sarah',    // American Female - Sarah
    'af_sky',      // American Female - Sky
    'am_adam',     // American Male - Adam
    'am_michael',  // American Male - Michael
    'bf_emma',     // British Female - Emma
    'bf_isabella', // British Female - Isabella
    'bm_george',   // British Male - George
    'bm_lewis',    // British Male - Lewis
];

export function useTTS(options: UseTTSOptions = {}): UseTTSReturn {
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [isGenerating, setIsGenerating] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [loadProgress, setLoadProgress] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [currentVoice, setCurrentVoice] = useState<KokoroVoice>(options.voice || 'af_heart');

    const audioContextRef = useRef<AudioContext | null>(null);
    const workerRef = useRef<Worker | null>(null);

    // Scheduling Refs
    const nextStartTimeRef = useRef(0);
    const scheduledSourcesRef = useRef<AudioBufferSourceNode[]>([]);
    const generationIdRef = useRef(0);
    const isGeneratingRef = useRef(false);

    // Initialize Worker
    const initWorker = useCallback(() => {
        if (!workerRef.current) {
            // Updated to remove .ts extension as per Next.js/Webpack requirements
            workerRef.current = new Worker(new URL('../workers/tts-worker', import.meta.url));
            workerRef.current.onmessage = handleWorkerMessage;
            workerRef.current.postMessage({ type: 'LOAD' } as TTSWorkerMessage);
            setIsLoading(true);
        }
        return workerRef.current;
    }, []);

    // Initialize AudioContext
    const getAudioContext = useCallback(() => {
        if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
            audioContextRef.current = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
        }
        return audioContextRef.current;
    }, []);

    // Handle messages from worker
    const handleWorkerMessage = useCallback((e: MessageEvent<TTSWorkerResponse>) => {
        const { type } = e.data;

        // Check cancellation if genId is present
        if ('genId' in e.data && e.data.genId !== generationIdRef.current) {
            console.debug('TTS: Ignoring chunk from cancelled generation', e.data.genId);
            return;
        }

        switch (type) {
            case 'LOAD_PROGRESS':
                if ('progress' in e.data) setLoadProgress(e.data.progress);
                break;
            case 'LOAD_COMPLETE':
                setIsLoading(false);
                setLoadProgress(100);
                console.log('TTS: Worker loaded model');
                break;
            case 'ERROR':
                if ('error' in e.data) {
                    setError(e.data.error);
                    setIsLoading(false);
                    setIsGenerating(false);
                    isGeneratingRef.current = false;
                    console.error('TTS Worker Error:', e.data.error);
                }
                break;
            case 'AUDIO_CHUNK':
                if ('audio' in e.data && e.data.audio.length > 0) {
                    const ctx = getAudioContext();

                    if (ctx.state === 'suspended') {
                        ctx.resume().catch(console.error);
                    }

                    const audioBuffer = ctx.createBuffer(1, e.data.audio.length, e.data.samplingRate);
                    audioBuffer.copyToChannel(e.data.audio as Float32Array, 0);

                    const source = ctx.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(ctx.destination);

                    const speed = options.speed || 1.0;
                    source.playbackRate.value = speed;

                    // UPDATED SCHEDULING LOGIC
                    const currentTime = ctx.currentTime;
                    let scheduleTime = nextStartTimeRef.current;
                    const isNewSentence = e.data.isNewSentence !== false;

                    // 1. Calculate Target Scheduling Time
                    // Ideally, we want exact stitching or precise gap.
                    // But if we are LATE (underrun), we must play ASAP.

                    // Target Gap
                    const requestedGap = isNewSentence ? 0.100 : 0.015;

                    // Ideal Start Time (Previous End + Gap)
                    let idealStartTime = scheduleTime + requestedGap;

                    // 2. Check for Underrun
                    if (idealStartTime < currentTime) {
                        // WE ARE LATE. 
                        // The previous chunk finished playing BEFORE we generated this one.
                        // We have an UNAVOIDABLE silence gap equal to (currentTime - scheduleTime).
                        // We should NOT add extra delay. Play immediately with tiny safety buffer.
                        scheduleTime = currentTime + 0.01; // 10ms safety to avoid glitch

                        // Note: The "Applied Gap" log will effectively show the huge delay of generation.
                    } else {
                        // WE ARE ON TIME (or Early).
                        // Stick to the ideal schedule to enforce the precise gap.
                        scheduleTime = idealStartTime;
                    }

                    // Log for verification
                    if (nextStartTimeRef.current > 0) {
                        const gapFromPrevEnd = (scheduleTime - nextStartTimeRef.current) * 1000;
                        console.debug(`TTS [Chunk ${e.data.sentenceIndex}]: isNewSentence=${isNewSentence}, Applied Gap=${gapFromPrevEnd.toFixed(1)}ms`);
                    }

                    source.start(scheduleTime);

                    // Final endpoint of THIS chunk
                    nextStartTimeRef.current = scheduleTime + (audioBuffer.duration / speed);

                    scheduledSourcesRef.current.push(source);

                    source.onended = () => {
                        const idx = scheduledSourcesRef.current.indexOf(source);
                        if (idx > -1) scheduledSourcesRef.current.splice(idx, 1);
                        if (!isGeneratingRef.current && scheduledSourcesRef.current.length === 0) {
                            setIsSpeaking(false);
                            nextStartTimeRef.current = 0;
                        }
                    };

                    if (!isSpeaking) setIsSpeaking(true);
                }
                break;
            case 'PLAYBACK_COMPLETE':
                setIsGenerating(false);
                isGeneratingRef.current = false;
                break;
        }
    }, [getAudioContext, isSpeaking, options.speed]);

    // Stop playback
    const stop = useCallback(() => {
        generationIdRef.current++;
        scheduledSourcesRef.current.forEach(source => {
            try {
                source.stop();
                source.disconnect();
            } catch (e) { /* ignore */ }
        });
        scheduledSourcesRef.current = [];
        nextStartTimeRef.current = 0;
        if (workerRef.current) {
            workerRef.current.postMessage({ type: 'STOP' } as TTSWorkerMessage);
        }
        setIsSpeaking(false);
        setIsGenerating(false);
        isGeneratingRef.current = false;
    }, []);

    const speak = useCallback(async (text: string) => {
        if (!text.trim()) return;

        stop();
        generationIdRef.current++;
        const myGenId = generationIdRef.current;

        setError(null);
        setIsGenerating(true);
        isGeneratingRef.current = true;
        nextStartTimeRef.current = 0;

        const worker = initWorker();
        const ctx = getAudioContext();
        if (ctx.state === 'suspended') {
            await ctx.resume();
        }

        worker.postMessage({
            type: 'SPEAK',
            text,
            voice: currentVoice,
            speed: options.speed || 1,
            genId: myGenId
        } as TTSWorkerMessage);

    }, [initWorker, currentVoice, options.speed, stop, getAudioContext]);

    // Cleanup
    useEffect(() => {
        initWorker();
        return () => {
            if (workerRef.current) {
                workerRef.current.terminate();
                workerRef.current = null;
            }
            if (audioContextRef.current) {
                audioContextRef.current.close();
            }
        };
    }, []);

    useEffect(() => {
        if (workerRef.current) {
            workerRef.current.onmessage = handleWorkerMessage;
        }
    }, [handleWorkerMessage]);

    return {
        speak,
        stop,
        isSpeaking,
        isGenerating,
        isLoading,
        loadProgress,
        currentVoice,
        setVoice: setCurrentVoice,
        availableVoices: KOKORO_VOICES,
        error,
    };
}
