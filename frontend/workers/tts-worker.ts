
import { preprocessTextForTTS } from '../lib/tts-preprocessing';

// Defines the interface for messages sent to the worker
export type TTSWorkerMessage =
    | { type: 'LOAD' }
    | { type: 'SPEAK'; text: string; voice: string; speed: number; genId: number }
    | { type: 'STOP' };

// Defines the interface for messages sent from the worker
export type TTSWorkerResponse =
    | { type: 'LOAD_PROGRESS'; progress: number }
    | { type: 'LOAD_COMPLETE' }
    | { type: 'ERROR'; error: string; genId?: number }
    | { type: 'AUDIO_CHUNK'; audio: Float32Array; samplingRate: number; sentenceIndex: number; totalSentences?: number; genId: number; isNewSentence?: boolean }
    | { type: 'PLAYBACK_COMPLETE'; genId: number };

// Global TTS instance
let ttsInstance: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let ttsLoadingPromise: Promise<any> | null = null;
let currentGenerationId = 0;

// Type assertion for Worker context
const ctx: Worker = self as any;

// Helper: Trim silence from beginning and end of Float32Array
function trimSilence(audio: Float32Array, threshold = 0.01): Float32Array {
    let start = 0;
    while (start < audio.length && Math.abs(audio[start]) < threshold) {
        start++;
    }

    // If all silence
    if (start >= audio.length) return new Float32Array(0);

    let end = audio.length - 1;
    while (end > start && Math.abs(audio[end]) < threshold) {
        end--;
    }

    return audio.slice(start, end + 1);
}

// Load the TTS model
async function loadTTS() {
    if (ttsInstance) return ttsInstance;
    if (ttsLoadingPromise) return ttsLoadingPromise;

    ttsLoadingPromise = (async () => {
        try {
            ctx.postMessage({ type: 'LOAD_PROGRESS', progress: 0 });

            // Import kokoro-js
            const { KokoroTTS } = await import('kokoro-js');

            // Configure thread limit to avoid blocking main UI thread
            try {
                // @ts-ignore
                const ort = await import('onnxruntime-web');
                if (ort?.env) {
                    ort.env.logLevel = 'error';
                    // Use all available cores for maximum speed per user request
                    // @ts-ignore
                    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
                    // @ts-ignore
                    ort.env.wasm.simd = true;
                }
            } catch (e) { /* ignore */ }

            ctx.postMessage({ type: 'LOAD_PROGRESS', progress: 20 });

            // Console suppression for ONNX
            const originalConsole = {
                log: console.log,
                warn: console.warn,
                error: console.error
            };
            const suppressONNX = (method: 'log' | 'warn' | 'error', args: unknown[]) => {
                const msg = args[0];
                if (typeof msg === 'string' && (msg.includes('onnxruntime') || msg.includes('VerifyEachNodeIsAssignedToAnEp') || msg.includes('session_state'))) return;
                originalConsole[method].apply(console, args);
            };
            console.log = (...args) => suppressONNX('log', args);
            console.warn = (...args) => suppressONNX('warn', args);
            console.error = (...args) => suppressONNX('error', args);

            ctx.postMessage({ type: 'LOAD_PROGRESS', progress: 30 });

            try {
                // Switching to q4 (faster, smaller) + WASM (stable audio)
                console.log('TTS Worker: Loading Kokoro-82M-ONNX model (q4, wasm backend) with max threads...');

                const tts = await KokoroTTS.from_pretrained('onnx-community/Kokoro-82M-ONNX', {
                    dtype: 'q4', // Faster loading/inference
                    device: 'wasm', // Stable audio
                });
                ttsInstance = tts;
                ctx.postMessage({ type: 'LOAD_PROGRESS', progress: 100 });
                ctx.postMessage({ type: 'LOAD_COMPLETE' });
                return tts;
            } finally {
                console.log = originalConsole.log;
                console.warn = originalConsole.warn;
                console.error = originalConsole.error;
            }
        } catch (err) {
            const msg = err instanceof Error ? err.message : 'Unknown error';
            ctx.postMessage({ type: 'ERROR', error: msg });
            ttsLoadingPromise = null;
            throw err;
        }
    })();

    return ttsLoadingPromise;
}

// Wrapper interface for sentence chunks
interface SentenceChunk {
    text: string;
    isNewSentence: boolean;
}

// Speak logic with generation ID support
async function speak(text: string, voice: string, genId: number) {
    if (genId !== currentGenerationId) return;

    try {
        const tts = await loadTTS();
        if (genId !== currentGenerationId) return;

        console.log('TTS Worker: Preprocessing text:', text);
        const processedText = preprocessTextForTTS(text);

        // Split text by sentence (.!?) NEWLINE only
        const rawSentences = processedText.match(/[^.!?\n]+[.!?\n]+|[^.!?\n]+$/g) || [processedText];

        // Prepare queue of chunks with flags
        const queue: SentenceChunk[] = [];

        // Stitch Logic
        const stitchSplit = (text: string): string[] => {
            const result: string[] = [];
            let remaining = text.trim();

            while (remaining.length > 0) {
                if (remaining.length < 75) {
                    result.push(remaining);
                    break;
                }

                // Check punctuation window 50-125
                const minSplit = 50;
                const maxSplit = Math.min(125, remaining.length);
                const punctuationRegex = /[,;:\u2014-]/g;
                let splitIndex = -1;
                let match;

                // Priority 1: Punctuation
                if (remaining.length > 100) {
                    const searchFrom = 75;
                    const searchLimit = 125;
                    while ((match = punctuationRegex.exec(remaining)) !== null) {
                        if (match.index >= searchFrom && match.index <= searchLimit) {
                            splitIndex = match.index;
                            break;
                        }
                    }

                    // Priority 2: Force Split after 75
                    if (splitIndex === -1) {
                        const spaceIndex = remaining.indexOf(' ', 75);
                        if (spaceIndex !== -1 && spaceIndex < 150) {
                            splitIndex = spaceIndex;
                        } else {
                            splitIndex = (remaining.length > 100) ? 100 : remaining.length;
                        }
                    }
                } else {
                    result.push(remaining);
                    break;
                }

                if (splitIndex !== -1 && splitIndex < remaining.length) {
                    const chunk = remaining.substring(0, splitIndex + 1).trim();
                    result.push(chunk);
                    remaining = remaining.substring(splitIndex + 1).trim();
                } else {
                    result.push(remaining);
                    break;
                }
            }
            return result;
        };

        for (const s of rawSentences) {
            const parts = stitchSplit(s);
            if (parts.length > 0) {
                // First part is NEW sentence start
                queue.push({ text: parts[0], isNewSentence: true });
                // Subsequent parts are continuations
                for (let i = 1; i < parts.length; i++) {
                    queue.push({ text: parts[i], isNewSentence: false });
                }
            }
        }

        for (let i = 0; i < queue.length; i++) {
            const item = queue[i];
            if (!item.text.trim()) continue;

            // Check cancellation before generation
            if (genId !== currentGenerationId) break;

            // Yield to event loop to allow STOP messages
            await new Promise(r => setTimeout(r, 0));

            // Generate Audio
            const result = await tts.generate(item.text, { voice });

            // Check cancellation after generation
            if (genId !== currentGenerationId) break;

            if (result.audio && result.audio.length > 0) {
                const trimmedAudio = trimSilence(result.audio);
                if (trimmedAudio.length === 0) continue;

                ctx.postMessage({
                    type: 'AUDIO_CHUNK',
                    audio: trimmedAudio,
                    samplingRate: result.sampling_rate || 24000,
                    sentenceIndex: i,
                    totalSentences: queue.length,
                    genId: genId,
                    isNewSentence: item.isNewSentence // Pass Flag
                } as TTSWorkerResponse);
            }
        }

        if (genId === currentGenerationId) {
            ctx.postMessage({ type: 'PLAYBACK_COMPLETE', genId });
        }
    } catch (err) {
        console.error('TTS Worker Error:', err);
        const msg = err instanceof Error ? err.message : 'Synthesis error';
        ctx.postMessage({ type: 'ERROR', error: msg, genId });
    }
}

// Message handler
ctx.onmessage = (e: MessageEvent<TTSWorkerMessage>) => {
    const { type } = e.data;

    switch (type) {
        case 'LOAD':
            loadTTS();
            break;
        case 'SPEAK':
            if ('text' in e.data && 'genId' in e.data) {
                console.log('TTS Worker: Received SPEAK request (GenID:', e.data.genId + ')');
                currentGenerationId = e.data.genId;
                speak(e.data.text, e.data.voice, e.data.genId);
            }
            break;
        case 'STOP':
            console.log('TTS Worker: STOP received');
            currentGenerationId++; // Invalidate current ID
            break;
    }
};
