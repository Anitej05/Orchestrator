'use client';

import React from 'react';

interface AudioWaveAnimationProps {
    isActive: boolean;
    audioLevel?: number; // 0-100
    barCount?: number;
    className?: string;
    color?: string;
}

export function AudioWaveAnimation({
    isActive,
    audioLevel = 50,
    barCount = 5,
    className = '',
    color = 'currentColor',
}: AudioWaveAnimationProps) {
    // Generate bars with varying heights based on audio level
    const bars = Array.from({ length: barCount }, (_, i) => {
        // Create a wave pattern - middle bars are taller
        const centerOffset = Math.abs(i - (barCount - 1) / 2);
        const baseHeight = 1 - (centerOffset / ((barCount - 1) / 2)) * 0.5;

        // Apply audio level to height
        const audioMultiplier = isActive ? (0.3 + (audioLevel / 100) * 0.7) : 0.2;
        const height = baseHeight * audioMultiplier;

        // Stagger animation delays
        const delay = i * 0.1;

        return { height, delay };
    });

    return (
        <div
            className={`flex items-center justify-center gap-1 ${className}`}
            role="img"
            aria-label={isActive ? "Recording audio" : "Audio inactive"}
        >
            {bars.map((bar, index) => (
                <div
                    key={index}
                    className={`rounded-full transition-all ${isActive ? 'animate-pulse' : ''}`}
                    style={{
                        width: '4px',
                        height: `${Math.max(8, bar.height * 32)}px`,
                        backgroundColor: color,
                        animationDelay: `${bar.delay}s`,
                        animationDuration: '0.6s',
                        transform: isActive ? 'scaleY(1)' : 'scaleY(0.3)',
                        transformOrigin: 'center',
                    }}
                />
            ))}

            <style jsx>{`
        @keyframes audioWave {
          0%, 100% {
            transform: scaleY(0.5);
          }
          50% {
            transform: scaleY(1);
          }
        }
        
        div[class*="animate-pulse"] {
          animation: audioWave 0.6s ease-in-out infinite;
        }
      `}</style>
        </div>
    );
}

// More sophisticated wave animation with SVG
export function AudioWaveSVG({
    isActive,
    audioLevel = 50,
    width = 120,
    height = 40,
    className = '',
    color = '#3b82f6',
}: {
    isActive: boolean;
    audioLevel?: number;
    width?: number;
    height?: number;
    className?: string;
    color?: string;
}) {
    const barCount = 7;
    const barWidth = 6;
    const gap = (width - barCount * barWidth) / (barCount + 1);

    return (
        <svg
            width={width}
            height={height}
            viewBox={`0 0 ${width} ${height}`}
            className={className}
            role="img"
            aria-label={isActive ? "Recording audio" : "Audio inactive"}
        >
            {Array.from({ length: barCount }, (_, i) => {
                // Create wave pattern
                const centerOffset = Math.abs(i - (barCount - 1) / 2);
                const baseHeight = 1 - (centerOffset / ((barCount - 1) / 2)) * 0.4;

                // Apply audio level
                const levelMultiplier = isActive ? (0.4 + (audioLevel / 100) * 0.6) : 0.2;
                const barHeight = baseHeight * levelMultiplier * height * 0.8;

                const x = gap + i * (barWidth + gap);
                const y = (height - barHeight) / 2;

                return (
                    <rect
                        key={i}
                        x={x}
                        y={y}
                        width={barWidth}
                        height={barHeight}
                        rx={barWidth / 2}
                        fill={color}
                        opacity={isActive ? 1 : 0.5}
                        style={{
                            transformOrigin: `${x + barWidth / 2}px ${height / 2}px`,
                            animation: isActive
                                ? `audioBar 0.5s ease-in-out infinite alternate`
                                : 'none',
                            animationDelay: `${i * 0.08}s`,
                        }}
                    />
                );
            })}

            <style>{`
        @keyframes audioBar {
          0% {
            transform: scaleY(0.6);
          }
          100% {
            transform: scaleY(1);
          }
        }
      `}</style>
        </svg>
    );
}

// Compact inline wave for showing in input area
export function AudioWaveInline({
    isActive,
    audioLevel = 50,
    className = '',
}: {
    isActive: boolean;
    audioLevel?: number;
    className?: string;
}) {
    return (
        <div className={`inline-flex items-center gap-0.5 h-4 ${className}`}>
            {[0, 1, 2, 3, 4].map((i) => {
                const centerOffset = Math.abs(i - 2);
                const baseHeight = 1 - centerOffset * 0.2;
                const levelMultiplier = isActive ? (0.5 + (audioLevel / 100) * 0.5) : 0.3;

                return (
                    <div
                        key={i}
                        className={`w-0.5 rounded-full bg-blue-500 dark:bg-blue-400 ${isActive ? 'animate-[audioWaveSmall_0.5s_ease-in-out_infinite_alternate]' : ''
                            }`}
                        style={{
                            height: `${baseHeight * levelMultiplier * 16}px`,
                            minHeight: '4px',
                            animationDelay: `${i * 0.1}s`,
                        }}
                    />
                );
            })}

            <style jsx global>{`
        @keyframes audioWaveSmall {
          0% {
            transform: scaleY(0.6);
          }
          100% {
            transform: scaleY(1);
          }
        }
      `}</style>
        </div>
    );
}
