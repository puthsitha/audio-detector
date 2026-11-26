import React, { useEffect, useRef, useState } from 'react';
import { Play, Pause, Volume2 } from 'lucide-react';
import { formatTime } from '../api';

interface WaveformPlayerProps {
  src: string;
  accentColor?: string;
}

const BAR_COUNT = 40;

export const WaveformPlayer: React.FC<WaveformPlayerProps> = ({
  src,
  accentColor = '#818cf8',
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [peaks, setPeaks] = useState<number[]>(() =>
    Array.from({ length: BAR_COUNT }, (_, i) => 0.22 + 0.45 * Math.abs(Math.sin(i * 0.45))),
  );

  // Decode waveform peaks locally via Web Audio API for crisp scrubber visualization
  useEffect(() => {
    let cancelled = false;
    setIsPlaying(false);
    setCurrentTime(0);

    async function computePeaks() {
      try {
        const res = await fetch(src);
        if (!res.ok) return;
        const arrayBuffer = await res.arrayBuffer();
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        if (!AudioCtx) return;
        const ctx = new AudioCtx();
        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        const channel = audioBuffer.getChannelData(0);
        const blockSize = Math.max(1, Math.floor(channel.length / BAR_COUNT));
        const computed: number[] = [];
        let maxPeak = 0.001;

        for (let i = 0; i < BAR_COUNT; i++) {
          const start = i * blockSize;
          let sumSq = 0;
          const end = Math.min(channel.length, start + blockSize);
          for (let j = start; j < end; j++) {
            sumSq += channel[j] * channel[j];
          }
          const rms = Math.sqrt(sumSq / Math.max(1, end - start));
          computed.push(rms);
          if (rms > maxPeak) maxPeak = rms;
        }
        await ctx.close();
        if (!cancelled) {
          setPeaks(computed.map((v) => Math.max(0.12, Math.min(1, v / maxPeak))));
          setDuration(audioBuffer.duration);
        }
      } catch {
        // Gracefully keep decorative fallback bars if decode fails
      }
    }

    computePeaks();
    return () => {
      cancelled = true;
    };
  }, [src]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (isPlaying) {
      audio.pause();
    } else {
      audio.play().catch(() => {
        setIsPlaying(false);
      });
    }
  };

  const handleScrub = (event: React.MouseEvent<HTMLDivElement>) => {
    const audio = audioRef.current;
    if (!audio || !duration) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
    audio.currentTime = ratio * duration;
    setCurrentTime(audio.currentTime);
  };

  const progress = duration > 0 ? Math.min(1, currentTime / duration) : 0;

  return (
    <div className="rounded-xl border border-slate-800/90 bg-slate-950/80 p-3">
      <audio
        ref={audioRef}
        src={src}
        preload="metadata"
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => {
          setIsPlaying(false);
          setCurrentTime(0);
        }}
        onLoadedMetadata={(e) => {
          const d = e.currentTarget.duration;
          if (Number.isFinite(d) && d > 0) setDuration(d);
        }}
        onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
      />

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={togglePlay}
          aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-indigo-600 text-white shadow-md shadow-indigo-950/60 transition hover:bg-indigo-500 active:scale-95"
        >
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
        </button>

        {/* Interactive SVG Waveform Scrubber */}
        <div
          role="slider"
          aria-label="Audio playback progress"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(currentTime)}
          tabIndex={0}
          onClick={handleScrub}
          className="h-10 min-w-[60px] flex-1 cursor-pointer select-none px-1"
        >
          <svg
            viewBox={`0 0 ${BAR_COUNT * 5} 40`}
            preserveAspectRatio="none"
            className="h-full w-full overflow-visible"
            aria-hidden="true"
          >
            {peaks.map((peak, idx) => {
              const barRatio = idx / peaks.length;
              const active = barRatio <= progress;
              const barHeight = Math.max(5, Math.round(peak * 36));
              const y = (40 - barHeight) / 2;
              return (
                <rect
                  key={idx}
                  x={idx * 5 + 1}
                  y={y}
                  width={3}
                  height={barHeight}
                  rx={1.5}
                  fill={active ? accentColor : 'rgba(148, 163, 184, 0.35)'}
                />
              );
            })}
          </svg>
        </div>

        <div className="flex shrink-0 items-center gap-1.5 font-mono text-[11px] text-slate-400">
          <Volume2 className="h-3.5 w-3.5 text-slate-500" />
          <span>{formatTime(currentTime)}</span>
          <span className="text-slate-600">/</span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>
    </div>
  );
};
