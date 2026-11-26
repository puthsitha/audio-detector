import React, { useState } from 'react';
import {
  Users,
  Clock,
  Cpu,
  Download,
  AlertTriangle,
  Sparkles,
  RotateCcw,
} from 'lucide-react';
import type { SummaryBreakdown } from '../types';
import { formatDurationShort, getDownloadAllUrl } from '../api';

interface SummaryBarProps {
  jobId: string;
  originalFilename: string;
  summary: SummaryBreakdown;
  onReset: () => void;
}

export const SummaryBar: React.FC<SummaryBarProps> = ({
  jobId,
  originalFilename,
  summary,
  onReset,
}) => {
  const [downloadFormat, setDownloadFormat] = useState<'wav' | 'mp3'>('wav');

  const maleCount = summary.gender_counts.Male ?? 0;
  const femaleCount = summary.gender_counts.Female ?? 0;
  const youngCount = summary.age_group_counts.Young ?? 0;
  const adultCount = summary.age_group_counts.Adult ?? 0;
  const oldCount = summary.age_group_counts.Old ?? 0;

  const isPyannote = summary.diarization_engine.toLowerCase().includes('pyannote');

  return (
    <div className="space-y-4">
      <div className="rounded-2xl border border-slate-800 bg-slate-900/80 p-5 shadow-xl">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          {/* Left metrics */}
          <div className="flex flex-wrap items-center gap-4">
            <div className="flex items-center gap-3 rounded-xl border border-indigo-500/30 bg-indigo-500/10 px-4 py-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-300">
                <Users className="h-5 w-5" />
              </div>
              <div>
                <p className="text-[11px] font-medium uppercase tracking-wider text-indigo-300/80">
                  Characters Detected
                </p>
                <p className="text-2xl font-bold leading-tight text-white">
                  {summary.total_characters}
                </p>
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="rounded-md bg-sky-500/15 px-2.5 py-1 font-medium text-sky-300 ring-1 ring-sky-500/30">
                  {maleCount} Male
                </span>
                <span className="rounded-md bg-fuchsia-500/15 px-2.5 py-1 font-medium text-fuchsia-300 ring-1 ring-fuchsia-500/30">
                  {femaleCount} Female
                </span>
                <span className="text-slate-600">|</span>
                <span className="rounded-md bg-emerald-500/15 px-2.5 py-1 font-medium text-emerald-300 ring-1 ring-emerald-500/30">
                  {youngCount} Young
                </span>
                <span className="rounded-md bg-amber-500/15 px-2.5 py-1 font-medium text-amber-300 ring-1 ring-amber-500/30">
                  {adultCount} Adult
                </span>
                <span className="rounded-md bg-purple-500/15 px-2.5 py-1 font-medium text-purple-300 ring-1 ring-purple-500/30">
                  {oldCount} Old
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-3 text-xs text-slate-400">
                <span className="inline-flex items-center gap-1.5">
                  <Clock className="h-3.5 w-3.5 text-slate-500" />
                  Duration: <strong className="text-slate-200">{formatDurationShort(summary.audio_duration)}</strong>
                </span>
                <span className="text-slate-700">•</span>
                <span className="inline-flex items-center gap-1.5">
                  <Cpu className="h-3.5 w-3.5 text-slate-500" />
                  Engine:
                  <span
                    className={[
                      'inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium',
                      isPyannote
                        ? 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30'
                        : 'bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/30',
                    ].join(' ')}
                  >
                    <Sparkles className="h-3 w-3" />
                    {summary.diarization_engine}
                  </span>
                </span>
                <span className="text-slate-700">•</span>
                <span className="max-w-[220px] truncate text-slate-500" title={originalFilename}>
                  {originalFilename}
                </span>
              </div>
            </div>
          </div>

          {/* Right actions */}
          <div className="flex flex-wrap items-center gap-2.5">
            <div className="inline-flex rounded-lg border border-slate-700/80 bg-slate-950 p-0.5 text-xs">
              <button
                type="button"
                onClick={() => setDownloadFormat('wav')}
                className={[
                  'rounded-md px-2.5 py-1.5 font-medium transition-colors',
                  downloadFormat === 'wav'
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-400 hover:text-slate-200',
                ].join(' ')}
              >
                WAV
              </button>
              <button
                type="button"
                onClick={() => setDownloadFormat('mp3')}
                className={[
                  'rounded-md px-2.5 py-1.5 font-medium transition-colors',
                  downloadFormat === 'mp3'
                    ? 'bg-slate-800 text-white'
                    : 'text-slate-400 hover:text-slate-200',
                ].join(' ')}
              >
                MP3
              </button>
            </div>

            <a
              href={getDownloadAllUrl(jobId, downloadFormat, 'combined')}
              download
              className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-lg shadow-indigo-900/30 transition-all hover:bg-indigo-500 active:scale-[0.98]"
            >
              <Download className="h-4 w-4" />
              Download All (.zip)
            </a>

            <button
              type="button"
              onClick={onReset}
              className="inline-flex items-center gap-1.5 rounded-xl border border-slate-700/80 bg-slate-800/70 px-3.5 py-2.5 text-sm font-medium text-slate-200 transition-colors hover:border-slate-600 hover:bg-slate-800"
            >
              <RotateCcw className="h-4 w-4" />
              New File
            </button>
          </div>
        </div>
      </div>

      {summary.fallback_notice && (
        <div className="flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-xs text-amber-200">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
          <div className="space-y-1">
            <p className="font-semibold text-amber-100">
              Fallback Diarization Engine Active ({summary.diarization_engine})
            </p>
            <p className="leading-relaxed text-amber-200/90">{summary.fallback_notice}</p>
          </div>
        </div>
      )}
    </div>
  );
};
