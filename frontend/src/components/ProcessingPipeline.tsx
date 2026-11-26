import React from 'react';
import { CheckCircle2, Circle, Loader2, XCircle } from 'lucide-react';
import type { JobStage, JobState } from '../types';

const STAGES: { key: JobStage; label: string }[] = [
  { key: 'uploading', label: 'Uploading' },
  { key: 'converting', label: 'Converting Audio' },
  { key: 'detecting_speakers', label: 'Detecting Speakers' },
  { key: 'classifying', label: 'Classifying Age & Gender' },
  { key: 'extracting_clips', label: 'Extracting Clips' },
  { key: 'done', label: 'Done' },
];

interface ProcessingPipelineProps {
  job: JobState;
}

export const ProcessingPipeline: React.FC<ProcessingPipelineProps> = ({ job }) => {
  const activeIdx = STAGES.findIndex((s) => s.key === job.stage);
  const isError = job.status === 'failed' || job.stage === 'error';

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/75 p-6 shadow-xl">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-indigo-400">
            Local ML Pipeline
          </p>
          <h3 className="mt-0.5 text-base font-semibold text-slate-100">
            {job.original_filename}
          </h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-sm font-semibold text-indigo-300">
            {job.progress}%
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mt-4 h-2 w-full overflow-hidden rounded-full bg-slate-800">
        <div
          className={[
            'h-full rounded-full transition-all duration-500',
            isError
              ? 'bg-rose-500'
              : job.stage === 'done'
                ? 'bg-emerald-500'
                : 'bg-gradient-to-r from-indigo-500 via-violet-500 to-cyan-400',
          ].join(' ')}
          style={{ width: `${Math.max(4, job.progress)}%` }}
        />
      </div>

      {/* Stage Stepper */}
      <div className="mt-6 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {STAGES.map((step, index) => {
          const isComplete = !isError && (job.stage === 'done' || index < activeIdx);
          const isCurrent = !isError && job.stage !== 'done' && index === activeIdx;

          return (
            <div
              key={step.key}
              className={[
                'flex items-center gap-2.5 rounded-xl border px-3 py-2.5 text-xs font-medium transition-colors',
                isComplete
                  ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200'
                  : isCurrent
                    ? 'border-indigo-500/50 bg-indigo-500/15 text-indigo-100 shadow-sm'
                    : 'border-slate-800/80 bg-slate-950/40 text-slate-500',
              ].join(' ')}
            >
              {isComplete ? (
                <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-400" />
              ) : isCurrent ? (
                <Loader2 className="h-4 w-4 shrink-0 animate-spin text-indigo-400" />
              ) : (
                <Circle className="h-4 w-4 shrink-0 text-slate-600" />
              )}
              <span className="leading-tight">{step.label}</span>
            </div>
          );
        })}
      </div>

      {/* Live Status Message */}
      <div className="mt-4 flex items-center gap-2 text-sm text-slate-300">
        {isError ? (
          <>
            <XCircle className="h-4 w-4 shrink-0 text-rose-400" />
            <span className="text-rose-200">{job.error || job.stage_message}</span>
          </>
        ) : (
          <span className="text-slate-400">{job.stage_message}</span>
        )}
      </div>
    </div>
  );
};
