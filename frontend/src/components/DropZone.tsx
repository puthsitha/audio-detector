import React, { useCallback, useRef, useState } from 'react';
import { FileAudio, Upload, AlertCircle } from 'lucide-react';

const ACCEPTED_EXTENSIONS = ['.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.webm'];
const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.join(',');

interface DropZoneProps {
  disabled?: boolean;
  onFileSelected: (file: File) => void;
}

export const DropZone: React.FC<DropZoneProps> = ({ disabled = false, onFileSelected }) => {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  const validateAndEmit = useCallback(
    (file: File | undefined) => {
      setLocalError(null);
      if (!file) return;
      const lower = file.name.toLowerCase();
      const valid = ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
      if (!valid) {
        setLocalError(
          `Unsupported file format. Please upload one of: ${ACCEPTED_EXTENSIONS.join(', ')}`,
        );
        return;
      }
      if (file.size === 0) {
        setLocalError('Selected file is empty.');
        return;
      }
      onFileSelected(file);
    },
    [onFileSelected],
  );

  const handleDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      event.stopPropagation();
      setIsDragging(false);
      if (disabled) return;
      const file = event.dataTransfer.files?.[0];
      validateAndEmit(file);
    },
    [disabled, validateAndEmit],
  );

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    setIsDragging(false);
  };

  return (
    <div className="w-full">
      <div
        role="button"
        tabIndex={disabled ? -1 : 0}
        aria-label="Upload audio file"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if (!disabled && (e.key === 'Enter' || e.key === ' ')) {
            e.preventDefault();
            inputRef.current?.click();
          }
        }}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={[
          'group relative flex flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition-all duration-200',
          disabled
            ? 'cursor-not-allowed border-slate-800 bg-slate-900/40 opacity-60'
            : isDragging
              ? 'cursor-pointer border-indigo-400 bg-indigo-500/10 shadow-[0_0_40px_-10px_rgba(99,102,241,0.5)]'
              : 'cursor-pointer border-slate-700/80 bg-slate-900/60 hover:border-indigo-500/70 hover:bg-slate-900/90',
        ].join(' ')}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          disabled={disabled}
          className="hidden"
          onChange={(e) => {
            validateAndEmit(e.target.files?.[0]);
            e.target.value = '';
          }}
        />

        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-500/15 text-indigo-400 ring-1 ring-indigo-500/30 transition-transform duration-200 group-hover:scale-105">
          {isDragging ? <FileAudio className="h-7 w-7" /> : <Upload className="h-7 w-7" />}
        </div>

        <h2 className="text-lg font-semibold text-slate-100">
          {isDragging ? 'Drop your audio file here' : 'Drop an audio file or click to browse'}
        </h2>
        <p className="mt-1.5 max-w-md text-sm text-slate-400">
          Automatically detects every distinct speaker, predicts gender &amp; age group locally on
          your Mac, and extracts isolated clips ready for playback or voice cloning.
        </p>

        <div className="mt-5 flex flex-wrap items-center justify-center gap-1.5">
          {ACCEPTED_EXTENSIONS.map((ext) => (
            <span
              key={ext}
              className="rounded-md bg-slate-800/90 px-2 py-0.5 font-mono text-xs text-slate-300 ring-1 ring-slate-700/70"
            >
              {ext}
            </span>
          ))}
        </div>
      </div>

      {localError && (
        <div className="mt-3 flex items-center gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
          <span>{localError}</span>
        </div>
      )}
    </div>
  );
};
