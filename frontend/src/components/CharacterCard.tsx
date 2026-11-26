import React, { useState } from 'react';
import {
  Check,
  ChevronDown,
  ChevronUp,
  Clock,
  Download,
  Edit3,
  Sparkles,
  User,
  X,
} from 'lucide-react';
import type { CharacterResult } from '../types';
import {
  formatDurationShort,
  formatTime,
  getCharacterDownloadUrl,
  renameCharacter,
} from '../api';
import { WaveformPlayer } from './WaveformPlayer';

interface CharacterCardProps {
  jobId: string;
  character: CharacterResult;
  index: number;
  onUpdated: (updated: CharacterResult) => void;
}

const ACCENT_PALETTE = [
  '#818cf8', // indigo
  '#38bdf8', // sky
  '#e879f9', // fuchsia
  '#34d399', // emerald
  '#fbbf24', // amber
  '#fb7185', // rose
];

export const CharacterCard: React.FC<CharacterCardProps> = ({
  jobId,
  character,
  index,
  onUpdated,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [draftName, setDraftName] = useState(character.name);
  const [savingName, setSavingName] = useState(false);
  const [clipMode, setClipMode] = useState<'combined' | 'clean'>('combined');
  const [downloadFormat, setDownloadFormat] = useState<'wav' | 'mp3'>('wav');
  const [showTimestamps, setShowTimestamps] = useState(false);

  const accent = ACCENT_PALETTE[index % ACCENT_PALETTE.length];
  const activeAudioUrl =
    clipMode === 'clean' && character.clean_clip_url
      ? character.clean_clip_url
      : character.audio_url;

  const handleSaveName = async () => {
    const trimmed = draftName.trim();
    if (!trimmed || trimmed === character.name) {
      setDraftName(character.name);
      setIsEditing(false);
      return;
    }
    setSavingName(true);
    try {
      const updated = await renameCharacter(jobId, character.id, trimmed);
      onUpdated(updated);
      setDraftName(updated.name);
      setIsEditing(false);
    } catch {
      // Keep editor open if rename failed
    } finally {
      setSavingName(false);
    }
  };

  const genderBadgeClasses =
    character.gender === 'Male'
      ? 'bg-sky-500/15 text-sky-300 ring-1 ring-sky-500/35'
      : 'bg-fuchsia-500/15 text-fuchsia-300 ring-1 ring-fuchsia-500/35';

  const ageBadgeClasses =
    character.age_group === 'Young'
      ? 'bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/35'
      : character.age_group === 'Adult'
        ? 'bg-amber-500/15 text-amber-300 ring-1 ring-amber-500/35'
        : 'bg-purple-500/15 text-purple-300 ring-1 ring-purple-500/35';

  const downloadUrl = getCharacterDownloadUrl(
    jobId,
    character.id,
    downloadFormat,
    clipMode,
    character.name,
  );

  return (
    <div className="flex flex-col justify-between rounded-2xl border border-slate-800/90 bg-slate-900/75 p-5 shadow-lg transition-colors hover:border-slate-700/90">
      <div className="space-y-4">
        {/* Header: Editable name & avatar */}
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 flex-1 items-center gap-3">
            <div
              className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl text-white shadow-inner"
              style={{
                backgroundColor: `${accent}22`,
                border: `1px solid ${accent}55`,
                color: accent,
              }}
            >
              <User className="h-5 w-5" />
            </div>

            <div className="min-w-0 flex-1">
              {isEditing ? (
                <div className="flex items-center gap-1.5">
                  <input
                    type="text"
                    value={draftName}
                    maxLength={80}
                    autoFocus
                    disabled={savingName}
                    onChange={(e) => setDraftName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleSaveName();
                      if (e.key === 'Escape') {
                        setDraftName(character.name);
                        setIsEditing(false);
                      }
                    }}
                    className="w-full rounded-lg border border-indigo-500/60 bg-slate-950 px-2.5 py-1 text-sm font-semibold text-white outline-none ring-2 ring-indigo-500/20"
                    placeholder="Character name..."
                  />
                  <button
                    type="button"
                    disabled={savingName}
                    onClick={handleSaveName}
                    aria-label="Save character name"
                    className="rounded-lg bg-emerald-600/90 p-1.5 text-white hover:bg-emerald-500"
                  >
                    <Check className="h-3.5 w-3.5" />
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setDraftName(character.name);
                      setIsEditing(false);
                    }}
                    aria-label="Cancel rename"
                    className="rounded-lg bg-slate-800 p-1.5 text-slate-300 hover:bg-slate-700"
                  >
                    <X className="h-3.5 w-3.5" />
                  </button>
                </div>
              ) : (
                <div className="flex items-center gap-2">
                  <h3 className="truncate text-base font-semibold text-slate-100">
                    {character.name}
                  </h3>
                  <button
                    type="button"
                    onClick={() => {
                      setDraftName(character.name);
                      setIsEditing(true);
                    }}
                    title="Rename character before downloading"
                    className="rounded-md p-1 text-slate-500 transition-colors hover:bg-slate-800 hover:text-slate-200"
                  >
                    <Edit3 className="h-3.5 w-3.5" />
                  </button>
                </div>
              )}

              <div className="mt-0.5 flex items-center gap-2 text-xs text-slate-400">
                <Clock className="h-3 w-3 text-slate-500" />
                <span>{formatDurationShort(character.total_speaking_duration)} speaking</span>
                <span className="text-slate-600">•</span>
                <span>
                  {character.segment_count}{' '}
                  {character.segment_count === 1 ? 'segment' : 'segments'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Gender & Age Badges */}
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <span
              className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-semibold ${genderBadgeClasses}`}
            >
              {character.gender}
              <span className="font-mono text-[11px] opacity-80">
                {Math.round(character.gender_confidence * 100)}%
              </span>
            </span>

            <span
              className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1 text-xs font-semibold ${ageBadgeClasses}`}
            >
              {character.age_group}
              <span className="font-mono text-[11px] opacity-85">
                ~{Math.round(character.estimated_age)} yrs
              </span>
              <span className="font-mono text-[10px] opacity-70">
                ({Math.round(character.age_confidence * 100)}%)
              </span>
            </span>
          </div>
          <p className="text-[11px] text-slate-500">
            Voice-based age estimate is an approximation derived from acoustic speech features.
          </p>
        </div>

        {/* Clip Mode Toggle (Combined vs Clean 5-15s Voice Cloning Sample) */}
        <div className="flex items-center justify-between gap-2">
          <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-0.5 text-[11px]">
            <button
              type="button"
              onClick={() => setClipMode('combined')}
              className={[
                'rounded-md px-2.5 py-1 font-medium transition-colors',
                clipMode === 'combined'
                  ? 'bg-slate-800 text-white'
                  : 'text-slate-400 hover:text-slate-200',
              ].join(' ')}
            >
              Full Voice ({formatDurationShort(character.combined_clip_duration)})
            </button>
            {character.clean_clip_url && (
              <button
                type="button"
                onClick={() => setClipMode('clean')}
                className={[
                  'inline-flex items-center gap-1 rounded-md px-2.5 py-1 font-medium transition-colors',
                  clipMode === 'clean'
                    ? 'bg-indigo-600/80 text-white'
                    : 'text-slate-400 hover:text-slate-200',
                ].join(' ')}
              >
                <Sparkles className="h-3 w-3" />
                Clean Clip (
                {formatDurationShort(character.clean_clip_duration ?? character.combined_clip_duration)}
                )
              </button>
            )}
          </div>
        </div>

        {/* Audio Player */}
        <WaveformPlayer src={activeAudioUrl} accentColor={accent} />

        {/* Collapsible Timestamps */}
        <div className="rounded-xl border border-slate-800/70 bg-slate-950/50">
          <button
            type="button"
            onClick={() => setShowTimestamps((prev) => !prev)}
            className="flex w-full items-center justify-between px-3 py-2 text-xs font-medium text-slate-400 hover:text-slate-200"
          >
            <span>
              Speaking Timestamps ({character.segments.length})
              {character.clean_clip_start != null && character.clean_clip_end != null && (
                <span className="ml-2 font-mono text-[11px] text-indigo-400">
                  Best clip: {formatTime(character.clean_clip_start)}–
                  {formatTime(character.clean_clip_end)}
                </span>
              )}
            </span>
            {showTimestamps ? (
              <ChevronUp className="h-3.5 w-3.5" />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" />
            )}
          </button>

          {showTimestamps && (
            <div className="max-h-32 overflow-y-auto border-t border-slate-800/70 px-3 py-2">
              <div className="flex flex-wrap gap-1.5">
                {character.segments.map((seg, i) => (
                  <span
                    key={`${seg.start}-${i}`}
                    className="rounded-md bg-slate-900 px-2 py-0.5 font-mono text-[11px] text-slate-300 ring-1 ring-slate-800"
                  >
                    {formatTime(seg.start)}–{formatTime(seg.end)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer: Download Controls */}
      <div className="mt-4 flex items-center justify-between gap-2 border-t border-slate-800/80 pt-4">
        <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setDownloadFormat('wav')}
            className={[
              'rounded-md px-2.5 py-1 font-medium transition-colors',
              downloadFormat === 'wav'
                ? 'bg-slate-800 text-white'
                : 'text-slate-400 hover:text-slate-200',
            ].join(' ')}
          >
            .wav
          </button>
          <button
            type="button"
            onClick={() => setDownloadFormat('mp3')}
            className={[
              'rounded-md px-2.5 py-1 font-medium transition-colors',
              downloadFormat === 'mp3'
                ? 'bg-slate-800 text-white'
                : 'text-slate-400 hover:text-slate-200',
            ].join(' ')}
          >
            .mp3
          </button>
        </div>

        <a
          href={downloadUrl}
          download
          className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-slate-800 px-3.5 py-2 text-xs font-semibold text-white ring-1 ring-slate-700 transition-all hover:bg-indigo-600 hover:ring-indigo-500 active:scale-[0.98]"
        >
          <Download className="h-3.5 w-3.5" />
          Download {clipMode === 'clean' ? 'Clean Clip' : 'Voice'} (.{downloadFormat})
        </a>
      </div>
    </div>
  );
};
