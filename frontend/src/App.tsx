import React, { useCallback, useEffect, useState } from 'react';
import { AudioWaveform, AlertCircle, ShieldCheck } from 'lucide-react';
import type { CharacterResult, JobState } from './types';
import { fetchJob, uploadAudio } from './api';
import { DropZone } from './components/DropZone';
import { ProcessingPipeline } from './components/ProcessingPipeline';
import { SummaryBar } from './components/SummaryBar';
import { CharacterCard } from './components/CharacterCard';

export const App: React.FC = () => {
  const [job, setJob] = useState<JobState | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const isRunning =
    isUploading ||
    (job !== null && (job.status === 'queued' || job.status === 'processing'));

  const handleFileSelected = useCallback(async (file: File) => {
    setUploadError(null);
    setIsUploading(true);
    setJob({
      job_id: 'pending',
      status: 'processing',
      stage: 'uploading',
      progress: 5,
      stage_message: `Uploading ${file.name}...`,
      original_filename: file.name,
      created_at: new Date().toISOString(),
      completed_at: null,
      error: null,
      summary: null,
      characters: [],
    });

    try {
      const response = await uploadAudio(file);
      const initialState = await fetchJob(response.job_id);
      setJob(initialState);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to upload audio file.';
      setUploadError(message);
      setJob(null);
    } finally {
      setIsUploading(false);
    }
  }, []);

  // Poll job status until completed or failed
  useEffect(() => {
    if (!job || job.job_id === 'pending') return;
    if (job.status === 'completed' || job.status === 'failed') return;

    let active = true;
    const timer = window.setInterval(async () => {
      try {
        const latest = await fetchJob(job.job_id);
        if (active) setJob(latest);
      } catch {
        // Keep polling on transient error
      }
    }, 800);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [job?.job_id, job?.status]);

  const handleCharacterUpdated = useCallback((updated: CharacterResult) => {
    setJob((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        characters: prev.characters.map((c) => (c.id === updated.id ? updated : c)),
      };
    });
  }, []);

  const handleReset = () => {
    setJob(null);
    setUploadError(null);
    setIsUploading(false);
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_rgba(79,70,229,0.14),_transparent_55%)] bg-slate-950">
      {/* Top Navbar */}
      <header className="sticky top-0 z-20 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3.5 sm:px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-indigo-600 text-white shadow-lg shadow-indigo-900/40">
              <AudioWaveform className="h-5 w-5" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-tight text-white">
                VoiceCast
              </h1>
              <p className="text-[11px] text-slate-400">
                Local Audio Character Detector &amp; Voice Extractor
              </p>
            </div>
          </div>

          <div className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/25 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-300">
            <ShieldCheck className="h-3.5 w-3.5" />
            100% Local Processing
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto max-w-6xl space-y-6 px-4 py-8 sm:px-6">
        {/* Upload Area (shown when no completed job, or when error) */}
        {(!job || job.status === 'failed') && (
          <section className="space-y-4">
            <div className="max-w-2xl">
              <h2 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
                Detect speakers, gender &amp; age from any audio
              </h2>
              <p className="mt-2 text-sm text-slate-400">
                Upload a podcast, dialogue, interview, or audiobook clip. VoiceCast separates every
                character using <span className="font-mono text-slate-300">pyannote 3.1</span> and
                classifies demographics with{' '}
                <span className="font-mono text-slate-300">wav2vec2</span>, producing clean voice
                clips ready to download.
              </p>
            </div>

            <DropZone disabled={isRunning} onFileSelected={handleFileSelected} />

            {uploadError && (
              <div className="flex items-center gap-2.5 rounded-xl border border-rose-500/35 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
                <span>{uploadError}</span>
              </div>
            )}
          </section>
        )}

        {/* Live Progress Tracker */}
        {job && job.status !== 'completed' && (
          <section>
            <ProcessingPipeline job={job} />
          </section>
        )}

        {/* Results Dashboard */}
        {job && job.status === 'completed' && job.summary && (
          <section className="space-y-6">
            <SummaryBar
              jobId={job.job_id}
              originalFilename={job.original_filename}
              summary={job.summary}
              onReset={handleReset}
            />

            {job.characters.length === 0 ? (
              <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-10 text-center">
                <p className="text-base font-semibold text-slate-200">
                  No distinct speech segments were detected in this audio.
                </p>
                <p className="mt-1 text-sm text-slate-400">
                  Try uploading a clearer speech recording with audible dialogue.
                </p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                {job.characters.map((character, idx) => (
                  <CharacterCard
                    key={character.id}
                    jobId={job.job_id}
                    character={character}
                    index={idx}
                    onUpdated={handleCharacterUpdated}
                  />
                ))}
              </div>
            )}
          </section>
        )}
      </main>
    </div>
  );
};

export default App;
