import type { AnalyzeResponse, CharacterResult, JobState } from './types';

const API_BASE = '/api';

export async function uploadAudio(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let detail = `Upload failed (${response.status})`;
    try {
      const data = await response.json();
      if (data?.detail) detail = String(data.detail);
    } catch {
      // ignore json parse error
    }
    throw new Error(detail);
  }

  return (await response.json()) as AnalyzeResponse;
}

export async function fetchJob(jobId: string): Promise<JobState> {
  const response = await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}`);
  if (!response.ok) {
    let detail = `Failed to load job status (${response.status})`;
    try {
      const data = await response.json();
      if (data?.detail) detail = String(data.detail);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return (await response.json()) as JobState;
}

export async function renameCharacter(
  jobId: string,
  characterId: string,
  name: string,
): Promise<CharacterResult> {
  const response = await fetch(
    `${API_BASE}/jobs/${encodeURIComponent(jobId)}/characters/${encodeURIComponent(characterId)}`,
    {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    },
  );
  if (!response.ok) {
    let detail = `Failed to rename character (${response.status})`;
    try {
      const data = await response.json();
      if (data?.detail) detail = String(data.detail);
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return (await response.json()) as CharacterResult;
}

export function getCharacterDownloadUrl(
  jobId: string,
  characterId: string,
  format: 'wav' | 'mp3',
  clip: 'combined' | 'clean' = 'combined',
  customName?: string,
): string {
  const params = new URLSearchParams({ format, clip });
  if (customName && customName.trim()) {
    params.set('name', customName.trim());
  }
  return `${API_BASE}/jobs/${encodeURIComponent(jobId)}/download/${encodeURIComponent(characterId)}?${params.toString()}`;
}

export function getDownloadAllUrl(
  jobId: string,
  format: 'wav' | 'mp3' = 'wav',
  clip: 'combined' | 'clean' = 'combined',
): string {
  const params = new URLSearchParams({ format, clip });
  return `${API_BASE}/jobs/${encodeURIComponent(jobId)}/download-all?${params.toString()}`;
}

export function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const tenths = Math.floor((seconds % 1) * 10);
  if (seconds < 60) {
    return `${mins}:${secs.toString().padStart(2, '0')}.${tenths}`;
  }
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function formatDurationShort(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0.0s';
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const rem = Math.round(seconds % 60);
  return `${mins}m ${rem}s`;
}
