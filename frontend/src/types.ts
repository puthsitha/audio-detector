export type JobStatus = 'queued' | 'processing' | 'completed' | 'failed';

export type JobStage =
  | 'uploading'
  | 'converting'
  | 'detecting_speakers'
  | 'classifying'
  | 'extracting_clips'
  | 'done'
  | 'error';

export type GenderLabel = 'Male' | 'Female';
export type AgeGroupLabel = 'Young' | 'Adult' | 'Old';

export interface SegmentInterval {
  start: number;
  end: number;
  duration: number;
}

export interface CharacterResult {
  id: string;
  speaker_label: string;
  name: string;
  gender: GenderLabel;
  gender_confidence: number;
  age_group: AgeGroupLabel;
  estimated_age: number;
  child_probability: number;
  age_confidence: number;
  total_speaking_duration: number;
  segment_count: number;
  segments: SegmentInterval[];
  clean_clip_start: number | null;
  clean_clip_end: number | null;
  clean_clip_duration: number | null;
  combined_clip_duration: number;
  audio_url: string;
  clean_clip_url: string | null;
  default_filename: string;
}

export interface SummaryBreakdown {
  total_characters: number;
  gender_counts: Record<string, number>;
  age_group_counts: Record<string, number>;
  audio_duration: number;
  diarization_engine: string;
  device_used: string;
  fallback_notice: string | null;
}

export interface JobState {
  job_id: string;
  status: JobStatus;
  stage: JobStage;
  progress: number;
  stage_message: string;
  original_filename: string;
  created_at: string;
  completed_at: string | null;
  error: string | null;
  summary: SummaryBreakdown | null;
  characters: CharacterResult[];
}

export interface AnalyzeResponse {
  job_id: string;
  status: JobStatus;
  stage: JobStage;
  message: string;
}
