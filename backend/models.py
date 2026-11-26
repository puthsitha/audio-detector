"""Pydantic schemas for the Audio Character Detector API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "processing", "completed", "failed"]
JobStage = Literal[
    "uploading",
    "converting",
    "detecting_speakers",
    "classifying",
    "extracting_clips",
    "done",
    "error",
]
GenderLabel = Literal["Male", "Female"]
AgeGroupLabel = Literal["Young", "Adult", "Old"]


class SegmentInterval(BaseModel):
    start: float = Field(..., ge=0.0, description="Segment start time in seconds")
    end: float = Field(..., ge=0.0, description="Segment end time in seconds")
    duration: float = Field(..., ge=0.0, description="Segment duration in seconds")


class CharacterResult(BaseModel):
    id: str = Field(..., description="Unique character identifier within the job")
    speaker_label: str = Field(..., description="Raw diarizer speaker label")
    name: str = Field(..., description="Display name (editable by user)")
    gender: GenderLabel
    gender_confidence: float = Field(..., ge=0.0, le=1.0)
    age_group: AgeGroupLabel
    estimated_age: float = Field(..., ge=0.0, le=100.0)
    child_probability: float = Field(..., ge=0.0, le=1.0)
    age_confidence: float = Field(..., ge=0.0, le=1.0)
    total_speaking_duration: float = Field(..., ge=0.0)
    segment_count: int = Field(..., ge=0)
    segments: list[SegmentInterval] = Field(default_factory=list)
    clean_clip_start: Optional[float] = None
    clean_clip_end: Optional[float] = None
    clean_clip_duration: Optional[float] = None
    combined_clip_duration: float = Field(..., ge=0.0)
    audio_url: str
    clean_clip_url: Optional[str] = None
    default_filename: str


class SummaryBreakdown(BaseModel):
    total_characters: int = Field(..., ge=0)
    gender_counts: dict[str, int] = Field(default_factory=dict)
    age_group_counts: dict[str, int] = Field(default_factory=dict)
    audio_duration: float = Field(..., ge=0.0)
    diarization_engine: str
    device_used: str = "cpu"
    fallback_notice: Optional[str] = None


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    progress: int = Field(..., ge=0, le=100)
    stage_message: str
    original_filename: str
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    summary: Optional[SummaryBreakdown] = None
    characters: list[CharacterResult] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    job_id: str
    status: JobStatus
    stage: JobStage
    message: str


class RenameCharacterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
