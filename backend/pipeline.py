"""Sequential memory-managed processing pipeline and job state management."""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.audio_utils import (
    AudioValidationError,
    build_character_filename,
    convert_to_analysis_wav,
    convert_to_clean_wav,
    extract_clean_clip,
    extract_combined_clip,
    merge_and_pad_segments,
    probe_audio,
    select_cleanest_segment,
)
from backend.classifier import AgeGenderClassifier
from backend.config import JOB_OUTPUT_DIR, release_memory
from backend.diarizer import diarize_audio
from backend.models import (
    CharacterResult,
    JobStage,
    JobState,
    JobStatus,
    SegmentInterval,
    SummaryBreakdown,
)

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    """Manages analysis jobs, progress updates, disk persistence, and sequential ML execution."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir: Path = base_dir or JOB_OUTPUT_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, JobState] = {}
        # Single worker ensures diarization and classification run strictly sequentially on 16GB RAM
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ml-pipeline")
        self._load_existing_jobs()

    def _load_existing_jobs(self) -> None:
        if not self.base_dir.exists():
            return
        for job_json in self.base_dir.glob("*/job.json"):
            try:
                state = JobState.model_validate_json(job_json.read_text(encoding="utf-8"))
                self._jobs[state.job_id] = state
            except Exception:
                continue

    def job_dir(self, job_id: str) -> Path:
        return self.base_dir / job_id

    def _persist_job(self, state: JobState) -> None:
        jdir = self.job_dir(state.job_id)
        jdir.mkdir(parents=True, exist_ok=True)
        (jdir / "job.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")

    def create_job(self, original_filename: str, job_id: str | None = None) -> JobState:
        jid = job_id or uuid.uuid4().hex[:12]
        state = JobState(
            job_id=jid,
            status="queued",
            stage="uploading",
            progress=5,
            stage_message="Audio uploaded and queued for processing.",
            original_filename=original_filename,
            created_at=_utc_now_iso(),
        )
        with self._lock:
            self._jobs[jid] = state
            self._persist_job(state)
        return state

    def get_job(self, job_id: str) -> JobState | None:
        with self._lock:
            state = self._jobs.get(job_id)
            if state is not None:
                return state.model_copy(deep=True)
        # Try loading from disk in case created externally
        job_json = self.job_dir(job_id) / "job.json"
        if job_json.exists():
            try:
                loaded = JobState.model_validate_json(job_json.read_text(encoding="utf-8"))
                with self._lock:
                    self._jobs[job_id] = loaded
                return loaded.model_copy(deep=True)
            except Exception:
                return None
        return None

    def update_progress(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        stage: JobStage | None = None,
        progress: int | None = None,
        stage_message: str | None = None,
        error: str | None = None,
        summary: SummaryBreakdown | None = None,
        characters: list[CharacterResult] | None = None,
        completed_at: str | None = None,
    ) -> JobState:
        with self._lock:
            state = self._jobs[job_id]
            updates = {}
            if status is not None:
                updates["status"] = status
            if stage is not None:
                updates["stage"] = stage
            if progress is not None:
                updates["progress"] = progress
            if stage_message is not None:
                updates["stage_message"] = stage_message
            if error is not None:
                updates["error"] = error
            if summary is not None:
                updates["summary"] = summary
            if characters is not None:
                updates["characters"] = characters
            if completed_at is not None:
                updates["completed_at"] = completed_at

            updated = state.model_copy(update=updates)
            self._jobs[job_id] = updated
            self._persist_job(updated)
            return updated.model_copy(deep=True)

    def rename_character(self, job_id: str, character_id: str, new_name: str) -> CharacterResult | None:
        cleaned = new_name.strip()
        if not cleaned:
            return None
        with self._lock:
            state = self._jobs.get(job_id)
            if state is None:
                return None
            new_chars: list[CharacterResult] = []
            target: CharacterResult | None = None
            for char in state.characters:
                if char.id == character_id:
                    updated_char = char.model_copy(
                        update={
                            "name": cleaned,
                            "default_filename": build_character_filename(
                                cleaned, char.gender, char.age_group, ext="wav"
                            ),
                        }
                    )
                    target = updated_char
                    new_chars.append(updated_char)
                else:
                    new_chars.append(char)
            if target is None:
                return None
            updated_state = state.model_copy(update={"characters": new_chars})
            self._jobs[job_id] = updated_state
            self._persist_job(updated_state)
            return target.model_copy(deep=True)

    def submit_job(self, job_id: str, input_audio_path: Path) -> None:
        self._executor.submit(self.run_pipeline, job_id, input_audio_path)

    def run_pipeline(self, job_id: str, input_audio_path: Path) -> JobState:
        """Execute full sequential audio analysis pipeline for a job."""
        jdir = self.job_dir(job_id)
        clips_dir = jdir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        analysis_wav = jdir / "analysis_16k_mono.wav"
        clean_wav = jdir / "source_clean.wav"

        try:
            # Stage 1: Validate & Convert Audio
            self.update_progress(
                job_id,
                status="processing",
                stage="converting",
                progress=15,
                stage_message="Validating audio and normalizing to 16 kHz mono WAV...",
            )
            metadata = probe_audio(input_audio_path)
            convert_to_analysis_wav(input_audio_path, analysis_wav)
            convert_to_clean_wav(input_audio_path, clean_wav)

            # Stage 2: Speaker Diarization (load -> infer -> release memory)
            self.update_progress(
                job_id,
                stage="detecting_speakers",
                progress=38,
                stage_message="Detecting distinct speakers with pyannote 3.1...",
            )
            diar_result = diarize_audio(analysis_wav)
            release_memory()

            # Group segments by speaker ordered by first appearance
            speaker_order: list[str] = []
            speaker_raw_segments: dict[str, list[tuple[float, float]]] = {}
            for start, end, label in sorted(diar_result.segments, key=lambda x: x[0]):
                if label not in speaker_raw_segments:
                    speaker_order.append(label)
                    speaker_raw_segments[label] = []
                speaker_raw_segments[label].append((start, end))

            if not speaker_order:
                summary = SummaryBreakdown(
                    total_characters=0,
                    gender_counts={"Male": 0, "Female": 0},
                    age_group_counts={"Young": 0, "Adult": 0, "Old": 0},
                    audio_duration=round(metadata.duration, 2),
                    diarization_engine=diar_result.engine_name,
                    device_used=diar_result.device_used,
                    fallback_notice=diar_result.fallback_notice,
                )
                return self.update_progress(
                    job_id,
                    status="completed",
                    stage="done",
                    progress=100,
                    stage_message="Analysis complete. No speech detected.",
                    summary=summary,
                    characters=[],
                    completed_at=_utc_now_iso(),
                )

            # Stage 3: Gender & Age Classification (sequential after diarizer memory release)
            self.update_progress(
                job_id,
                stage="classifying",
                progress=65,
                stage_message=f"Classifying gender and age for {len(speaker_order)} detected speaker(s)...",
            )
            audio_16k, sr_16k = sf.read(str(analysis_wav), dtype="float32")
            if audio_16k.ndim == 2:
                audio_16k = np.mean(audio_16k, axis=1)

            classifier = AgeGenderClassifier()
            predictions = {}
            try:
                classifier.load()
                for idx, label in enumerate(speaker_order):
                    raw_segs = speaker_raw_segments[label]
                    predictions[label] = classifier.classify_speaker(audio_16k, int(sr_16k), raw_segs)
                    pct = 65 + int(((idx + 1) / len(speaker_order)) * 18)
                    self.update_progress(
                        job_id,
                        progress=pct,
                        stage_message=f"Classified speaker {idx + 1} of {len(speaker_order)}...",
                    )
            finally:
                classifier.unload()
                release_memory()

            # Stage 4: Extract combined and cleanest 5-15s character clips
            self.update_progress(
                job_id,
                stage="extracting_clips",
                progress=86,
                stage_message="Extracting smooth isolated audio clips per character...",
            )
            characters: list[CharacterResult] = []
            gender_counts = {"Male": 0, "Female": 0}
            age_counts = {"Young": 0, "Adult": 0, "Old": 0}

            for idx, label in enumerate(speaker_order, start=1):
                char_id = f"character_{idx}"
                display_name = f"Character {idx}"
                raw_segs = speaker_raw_segments[label]
                merged_segs = merge_and_pad_segments(raw_segs, audio_duration=metadata.duration)
                total_speaking = sum(e - s for s, e in raw_segs)

                combined_wav = clips_dir / f"{char_id}_combined.wav"
                combined_dur = extract_combined_clip(
                    source_wav=clean_wav,
                    segments=merged_segs,
                    output_wav=combined_wav,
                )

                clean_span = select_cleanest_segment(
                    speaker_segments=raw_segs,
                    audio_mono=audio_16k,
                    sr=int(sr_16k),
                    all_speaker_segments=diar_result.segments,
                    speaker_label=label,
                )
                clean_clip_wav = clips_dir / f"{char_id}_clean.wav"
                clean_dur, clean_start, clean_end = extract_clean_clip(
                    source_wav=clean_wav,
                    clean_span=clean_span,
                    combined_wav=combined_wav,
                    output_wav=clean_clip_wav,
                )

                pred = predictions[label]
                gender_counts[pred.gender] = gender_counts.get(pred.gender, 0) + 1
                age_counts[pred.age_group] = age_counts.get(pred.age_group, 0) + 1

                default_fname = build_character_filename(
                    display_name, pred.gender, pred.age_group, ext="wav"
                )
                characters.append(
                    CharacterResult(
                        id=char_id,
                        speaker_label=label,
                        name=display_name,
                        gender=pred.gender,
                        gender_confidence=pred.gender_confidence,
                        age_group=pred.age_group,
                        estimated_age=pred.estimated_age,
                        child_probability=pred.child_probability,
                        age_confidence=pred.age_confidence,
                        total_speaking_duration=round(total_speaking, 2),
                        segment_count=len(raw_segs),
                        segments=[
                            SegmentInterval(
                                start=round(s, 2),
                                end=round(e, 2),
                                duration=round(e - s, 2),
                            )
                            for s, e in raw_segs
                        ],
                        clean_clip_start=round(clean_start, 2) if clean_start is not None else None,
                        clean_clip_end=round(clean_end, 2) if clean_end is not None else None,
                        clean_clip_duration=round(clean_dur, 2),
                        combined_clip_duration=round(combined_dur, 2),
                        audio_url=f"/api/jobs/{job_id}/audio/{char_id}?clip=combined",
                        clean_clip_url=f"/api/jobs/{job_id}/audio/{char_id}?clip=clean",
                        default_filename=default_fname,
                    )
                )

            summary = SummaryBreakdown(
                total_characters=len(characters),
                gender_counts=gender_counts,
                age_group_counts=age_counts,
                audio_duration=round(metadata.duration, 2),
                diarization_engine=diar_result.engine_name,
                device_used=diar_result.device_used,
                fallback_notice=diar_result.fallback_notice,
            )

            return self.update_progress(
                job_id,
                status="completed",
                stage="done",
                progress=100,
                stage_message=f"Detected {len(characters)} character(s) successfully.",
                summary=summary,
                characters=characters,
                completed_at=_utc_now_iso(),
            )

        except AudioValidationError as exc:
            logger.warning("Job %s validation failed: %s", job_id, exc)
            return self.update_progress(
                job_id,
                status="failed",
                stage="error",
                progress=100,
                stage_message=str(exc),
                error=str(exc),
                completed_at=_utc_now_iso(),
            )
        except Exception as exc:
            logger.exception("Job %s failed unexpectedly", job_id)
            return self.update_progress(
                job_id,
                status="failed",
                stage="error",
                progress=100,
                stage_message=f"Analysis failed: {exc}",
                error=str(exc),
                completed_at=_utc_now_iso(),
            )
        finally:
            release_memory()


job_manager = JobManager()
