"""Audio validation, ffmpeg conversion, segment merging, clip extraction, and ZIP bundling."""

from __future__ import annotations

import json
import re
import subprocess
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf

from backend.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    CLEAN_CLIP_MAX_SECONDS,
    CLEAN_CLIP_MIN_SECONDS,
    CLIP_FADE_SECONDS,
    CLIP_SILENCE_GAP_SECONDS,
    MAX_AUDIO_DURATION_SECONDS,
    SEGMENT_MERGE_GAP_SECONDS,
    SEGMENT_PADDING_SECONDS,
)


class AudioValidationError(ValueError):
    """Raised when an uploaded audio file fails validation or decoding."""


@dataclass(frozen=True)
class AudioMetadata:
    duration: float
    sample_rate: int
    channels: int
    format_name: str


def validate_extension(filename: str) -> str:
    """Validate and return the normalized lowercase extension for an audio filename."""
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        allowed_list = ", ".join(sorted(ALLOWED_AUDIO_EXTENSIONS))
        raise AudioValidationError(
            f"Unsupported audio format '{ext or 'unknown'}'. Supported formats: {allowed_list}."
        )
    return ext


def probe_audio(
    file_path: Path,
    max_duration_seconds: float | None = MAX_AUDIO_DURATION_SECONDS,
) -> AudioMetadata:
    """Inspect an audio file with ffprobe and validate that it is decodable."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        raise AudioValidationError("Uploaded audio file is empty or missing.")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name",
        "-show_entries",
        "stream=codec_type,sample_rate,channels,duration",
        "-of",
        "json",
        str(file_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or "File could not be decoded as audio."
        raise AudioValidationError(f"Invalid or corrupted audio file: {detail}")

    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AudioValidationError("Failed to read audio metadata.") from exc

    streams = payload.get("streams") or []
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not audio_streams:
        raise AudioValidationError("File does not contain a valid audio stream.")

    stream = audio_streams[0]
    fmt = payload.get("format") or {}

    raw_duration = fmt.get("duration") or stream.get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as exc:
        raise AudioValidationError("Could not determine audio duration.") from exc

    if not np.isfinite(duration) or duration <= 0.0:
        raise AudioValidationError("Audio duration must be greater than zero.")

    if max_duration_seconds is not None and duration > max_duration_seconds:
        raise AudioValidationError(
            f"Audio duration ({duration:.1f}s) exceeds maximum allowed ({max_duration_seconds:.0f}s)."
        )

    sample_rate = int(stream.get("sample_rate") or 16000)
    channels = int(stream.get("channels") or 1)
    format_name = str(fmt.get("format_name") or "unknown")
    return AudioMetadata(
        duration=duration,
        sample_rate=sample_rate,
        channels=channels,
        format_name=format_name,
    )


def convert_to_analysis_wav(input_path: Path, output_path: Path) -> Path:
    """Convert any supported audio file to 16 kHz mono 16-bit PCM WAV for ML models."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        detail = result.stderr.strip() or "ffmpeg conversion failed"
        raise AudioValidationError(f"Failed to convert audio for analysis: {detail}")
    return output_path


def convert_to_clean_wav(input_path: Path, output_path: Path) -> Path:
    """Convert input audio to high-quality PCM WAV preserving original sample rate/channels."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not output_path.exists():
        detail = result.stderr.strip() or "ffmpeg clean WAV conversion failed"
        raise AudioValidationError(f"Failed to prepare clean audio: {detail}")
    return output_path


def transcode_wav_to_mp3(wav_path: Path, mp3_path: Path) -> Path:
    """Transcode a WAV clip to high-quality VBR MP3."""
    mp3_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(wav_path),
        "-codec:a",
        "libmp3lame",
        "-qscale:a",
        "2",
        str(mp3_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not mp3_path.exists():
        detail = result.stderr.strip() or "ffmpeg MP3 encoding failed"
        raise RuntimeError(f"Failed to transcode clip to MP3: {detail}")
    return mp3_path


def merge_and_pad_segments(
    segments: Sequence[tuple[float, float]],
    audio_duration: float,
    pad_seconds: float = SEGMENT_PADDING_SECONDS,
    merge_gap_seconds: float = SEGMENT_MERGE_GAP_SECONDS,
) -> list[tuple[float, float]]:
    """Pad segments slightly, clamp to [0, audio_duration], and merge close/overlapping spans."""
    if not segments or audio_duration <= 0.0:
        return []

    cleaned: list[tuple[float, float]] = []
    for start, end in segments:
        s = max(0.0, min(float(start), audio_duration))
        e = max(0.0, min(float(end), audio_duration))
        if e > s:
            cleaned.append((s, e))

    if not cleaned:
        return []

    cleaned.sort(key=lambda item: (item[0], item[1]))

    padded = [
        (max(0.0, s - pad_seconds), min(audio_duration, e + pad_seconds))
        for s, e in cleaned
    ]

    merged: list[tuple[float, float]] = [padded[0]]
    for start, end in padded[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + merge_gap_seconds:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _apply_edge_fades(chunk: np.ndarray, fade_samples: int) -> np.ndarray:
    """Apply short cosine fade-in and fade-out ramps to prevent clicks between segments."""
    if chunk.shape[0] < 4 or fade_samples <= 0:
        return chunk
    n = min(fade_samples, chunk.shape[0] // 2)
    if n <= 1:
        return chunk

    out = chunk.copy()
    ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n, dtype=np.float32)))
    if out.ndim == 2:
        ramp = ramp[:, None]
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def extract_combined_clip(
    source_wav: Path,
    segments: Sequence[tuple[float, float]],
    output_wav: Path,
    fade_seconds: float = CLIP_FADE_SECONDS,
    silence_gap_seconds: float = CLIP_SILENCE_GAP_SECONDS,
) -> float:
    """Slice and smoothly concatenate speech segments from source_wav into output_wav."""
    data, sr = sf.read(str(source_wav), dtype="float32", always_2d=True)
    total_samples = data.shape[0]
    channels = data.shape[1]
    fade_samples = int(round(fade_seconds * sr))
    gap_samples = int(round(silence_gap_seconds * sr))

    chunks: list[np.ndarray] = []
    silence = np.zeros((gap_samples, channels), dtype=np.float32)

    for idx, (start, end) in enumerate(segments):
        s_idx = max(0, min(total_samples, int(round(start * sr))))
        e_idx = max(0, min(total_samples, int(round(end * sr))))
        if e_idx <= s_idx:
            continue
        piece = _apply_edge_fades(data[s_idx:e_idx], fade_samples)
        if chunks and gap_samples > 0:
            chunks.append(silence)
        chunks.append(piece)

    if not chunks:
        raise AudioValidationError("No non-empty audio segments available to extract.")

    combined = np.concatenate(chunks, axis=0)
    # Gentle peak limiting if needed without distorting normal audio
    peak = float(np.max(np.abs(combined)))
    if peak > 0.99:
        combined = combined * (0.98 / peak)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), combined, sr, subtype="PCM_16")
    return float(combined.shape[0] / sr)


def _subtract_intervals(
    base: tuple[float, float],
    blockers: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Subtract overlapping intervals from a single [start, end] interval."""
    intervals = [base]
    for b_start, b_end in blockers:
        next_intervals: list[tuple[float, float]] = []
        for s, e in intervals:
            if b_end <= s or b_start >= e:
                next_intervals.append((s, e))
            else:
                if b_start > s:
                    next_intervals.append((s, b_start))
                if b_end < e:
                    next_intervals.append((b_end, e))
        intervals = next_intervals
        if not intervals:
            break
    return [(s, e) for s, e in intervals if (e - s) > 0.25]


def _score_window(audio_mono: np.ndarray, sr: int, start: float, end: float) -> float:
    """Score a candidate clean voice-cloning window by energy, non-clipping, and ideal length."""
    s_idx = max(0, int(round(start * sr)))
    e_idx = min(len(audio_mono), int(round(end * sr)))
    if e_idx <= s_idx:
        return -1e9
    clip = audio_mono[s_idx:e_idx]
    duration = (e_idx - s_idx) / sr
    rms = float(np.sqrt(np.mean(clip**2) + 1e-12))
    peak = float(np.max(np.abs(clip)))
    clipping_penalty = 0.5 if peak > 0.985 else 0.0

    # Active speech ratio (frames above small energy floor)
    frame_len = max(1, int(0.05 * sr))
    n_frames = len(clip) // frame_len
    if n_frames > 0:
        frames = clip[: n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        active_ratio = float(np.mean(frame_rms > max(0.008, rms * 0.2)))
    else:
        active_ratio = 1.0

    # Prefer ~10s duration inside [5s, 15s]
    ideal_duration = 10.0
    if CLEAN_CLIP_MIN_SECONDS <= duration <= CLEAN_CLIP_MAX_SECONDS:
        duration_score = 1.5 - abs(duration - ideal_duration) / 10.0
    else:
        duration_score = min(duration / CLEAN_CLIP_MIN_SECONDS, 1.0) * 0.8

    energy_score = min(rms / 0.08, 1.0)
    return duration_score + 0.8 * active_ratio + 0.5 * energy_score - clipping_penalty


def select_cleanest_segment(
    speaker_segments: Sequence[tuple[float, float]],
    audio_mono: np.ndarray,
    sr: int,
    all_speaker_segments: Sequence[tuple[float, float, str]] | None = None,
    speaker_label: str | None = None,
    min_seconds: float = CLEAN_CLIP_MIN_SECONDS,
    max_seconds: float = CLEAN_CLIP_MAX_SECONDS,
) -> tuple[float, float] | None:
    """Identify the cleanest 5-15s non-overlapping segment for voice cloning."""
    if not speaker_segments:
        return None

    other_segments: list[tuple[float, float]] = []
    if all_speaker_segments and speaker_label is not None:
        other_segments = [
            (s, e) for s, e, lbl in all_speaker_segments if lbl != speaker_label
        ]

    # Build non-overlapping candidate spans first
    clean_spans: list[tuple[float, float]] = []
    for seg in speaker_segments:
        if other_segments:
            clean_spans.extend(_subtract_intervals(seg, other_segments))
        else:
            clean_spans.append(seg)

    if not clean_spans:
        clean_spans = list(speaker_segments)

    candidates: list[tuple[float, float]] = []
    for s, e in clean_spans:
        dur = e - s
        if dur <= max_seconds:
            candidates.append((s, e))
        else:
            # Slide a 10s (up to max_seconds) window across long clean spans
            win = min(10.0, max_seconds)
            step = 2.5
            cursor = s
            while cursor + win <= e + 1e-6:
                candidates.append((cursor, min(e, cursor + win)))
                cursor += step
            candidates.append((max(s, e - win), e))

    # Prefer candidates meeting min_seconds if any exist
    in_range = [c for c in candidates if (c[1] - c[0]) >= min_seconds]
    pool = in_range if in_range else candidates
    if not pool:
        return None

    best = max(pool, key=lambda span: _score_window(audio_mono, sr, span[0], span[1]))
    return (round(best[0], 3), round(best[1], 3))


def extract_clean_clip(
    source_wav: Path,
    clean_span: tuple[float, float] | None,
    combined_wav: Path,
    output_wav: Path,
    max_seconds: float = CLEAN_CLIP_MAX_SECONDS,
) -> tuple[float, float | None, float | None]:
    """Extract the cleanest 5-15s segment (or fallback to first <=15s of combined clip)."""
    if clean_span is not None and (clean_span[1] - clean_span[0]) >= CLEAN_CLIP_MIN_SECONDS:
        dur = extract_combined_clip(
            source_wav=source_wav,
            segments=[clean_span],
            output_wav=output_wav,
            silence_gap_seconds=0.0,
        )
        return dur, clean_span[0], clean_span[1]

    # If no single continuous 5s+ segment exists, take up to max_seconds of the combined clip
    # (or the single clean span if combined is identical/shorter)
    data, sr = sf.read(str(combined_wav), dtype="float32", always_2d=True)
    max_samples = int(round(max_seconds * sr))
    sliced = data[:max_samples]
    sliced = _apply_edge_fades(sliced, int(round(CLIP_FADE_SECONDS * sr)))
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_wav), sliced, sr, subtype="PCM_16")
    dur = float(sliced.shape[0] / sr)
    if clean_span is not None and dur <= (clean_span[1] - clean_span[0]) + 0.5:
        return dur, clean_span[0], clean_span[1]
    return dur, clean_span[0] if clean_span else None, clean_span[1] if clean_span else None


def sanitize_filename(name: str, fallback: str = "character") -> str:
    """Convert an arbitrary user-provided name into a safe filesystem slug."""
    normalized = unicodedata.normalize("NFKD", name or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_only).strip("_").lower()
    slug = re.sub(r"_+", "_", slug)
    if not slug:
        slug = fallback
    return slug[:64]


def build_character_filename(
    name: str,
    gender: str,
    age_group: str,
    ext: str = "wav",
    clip_kind: str = "combined",
) -> str:
    """Build a clean, descriptive download filename such as character_1_male_adult.wav."""
    clean_name = sanitize_filename(name, fallback="character")
    clean_gender = sanitize_filename(gender, fallback="unknown")
    clean_age = sanitize_filename(age_group, fallback="unknown")
    clean_ext = ext.lstrip(".").lower()
    if clean_ext not in {"wav", "mp3"}:
        clean_ext = "wav"
    suffix = "_clean" if clip_kind == "clean" else ""
    return f"{clean_name}_{clean_gender}_{clean_age}{suffix}.{clean_ext}"


def create_characters_zip(
    files: Iterable[tuple[Path, str]],
    output_zip: Path,
) -> Path:
    """Bundle character audio files into a ZIP archive with deduplicated archive names."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    seen_names: dict[str, int] = {}

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path, arcname in files:
            if not file_path.exists():
                continue
            base_name = Path(arcname).name
            stem = Path(base_name).stem
            suffix = Path(base_name).suffix
            count = seen_names.get(base_name, 0)
            if count > 0:
                unique_name = f"{stem}_{count + 1}{suffix}"
            else:
                unique_name = base_name
            seen_names[base_name] = count + 1
            zf.write(file_path, arcname=unique_name)

    return output_zip
