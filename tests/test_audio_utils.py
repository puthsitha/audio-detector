"""Unit tests for audio validation, conversion, segment merging, filename sanitization, and ZIP bundling."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.audio_utils import (
    AudioValidationError,
    build_character_filename,
    convert_to_analysis_wav,
    convert_to_clean_wav,
    create_characters_zip,
    extract_combined_clip,
    merge_and_pad_segments,
    probe_audio,
    sanitize_filename,
    select_cleanest_segment,
    validate_extension,
)


def test_validate_extension_supported_and_rejected():
    for ext in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".webm"):
        assert validate_extension(f"recording{ext}") == ext
        assert validate_extension(f"RECORDING{ext.upper()}") == ext

    for bad in ("script.txt", "video.mp4", "image.png", "no_extension"):
        with pytest.raises(AudioValidationError):
            validate_extension(bad)


def test_probe_and_convert_valid_formats(tmp_path: Path, sample_wav: Path, sample_mp3: Path, sample_m4a: Path):
    for audio_file in (sample_wav, sample_mp3, sample_m4a):
        meta = probe_audio(audio_file)
        assert meta.duration > 2.0
        assert meta.sample_rate > 0

        analysis_wav = convert_to_analysis_wav(audio_file, tmp_path / f"{audio_file.stem}_16k.wav")
        data, sr = sf.read(str(analysis_wav), dtype="float32")
        assert sr == 16000
        assert data.ndim == 1
        assert len(data) > 30000

        clean_wav = convert_to_clean_wav(audio_file, tmp_path / f"{audio_file.stem}_clean.wav")
        assert clean_wav.exists() and clean_wav.stat().st_size > 0


def test_probe_rejects_corrupt_empty_and_overlong(tmp_path: Path, corrupt_audio: Path, sample_wav: Path):
    with pytest.raises(AudioValidationError):
        probe_audio(corrupt_audio)

    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(AudioValidationError):
        probe_audio(empty)

    with pytest.raises(AudioValidationError):
        probe_audio(sample_wav, max_duration_seconds=1.0)


def test_merge_and_pad_segments():
    merged = merge_and_pad_segments(
        segments=[(5.0, 6.0), (1.0, 2.0), (2.2, 3.0), (0.05, 0.3)],
        audio_duration=10.0,
        pad_seconds=0.15,
        merge_gap_seconds=0.35,
    )
    # (0.05, 0.3) -> (0.0, 0.45)
    # (1.0, 2.0) -> (0.85, 2.15) merges with (2.2, 3.0) -> (2.05, 3.15) into (0.85, 3.15)
    # (5.0, 6.0) -> (4.85, 6.15)
    assert len(merged) == 3
    assert merged[0][0] == pytest.approx(0.0, abs=1e-4)
    assert merged[1] == (pytest.approx(0.85, abs=1e-4), pytest.approx(3.15, abs=1e-4))
    assert merged[2] == (pytest.approx(4.85, abs=1e-4), pytest.approx(6.15, abs=1e-4))


def test_extract_combined_clip_and_cleanest_segment(tmp_path: Path):
    sr = 16000
    duration = 20.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    audio = (0.2 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)
    src = tmp_path / "src.wav"
    sf.write(str(src), audio, sr, subtype="PCM_16")

    out_clip = tmp_path / "combined.wav"
    clip_dur = extract_combined_clip(src, [(1.0, 3.0), (5.0, 14.0)], out_clip)
    assert clip_dur > 10.5
    meta = probe_audio(out_clip)
    assert meta.duration > 10.5

    all_segs = [(1.0, 3.0, "SPK_0"), (1.5, 2.5, "SPK_1"), (5.0, 14.0, "SPK_0")]
    best = select_cleanest_segment(
        speaker_segments=[(1.0, 3.0), (5.0, 14.0)],
        audio_mono=audio,
        sr=sr,
        all_speaker_segments=all_segs,
        speaker_label="SPK_0",
    )
    assert best is not None
    assert best[0] >= 5.0 and best[1] <= 14.0
    assert 5.0 <= (best[1] - best[0]) <= 15.0


def test_filename_sanitization_and_building():
    assert sanitize_filename("Narrator (Main) / #1!!") == "narrator_main_1"
    assert sanitize_filename("../../etc/passwd") == "etc_passwd"
    assert sanitize_filename("   ") == "character"

    assert build_character_filename("Character 1", "Male", "Adult", "wav") == "character_1_male_adult.wav"
    assert build_character_filename("Dr. Sarah!", "Female", "Old", "mp3") == "dr_sarah_female_old.mp3"
    assert build_character_filename("Narrator", "Male", "Young", "wav", clip_kind="clean") == "narrator_male_young_clean.wav"


def test_create_characters_zip_deduplicates_names(tmp_path: Path, sample_wav: Path):
    f1 = tmp_path / "c1.wav"
    f2 = tmp_path / "c2.wav"
    f1.write_bytes(sample_wav.read_bytes())
    f2.write_bytes(sample_wav.read_bytes())

    zip_path = tmp_path / "bundle.zip"
    create_characters_zip([(f1, "narrator_male_adult.wav"), (f2, "narrator_male_adult.wav")], zip_path)
    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(zf.namelist())
        assert names == ["narrator_male_adult.wav", "narrator_male_adult_2.wav"]
