"""Shared pytest fixtures for unit, API, and real-audio integration tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def generate_sine_wav(path: Path, duration: float = 2.0, freq: float = 440.0, sr: int = 16000) -> Path:
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    wave = (0.25 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wave, sr, subtype="PCM_16")
    return path


def convert_with_ffmpeg(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), str(dst)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed converting {src} -> {dst}: {res.stderr}")
    return dst


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    return generate_sine_wav(tmp_path / "sample.wav", duration=2.5, freq=330.0, sr=22050)


@pytest.fixture
def sample_mp3(tmp_path: Path, sample_wav: Path) -> Path:
    return convert_with_ffmpeg(sample_wav, tmp_path / "sample.mp3")


@pytest.fixture
def sample_m4a(tmp_path: Path, sample_wav: Path) -> Path:
    return convert_with_ffmpeg(sample_wav, tmp_path / "sample.m4a")


@pytest.fixture
def corrupt_audio(tmp_path: Path) -> Path:
    bad = tmp_path / "corrupt.mp3"
    bad.write_bytes(b"NOT_A_VALID_AUDIO_STREAM_0000111122223333")
    return bad
