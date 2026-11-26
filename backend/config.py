"""Application configuration loaded safely from environment variables."""

from __future__ import annotations

import gc
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ALLOWED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".webm"}
)

MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "200"))
MAX_FILE_SIZE_BYTES: int = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS: float = float(
    os.getenv("MAX_AUDIO_DURATION_SECONDS", "3600")
)

_raw_job_dir = os.getenv("JOB_OUTPUT_DIR", "backend/jobs")
JOB_OUTPUT_DIR: Path = (
    Path(_raw_job_dir)
    if Path(_raw_job_dir).is_absolute()
    else (PROJECT_ROOT / _raw_job_dir)
)

DIARIZATION_MODEL_ID: str = "pyannote/speaker-diarization-3.1"
SEGMENTATION_MODEL_ID: str = "pyannote/segmentation-3.0"
AGE_GENDER_MODEL_ID: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"
FALLBACK_EMBEDDING_MODEL_ID: str = "speechbrain/spkrec-ecapa-voxceleb"

PYANNOTE_DIARIZATION_URL: str = f"https://huggingface.co/{DIARIZATION_MODEL_ID}"
PYANNOTE_SEGMENTATION_URL: str = f"https://huggingface.co/{SEGMENTATION_MODEL_ID}"

SEGMENT_PADDING_SECONDS: float = 0.15
SEGMENT_MERGE_GAP_SECONDS: float = 0.35
CLIP_FADE_SECONDS: float = 0.02
CLIP_SILENCE_GAP_SECONDS: float = 0.12
CLEAN_CLIP_MIN_SECONDS: float = 5.0
CLEAN_CLIP_MAX_SECONDS: float = 15.0
MIN_SPEAKER_DURATION_SECONDS: float = 0.4


def get_hf_token() -> str | None:
    """Return the Hugging Face token from the environment without logging it."""
    token = os.getenv("HF_TOKEN", "").strip()
    if not token or token == "your_huggingface_token_here":
        return None
    return token


def get_preferred_device() -> str:
    """Return 'mps' on Apple Silicon if available, otherwise 'cpu'."""
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def release_memory() -> None:
    """Release Python and Apple Silicon MPS memory between sequential ML stages."""
    gc.collect()
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
