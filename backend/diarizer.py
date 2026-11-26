"""Speaker diarization using pyannote/speaker-diarization-3.1 with local Silero+SpeechBrain fallback."""

from __future__ import annotations

import contextlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf
import torch

from backend.config import (
    DIARIZATION_MODEL_ID,
    FALLBACK_EMBEDDING_MODEL_ID,
    MIN_SPEAKER_DURATION_SECONDS,
    PYANNOTE_DIARIZATION_URL,
    PYANNOTE_SEGMENTATION_URL,
    get_hf_token,
    get_preferred_device,
    release_memory,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiarizationResult:
    segments: list[tuple[float, float, str]]  # (start, end, speaker_label)
    engine_name: str
    device_used: str
    fallback_notice: str | None = None


@contextlib.contextmanager
def _allow_pyannote_torch_load() -> Iterator[None]:
    """PyTorch 2.6+ defaults torch.load(weights_only=True/None), which breaks pyannote 3.x checkpoints.

    Scope weights_only=False strictly to trusted pyannote Hugging Face pipeline loading.
    """
    original_load = torch.load
    original_ser_load = torch.serialization.load

    def _compat_load(*args, **kwargs):
        # pytorch-lightning explicitly passes weights_only=None, which PyTorch 2.6+ treats as True
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _compat_load  # type: ignore[assignment]
    torch.serialization.load = _compat_load  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.load = original_load  # type: ignore[assignment]
        torch.serialization.load = original_ser_load  # type: ignore[assignment]


def _is_gated_or_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "403",
        "401",
        "gated",
        "restricted",
        "accept the user conditions",
        "access to model",
        "cannot access gated repo",
        "unauthorized",
    )
    return any(m in msg for m in markers)


def _load_pyannote_pipeline(token: str | None):
    from pyannote.audio import Pipeline

    sig = inspect.signature(Pipeline.from_pretrained)
    kwargs = {}
    if "token" in sig.parameters:
        kwargs["token"] = token
    else:
        kwargs["use_auth_token"] = token

    with _allow_pyannote_torch_load():
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL_ID, **kwargs)

    if pipeline is None:
        raise PermissionError(
            "pyannote/speaker-diarization-3.1 returned None (likely gated repository access not accepted)."
        )
    return pipeline


def _run_pyannote_diarization(audio_wav_path: Path, device: str) -> DiarizationResult:
    token = get_hf_token()
    pipeline = _load_pyannote_pipeline(token)
    active_device = device

    try:
        pipeline.to(torch.device(active_device))
    except Exception as exc:
        logger.warning("Could not move pyannote pipeline to %s (%s); falling back to CPU.", active_device, exc)
        active_device = "cpu"
        pipeline.to(torch.device("cpu"))

    try:
        # Load audio cleanly with soundfile to avoid torchaudio/torchcodec/FFmpeg issues
        audio, sr = sf.read(str(audio_wav_path), dtype="float32")
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        waveform = torch.from_numpy(audio).unsqueeze(0)
        input_payload = {"waveform": waveform, "sample_rate": int(sr)}

        try:
            diarization = pipeline(input_payload)
        except Exception as exc:
            if active_device != "cpu":
                logger.warning("pyannote inference failed on %s (%s); retrying on CPU.", active_device, exc)
                active_device = "cpu"
                pipeline.to(torch.device("cpu"))
                diarization = pipeline(input_payload)
            else:
                raise

        raw_segments: list[tuple[float, float, str]] = []
        # Support both pyannote 3.x Annotation and 4.x DiarizeOutput
        annotation = getattr(diarization, "speaker_diarization", diarization)
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = float(turn.start)
            end = float(turn.end)
            if end > start:
                raw_segments.append((start, end, str(speaker)))

        return DiarizationResult(
            segments=raw_segments,
            engine_name="pyannote 3.1",
            device_used=active_device,
            fallback_notice=None,
        )
    finally:
        del pipeline
        release_memory()


def _run_fallback_diarization(
    audio_wav_path: Path,
    fallback_notice: str,
) -> DiarizationResult:
    """Automatic local fallback: Silero VAD + SpeechBrain ECAPA-TDNN + Agglomerative Clustering."""
    from silero_vad import get_speech_timestamps, load_silero_vad
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from speechbrain.inference.speaker import EncoderClassifier

    audio, sr = sf.read(str(audio_wav_path), dtype="float32")
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        raise ValueError(f"Fallback diarizer requires 16kHz audio, got {sr}Hz.")

    vad_model = load_silero_vad()
    try:
        speech_ts = get_speech_timestamps(
            torch.from_numpy(audio),
            vad_model,
            sampling_rate=16000,
            min_speech_duration_ms=400,
            min_silence_duration_ms=300,
            return_seconds=True,
        )
    finally:
        del vad_model
        release_memory()

    if not speech_ts:
        return DiarizationResult(
            segments=[],
            engine_name="Silero VAD + SpeechBrain ECAPA-TDNN",
            device_used="cpu",
            fallback_notice=fallback_notice,
        )

    # Split long VAD regions into ~2.5s windows for speaker embedding clustering
    windows: list[tuple[float, float]] = []
    for item in speech_ts:
        s = float(item["start"])
        e = float(item["end"])
        if e - s <= 3.0:
            windows.append((s, e))
        else:
            cursor = s
            while cursor < e:
                w_end = min(e, cursor + 2.5)
                if w_end - cursor >= 0.5:
                    windows.append((cursor, w_end))
                cursor = w_end

    if len(windows) == 1:
        return DiarizationResult(
            segments=[(windows[0][0], windows[0][1], "SPEAKER_00")],
            engine_name="Silero VAD + SpeechBrain ECAPA-TDNN",
            device_used="cpu",
            fallback_notice=fallback_notice,
        )

    classifier = EncoderClassifier.from_hparams(
        source=FALLBACK_EMBEDDING_MODEL_ID,
        run_opts={"device": "cpu"},
    )
    try:
        embeddings = []
        valid_windows: list[tuple[float, float]] = []
        for s, e in windows:
            s_idx = max(0, int(round(s * sr)))
            e_idx = min(len(audio), int(round(e * sr)))
            chunk = audio[s_idx:e_idx]
            if len(chunk) < int(0.4 * sr):
                continue
            with torch.no_grad():
                emb = classifier.encode_batch(torch.from_numpy(chunk).unsqueeze(0))
            vec = emb.squeeze().cpu().numpy().astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
            valid_windows.append((s, e))
    finally:
        del classifier
        release_memory()

    if len(embeddings) <= 1:
        segs = [(s, e, "SPEAKER_00") for s, e in valid_windows]
        return DiarizationResult(
            segments=segs,
            engine_name="Silero VAD + SpeechBrain ECAPA-TDNN",
            device_used="cpu",
            fallback_notice=fallback_notice,
        )

    X = np.vstack(embeddings)
    # Check pairwise cosine distances to avoid over-splitting single-speaker audio
    sim_matrix = np.clip(X @ X.T, -1.0, 1.0)
    dist_matrix = 1.0 - sim_matrix
    triu = dist_matrix[np.triu_indices(len(X), k=1)]

    if len(X) == 2:
        labels = [0, 1] if float(triu[0]) > 0.45 else [0, 0]
    elif float(np.percentile(triu, 90)) < 0.38:
        labels = [0] * len(X)
    else:
        best_k = 1
        best_score = 0.18  # minimum silhouette threshold to split speakers
        max_k = min(8, len(X) - 1)
        best_labels = [0] * len(X)
        for k in range(2, max_k + 1):
            clusterer = AgglomerativeClustering(
                n_clusters=k,
                metric="cosine",
                linkage="average",
            )
            cand = clusterer.fit_predict(X)
            if len(set(cand)) < 2:
                continue
            score = float(silhouette_score(X, cand, metric="cosine"))
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = cand.tolist()
        labels = best_labels if best_k >= 2 else [0] * len(X)

    raw_segments = [
        (s, e, f"SPEAKER_{lbl:02d}")
        for (s, e), lbl in zip(valid_windows, labels)
    ]
    return DiarizationResult(
        segments=raw_segments,
        engine_name="Silero VAD + SpeechBrain ECAPA-TDNN",
        device_used="cpu",
        fallback_notice=fallback_notice,
    )


def filter_tiny_speakers(
    segments: list[tuple[float, float, str]],
    min_duration: float = MIN_SPEAKER_DURATION_SECONDS,
) -> list[tuple[float, float, str]]:
    """Filter out spurious ultra-short noise speakers while keeping all valid speakers."""
    totals: dict[str, float] = {}
    for s, e, spk in segments:
        totals[spk] = totals.get(spk, 0.0) + max(0.0, e - s)

    valid = {spk for spk, dur in totals.items() if dur >= min_duration}
    if not valid and totals:
        # Keep the longest speaker if everything is very short
        longest = max(totals, key=lambda k: totals[k])
        valid = {longest}

    return [item for item in segments if item[2] in valid]


def diarize_audio(audio_wav_path: Path, device: str | None = None) -> DiarizationResult:
    """Run primary pyannote 3.1 diarization, falling back cleanly if gated/unavailable."""
    target_device = device or get_preferred_device()

    try:
        result = _run_pyannote_diarization(audio_wav_path, device=target_device)
        filtered = filter_tiny_speakers(result.segments)
        return DiarizationResult(
            segments=filtered,
            engine_name=result.engine_name,
            device_used=result.device_used,
            fallback_notice=None,
        )
    except Exception as exc:
        if _is_gated_or_auth_error(exc):
            notice = (
                "pyannote/speaker-diarization-3.1 requires accepting Hugging Face license terms. "
                f"Accept access at {PYANNOTE_DIARIZATION_URL} and {PYANNOTE_SEGMENTATION_URL} "
                "using the account linked to HF_TOKEN in .env, then restart."
            )
            logger.warning("pyannote gated access error; activating fallback diarizer.")
        else:
            notice = f"pyannote 3.1 was unavailable ({exc}); used local fallback diarizer."
            logger.warning("pyannote error (%s); activating fallback diarizer.", exc)

        fallback_result = _run_fallback_diarization(audio_wav_path, fallback_notice=notice)
        filtered = filter_tiny_speakers(fallback_result.segments)
        return DiarizationResult(
            segments=filtered,
            engine_name=fallback_result.engine_name,
            device_used=fallback_result.device_used,
            fallback_notice=fallback_result.fallback_notice,
        )
