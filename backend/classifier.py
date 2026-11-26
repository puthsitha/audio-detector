"""Gender and Age classification using audeering/wav2vec2-large-robust-24-ft-age-gender."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from backend.config import (
    AGE_GENDER_MODEL_ID,
    get_hf_token,
    get_preferred_device,
    release_memory,
)
from backend.models import AgeGroupLabel, GenderLabel

logger = logging.getLogger(__name__)

# Official audeering/wav2vec2-large-robust-24-ft-age-gender output order:
# logits_gender -> [female, male, child]
# Verified directly against the model's Hugging Face config/model card.
GENDER_IDX_FEMALE = 0
GENDER_IDX_MALE = 1
GENDER_IDX_CHILD = 2


@dataclass(frozen=True)
class AgeGenderPrediction:
    gender: GenderLabel
    gender_confidence: float
    age_group: AgeGroupLabel
    estimated_age: float
    child_probability: float
    age_confidence: float


def bucket_age_group(
    estimated_age: float,
    child_prob: float,
    female_prob: float,
    male_prob: float,
) -> AgeGroupLabel:
    """Map predicted age (0-100) and child probability to Young / Adult / Old.

    Rules:
    - Young: Child prediction (child is top class or child_prob >= 0.45) OR predicted age < 22
    - Adult: predicted age 22-54 (i.e., 22 <= age < 55)
    - Old: predicted age >= 55
    """
    is_child_prediction = (
        child_prob >= 0.45
        or (child_prob > female_prob and child_prob > male_prob and child_prob >= 0.34)
    )
    if is_child_prediction or estimated_age < 22.0:
        return "Young"
    if estimated_age < 55.0:
        return "Adult"
    return "Old"


def classify_probabilities(
    raw_age_years: float,
    female_prob: float,
    male_prob: float,
    child_prob: float,
    age_std_years: float = 0.0,
    total_speech_seconds: float = 5.0,
) -> AgeGenderPrediction:
    """Convert raw model probabilities and age estimate into structured labels + confidences."""
    estimated_age = float(np.clip(raw_age_years, 0.0, 100.0))

    # Clamp and normalize probabilities cleanly
    f_p = max(0.0, float(female_prob))
    m_p = max(0.0, float(male_prob))
    c_p = max(0.0, float(child_prob))
    total_p = f_p + m_p + c_p
    if total_p > 0.0:
        f_p /= total_p
        m_p /= total_p
        c_p /= total_p
    else:
        f_p, m_p, c_p = 0.5, 0.5, 0.0

    gender: GenderLabel = "Male" if m_p >= f_p else "Female"
    adult_pair_sum = m_p + f_p
    if adult_pair_sum > 1e-6:
        pair_margin = max(m_p, f_p) / adult_pair_sum
        # Discount gender confidence proportionally when child probability is high
        gender_confidence = float(np.clip(pair_margin * (1.0 - 0.5 * c_p), 0.0, 1.0))
    else:
        gender_confidence = 0.5

    age_group = bucket_age_group(
        estimated_age=estimated_age,
        child_prob=c_p,
        female_prob=f_p,
        male_prob=m_p,
    )

    # Calibrated heuristic confidence for age estimate based on speech length & cross-segment agreement
    duration_factor = float(np.clip(total_speech_seconds / 8.0, 0.35, 1.0))
    consistency_factor = float(np.clip(1.0 - (age_std_years / 35.0), 0.4, 1.0))
    # Slightly higher confidence when well away from a bucket boundary (22 / 55)
    dist_to_boundary = min(abs(estimated_age - 22.0), abs(estimated_age - 55.0))
    boundary_factor = float(np.clip(0.75 + 0.25 * min(dist_to_boundary / 10.0, 1.0), 0.75, 1.0))
    age_confidence = float(
        np.clip(0.92 * duration_factor * consistency_factor * boundary_factor, 0.25, 0.95)
    )

    return AgeGenderPrediction(
        gender=gender,
        gender_confidence=round(gender_confidence, 4),
        age_group=age_group,
        estimated_age=round(estimated_age, 1),
        child_probability=round(c_p, 4),
        age_confidence=round(age_confidence, 4),
    )


class ModelHead(nn.Module):
    """Classification / regression head matching audeering's wav2vec2 age-gender checkpoint."""

    def __init__(self, config, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    """Speech emotion/demographics model predicting age (0..1) and gender logits [female, male, child]."""

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
        self, input_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)
        return hidden_states, logits_age, logits_gender


def _prepare_chunks_for_speaker(
    audio_16k: np.ndarray,
    sr: int,
    segments: Sequence[tuple[float, float]],
    max_chunk_seconds: float = 12.0,
    min_chunk_seconds: float = 0.4,
    max_chunks: int = 12,
) -> list[tuple[np.ndarray, float]]:
    """Slice speaker segments into bounded chunks for memory-safe duration-weighted inference."""
    chunks: list[tuple[np.ndarray, float]] = []
    total_samples = len(audio_16k)

    for start, end in segments:
        dur = end - start
        if dur < min_chunk_seconds:
            continue
        cursor = start
        while cursor < end - 1e-3:
            chunk_end = min(end, cursor + max_chunk_seconds)
            if chunk_end - cursor >= min_chunk_seconds:
                s_idx = max(0, min(total_samples, int(round(cursor * sr))))
                e_idx = max(0, min(total_samples, int(round(chunk_end * sr))))
                if e_idx - s_idx >= int(min_chunk_seconds * sr):
                    chunks.append((audio_16k[s_idx:e_idx], (e_idx - s_idx) / sr))
            cursor = chunk_end

    # Fallback if all segments were shorter than min_chunk_seconds
    if not chunks and segments:
        pieces = []
        for start, end in segments:
            s_idx = max(0, min(total_samples, int(round(start * sr))))
            e_idx = max(0, min(total_samples, int(round(end * sr))))
            if e_idx > s_idx:
                pieces.append(audio_16k[s_idx:e_idx])
        if pieces:
            merged = np.concatenate(pieces)[: int(max_chunk_seconds * sr)]
            chunks.append((merged, len(merged) / sr))

    # Keep the longest/most informative chunks up to max_chunks
    if len(chunks) > max_chunks:
        chunks.sort(key=lambda item: item[1], reverse=True)
        chunks = chunks[:max_chunks]

    return chunks


class AgeGenderClassifier:
    """Wrapper managing loading, MPS/CPU inference, duration-weighted aggregation, and cleanup."""

    def __init__(self, model_id: str = AGE_GENDER_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or get_preferred_device()
        self.processor: Wav2Vec2Processor | None = None
        self.model: AgeGenderModel | None = None

    def load(self) -> None:
        token = get_hf_token()
        kwargs = {"token": token} if token else {}
        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id, **kwargs)
        self.model = AgeGenderModel.from_pretrained(self.model_id, **kwargs)
        self.model.eval()
        try:
            self.model.to(self.device)
        except Exception as exc:
            logger.warning("Failed to move AgeGenderModel to %s (%s); falling back to CPU.", self.device, exc)
            self.device = "cpu"
            self.model.to("cpu")

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        release_memory()

    def _predict_single_chunk(self, chunk: np.ndarray, sr: int = 16000) -> tuple[float, float, float, float]:
        assert self.processor is not None and self.model is not None
        processed = self.processor(chunk, sampling_rate=sr)
        input_array = np.asarray(processed["input_values"], dtype=np.float32)
        input_values = torch.from_numpy(input_array)

        def _forward_on(dev: str) -> tuple[torch.Tensor, torch.Tensor]:
            with torch.no_grad():
                _, logits_age, probs_gender = self.model(input_values.to(dev))
            return logits_age.detach().cpu(), probs_gender.detach().cpu()

        try:
            logits_age, probs_gender = _forward_on(self.device)
        except Exception as exc:
            if self.device != "cpu":
                logger.warning("MPS inference failed in AgeGenderModel (%s); falling back to CPU.", exc)
                self.device = "cpu"
                self.model.to("cpu")
                logits_age, probs_gender = _forward_on("cpu")
            else:
                raise

        age_years = float(logits_age[0, 0].item()) * 100.0
        probs = probs_gender[0].numpy()
        female_prob = float(probs[GENDER_IDX_FEMALE])
        male_prob = float(probs[GENDER_IDX_MALE])
        child_prob = float(probs[GENDER_IDX_CHILD])
        return age_years, female_prob, male_prob, child_prob

    def classify_speaker(
        self,
        audio_16k: np.ndarray,
        sr: int,
        segments: Sequence[tuple[float, float]],
    ) -> AgeGenderPrediction:
        """Predict duration-weighted gender and age group across a speaker's segments."""
        if self.model is None or self.processor is None:
            self.load()

        chunks = _prepare_chunks_for_speaker(audio_16k, sr, segments)
        if not chunks:
            return classify_probabilities(30.0, 0.5, 0.5, 0.0, age_std_years=15.0, total_speech_seconds=0.0)

        ages: list[float] = []
        females: list[float] = []
        males: list[float] = []
        children: list[float] = []
        weights: list[float] = []

        for chunk_audio, duration in chunks:
            age_y, f_p, m_p, c_p = self._predict_single_chunk(chunk_audio, sr=sr)
            ages.append(age_y)
            females.append(f_p)
            males.append(m_p)
            children.append(c_p)
            weights.append(max(duration, 0.1))

        w = np.array(weights, dtype=np.float64)
        w_sum = float(np.sum(w))
        avg_age = float(np.sum(np.array(ages) * w) / w_sum)
        avg_female = float(np.sum(np.array(females) * w) / w_sum)
        avg_male = float(np.sum(np.array(males) * w) / w_sum)
        avg_child = float(np.sum(np.array(children) * w) / w_sum)
        age_std = float(np.std(ages)) if len(ages) > 1 else 0.0

        return classify_probabilities(
            raw_age_years=avg_age,
            female_prob=avg_female,
            male_prob=avg_male,
            child_prob=avg_child,
            age_std_years=age_std,
            total_speech_seconds=w_sum,
        )
