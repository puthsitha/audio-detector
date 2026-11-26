"""Unit tests for gender and age-group bucketing and confidence handling."""

from __future__ import annotations

import pytest

from backend.classifier import bucket_age_group, classify_probabilities


@pytest.mark.parametrize(
    ("age", "child_p", "female_p", "male_p", "expected_bucket"),
    [
        # Child prediction overrides adult estimated age
        (28.0, 0.72, 0.16, 0.12, "Young"),
        (35.0, 0.45, 0.30, 0.25, "Young"),
        # Boundary cases around 22 and 55
        (18.0, 0.05, 0.10, 0.85, "Young"),
        (21.9, 0.02, 0.90, 0.08, "Young"),
        (22.0, 0.01, 0.89, 0.10, "Adult"),
        (38.0, 0.01, 0.05, 0.94, "Adult"),
        (54.0, 0.01, 0.10, 0.89, "Adult"),
        (54.9, 0.01, 0.90, 0.09, "Adult"),
        (55.0, 0.01, 0.10, 0.89, "Old"),
        (75.0, 0.00, 0.85, 0.15, "Old"),
    ],
)
def test_age_group_bucketing_boundaries(
    age: float,
    child_p: float,
    female_p: float,
    male_p: float,
    expected_bucket: str,
):
    assert bucket_age_group(age, child_p, female_p, male_p) == expected_bucket


def test_classify_probabilities_gender_and_low_confidence():
    # High-confidence male adult
    res_male = classify_probabilities(
        raw_age_years=34.2,
        female_prob=0.03,
        male_prob=0.96,
        child_prob=0.01,
        age_std_years=2.1,
        total_speech_seconds=12.0,
    )
    assert res_male.gender == "Male"
    assert res_male.gender_confidence > 0.95
    assert res_male.age_group == "Adult"
    assert res_male.estimated_age == pytest.approx(34.2, abs=0.1)

    # High-confidence female senior
    res_female_old = classify_probabilities(
        raw_age_years=66.8,
        female_prob=0.94,
        male_prob=0.05,
        child_prob=0.01,
        age_std_years=3.0,
        total_speech_seconds=10.0,
    )
    assert res_female_old.gender == "Female"
    assert res_female_old.age_group == "Old"
    assert res_female_old.gender_confidence > 0.9

    # Low-confidence ambiguous gender & short speech
    res_low = classify_probabilities(
        raw_age_years=21.8,
        female_prob=0.49,
        male_prob=0.51,
        child_prob=0.0,
        age_std_years=18.0,
        total_speech_seconds=0.8,
    )
    assert res_low.gender == "Male"
    assert res_low.gender_confidence == pytest.approx(0.51, abs=0.02)
    assert res_low.age_group == "Young"
    assert res_low.age_confidence < 0.55

    # Child voice still resolves Male/Female while discounting gender confidence
    res_child = classify_probabilities(
        raw_age_years=12.0,
        female_prob=0.15,
        male_prob=0.05,
        child_prob=0.80,
    )
    assert res_child.age_group == "Young"
    assert res_child.gender == "Female"
    assert res_child.gender_confidence < 0.6
    assert res_child.child_probability == pytest.approx(0.80, abs=0.01)

    # Clamping out-of-range raw ages
    res_clamped = classify_probabilities(raw_age_years=140.0, female_prob=0.0, male_prob=0.0, child_prob=0.0)
    assert res_clamped.estimated_age == 100.0
    assert res_clamped.age_group == "Old"
