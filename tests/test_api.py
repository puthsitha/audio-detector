"""FastAPI endpoint tests with isolated JobManager and mocked ML pipeline."""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.classifier import AgeGenderPrediction
from backend.diarizer import DiarizationResult
from backend.main import create_app
from backend.pipeline import JobManager


def _make_mock_diarization(audio_path: Path):
    return DiarizationResult(
        segments=[(0.1, 1.1, "SPEAKER_00"), (1.2, 2.3, "SPEAKER_01")],
        engine_name="pyannote 3.1",
        device_used="cpu",
        fallback_notice=None,
    )


class _MockClassifier:
    def load(self):
        return None

    def unload(self):
        return None

    def classify_speaker(self, audio_16k, sr, segments):
        if segments[0][0] < 1.0:
            return AgeGenderPrediction(
                gender="Male",
                gender_confidence=0.95,
                age_group="Adult",
                estimated_age=36.0,
                child_probability=0.01,
                age_confidence=0.85,
            )
        return AgeGenderPrediction(
            gender="Female",
            gender_confidence=0.93,
            age_group="Young",
            estimated_age=19.5,
            child_probability=0.12,
            age_confidence=0.80,
        )


def test_health_endpoint(tmp_path: Path):
    manager = JobManager(base_dir=tmp_path / "jobs")
    client = TestClient(create_app(manager=manager))
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert ".mp3" in body["allowed_extensions"]


def test_analyze_rejects_invalid_and_corrupt_files(tmp_path: Path, corrupt_audio: Path):
    manager = JobManager(base_dir=tmp_path / "jobs")
    client = TestClient(create_app(manager=manager))

    # Unsupported extension
    res_ext = client.post(
        "/api/analyze",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert res_ext.status_code == 400

    # Corrupt audio with valid extension
    res_corrupt = client.post(
        "/api/analyze",
        files={"file": ("corrupt.mp3", corrupt_audio.read_bytes(), "audio/mpeg")},
    )
    assert res_corrupt.status_code == 400

    # Missing job returns 404
    assert client.get("/api/jobs/nonexistent").status_code == 404


def test_full_api_flow_analyze_rename_stream_download_zip(tmp_path: Path, sample_wav: Path):
    manager = JobManager(base_dir=tmp_path / "jobs")
    # Execute synchronously in unit tests to avoid sleep/polling flakiness
    manager.submit_job = lambda job_id, path: manager.run_pipeline(job_id, path)  # type: ignore[method-assign]
    client = TestClient(create_app(manager=manager))

    with (
        patch("backend.pipeline.diarize_audio", side_effect=_make_mock_diarization),
        patch("backend.pipeline.AgeGenderClassifier", return_value=_MockClassifier()),
    ):
        res = client.post(
            "/api/analyze",
            files={"file": ("dialogue.wav", sample_wav.read_bytes(), "audio/wav")},
        )
        assert res.status_code == 200
        job_id = res.json()["job_id"]

    job_res = client.get(f"/api/jobs/{job_id}")
    assert job_res.status_code == 200
    job_data = job_res.json()
    assert job_data["status"] == "completed"
    assert job_data["summary"]["total_characters"] == 2
    assert job_data["summary"]["gender_counts"] == {"Male": 1, "Female": 1}
    assert job_data["summary"]["age_group_counts"] == {"Young": 1, "Adult": 1, "Old": 0}
    assert len(job_data["characters"]) == 2

    # Rename character_1 -> Narrator
    rename_res = client.patch(
        f"/api/jobs/{job_id}/characters/character_1",
        json={"name": "Narrator"},
    )
    assert rename_res.status_code == 200
    assert rename_res.json()["name"] == "Narrator"
    assert rename_res.json()["default_filename"] == "narrator_male_adult.wav"

    # Stream audio inline
    stream_res = client.get(f"/api/jobs/{job_id}/audio/character_1")
    assert stream_res.status_code == 200
    assert stream_res.headers["content-type"].startswith("audio/wav")
    assert len(stream_res.content) > 100

    # Download WAV with attachment header reflecting renamed character
    dl_wav = client.get(f"/api/jobs/{job_id}/download/character_1?format=wav")
    assert dl_wav.status_code == 200
    assert "attachment" in dl_wav.headers["content-disposition"]
    assert "narrator_male_adult.wav" in dl_wav.headers["content-disposition"]
    assert len(dl_wav.content) > 100

    # Download MP3 with attachment header
    dl_mp3 = client.get(f"/api/jobs/{job_id}/download/character_1?format=mp3")
    assert dl_mp3.status_code == 200
    assert "attachment" in dl_mp3.headers["content-disposition"]
    assert "narrator_male_adult.mp3" in dl_mp3.headers["content-disposition"]
    assert len(dl_mp3.content) > 100

    # Download All ZIP
    dl_zip = client.get(f"/api/jobs/{job_id}/download-all?format=wav")
    assert dl_zip.status_code == 200
    assert "attachment" in dl_zip.headers["content-disposition"]
    assert ".zip" in dl_zip.headers["content-disposition"]
    with zipfile.ZipFile(BytesIO(dl_zip.content), "r") as zf:
        names = sorted(zf.namelist())
        assert "narrator_male_adult.wav" in names
        assert "character_2_female_young.wav" in names
