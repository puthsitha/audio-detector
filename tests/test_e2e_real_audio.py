"""Real multi-speaker audio end-to-end integration test using macOS `say` + ffmpeg + real ML models."""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.audio_utils import probe_audio
from backend.main import create_app
from backend.pipeline import JobManager


def _synthesize_dialogue_fixtures(out_dir: Path) -> tuple[Path, Path]:
    """Synthesize a realistic alternating male/female multi-speaker fixture via macOS `say` + `ffmpeg`."""
    if shutil.which("say") is None:
        pytest.skip("macOS `say` command is not available on this platform.")

    out_dir.mkdir(parents=True, exist_ok=True)
    turns = [
        (
            "Daniel",
            "Good morning everyone. Today we are reviewing the quarterly engineering roadmap and discussing the new local audio pipeline.",
        ),
        (
            "Samantha",
            "Thanks Daniel. I tested the acoustic benchmarks yesterday afternoon, and the speaker separation quality looks remarkably clear across every recording.",
        ),
        (
            "Daniel",
            "That is great news. Let us also verify that each isolated voice clip exports cleanly with smooth transitions and zero clipping.",
        ),
        (
            "Samantha",
            "Agreed. Once we confirm the download archive and playback controls, we will be ready to ship the local web application.",
        ),
    ]

    wav_parts: list[Path] = []
    silence_wav = out_dir / "silence.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=22050:cl=mono",
            "-t",
            "0.7",
            "-c:a",
            "pcm_s16le",
            str(silence_wav),
        ],
        check=True,
    )

    for idx, (voice, text) in enumerate(turns):
        aiff_path = out_dir / f"turn_{idx}.aiff"
        wav_path = out_dir / f"turn_{idx}.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff_path), text], check=True)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(aiff_path),
                "-ac",
                "1",
                "-ar",
                "22050",
                "-c:a",
                "pcm_s16le",
                str(wav_path),
            ],
            check=True,
        )
        wav_parts.append(wav_path)
        if idx < len(turns) - 1:
            wav_parts.append(silence_wav)

    concat_list = out_dir / "concat.txt"
    concat_list.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in wav_parts),
        encoding="utf-8",
    )

    mp3_path = out_dir / "dialogue_multi_speaker.mp3"
    m4a_path = out_dir / "dialogue_multi_speaker.m4a"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-codec:a",
            "libmp3lame",
            "-qscale:a",
            "2",
            str(mp3_path),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-codec:a",
            "aac",
            "-b:a",
            "160k",
            str(m4a_path),
        ],
        check=True,
    )
    return mp3_path, m4a_path


@pytest.mark.parametrize("ext", ["mp3", "m4a"])
def test_real_audio_end_to_end_pipeline(tmp_path: Path, ext: str):
    """Run real diarization + wav2vec2 age/gender inference on synthesized multi-speaker fixtures."""
    mp3_path, m4a_path = _synthesize_dialogue_fixtures(tmp_path / "fixtures")
    fixture_path = mp3_path if ext == "mp3" else m4a_path
    mime_type = "audio/mpeg" if ext == "mp3" else "audio/mp4"

    manager = JobManager(base_dir=tmp_path / "jobs")
    # Run synchronously inside the request so test completes deterministically
    manager.submit_job = lambda job_id, path: manager.run_pipeline(job_id, path)  # type: ignore[method-assign]
    client = TestClient(create_app(manager=manager))

    res = client.post(
        "/api/analyze",
        files={"file": (fixture_path.name, fixture_path.read_bytes(), mime_type)},
    )
    assert res.status_code == 200, res.text
    job_id = res.json()["job_id"]

    job_res = client.get(f"/api/jobs/{job_id}")
    assert job_res.status_code == 200
    job = job_res.json()
    assert job["status"] == "completed", f"Job failed: {job.get('error')}"

    summary = job["summary"]
    characters = job["characters"]
    print(
        f"\n[E2E {ext.upper()}] engine={summary['diarization_engine']} "
        f"device={summary['device_used']} total_characters={summary['total_characters']} "
        f"genders={summary['gender_counts']} ages={summary['age_group_counts']}"
    )
    for c in characters:
        print(
            f"  - {c['name']} ({c['speaker_label']}): {c['gender']} ({c['gender_confidence']:.2f}), "
            f"{c['age_group']} (~{c['estimated_age']}y), dur={c['total_speaking_duration']}s"
        )

    assert summary["total_characters"] >= 2
    genders = {c["gender"] for c in characters}
    assert "Male" in genders, f"Expected a Male speaker, got {characters}"
    assert "Female" in genders, f"Expected a Female speaker, got {characters}"

    # Verify extracted clips exist, pass ffprobe validation, and download properly
    for char in characters:
        cid = char["id"]
        combined_wav = manager.job_dir(job_id) / "clips" / f"{cid}_combined.wav"
        clean_wav = manager.job_dir(job_id) / "clips" / f"{cid}_clean.wav"
        assert combined_wav.exists() and combined_wav.stat().st_size > 1000
        assert clean_wav.exists() and clean_wav.stat().st_size > 1000
        assert probe_audio(combined_wav).duration > 0.5
        assert probe_audio(clean_wav).duration > 0.5

        dl = client.get(f"/api/jobs/{job_id}/download/{cid}?format=wav")
        assert dl.status_code == 200
        assert "attachment" in dl.headers["content-disposition"]
        downloaded_file = tmp_path / f"downloaded_{cid}.wav"
        downloaded_file.write_bytes(dl.content)
        assert probe_audio(downloaded_file).duration > 0.5

    # Verify Download All ZIP
    zip_res = client.get(f"/api/jobs/{job_id}/download-all?format=wav")
    assert zip_res.status_code == 200
    assert "attachment" in zip_res.headers["content-disposition"]
    with zipfile.ZipFile(BytesIO(zip_res.content), "r") as zf:
        assert len(zf.namelist()) == len(characters)
