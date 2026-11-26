"""FastAPI application for the Local Audio Character Detector."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from backend.audio_utils import (
    AudioValidationError,
    build_character_filename,
    create_characters_zip,
    probe_audio,
    transcode_wav_to_mp3,
    validate_extension,
)
from backend.config import (
    ALLOWED_AUDIO_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    get_preferred_device,
)
from backend.models import (
    AnalyzeResponse,
    CharacterResult,
    JobState,
    RenameCharacterRequest,
)
from backend.pipeline import JobManager, job_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_app(manager: JobManager | None = None) -> FastAPI:
    active_manager = manager or job_manager

    app = FastAPI(
        title="VoiceCast — Local Audio Character Detector",
        version="1.0.0",
        description="Local speaker diarization, gender/age classification, and character clip export.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "device": get_preferred_device(),
            "allowed_extensions": sorted(ALLOWED_AUDIO_EXTENSIONS),
            "max_file_size_mb": MAX_FILE_SIZE_MB,
        }

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze_audio(file: UploadFile = File(...)) -> AnalyzeResponse:
        filename = file.filename or "upload"
        try:
            ext = validate_extension(filename)
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        job = active_manager.create_job(original_filename=filename)
        jdir = active_manager.job_dir(job.job_id)
        jdir.mkdir(parents=True, exist_ok=True)
        upload_path = jdir / f"input{ext}"

        bytes_written = 0
        try:
            with upload_path.open("wb") as out_f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > MAX_FILE_SIZE_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds the maximum allowed size of {MAX_FILE_SIZE_MB} MB.",
                        )
                    out_f.write(chunk)

            if bytes_written == 0:
                raise AudioValidationError("Uploaded file is empty.")

            # Validate immediately with ffprobe so corrupted files get an immediate 400 error
            probe_audio(upload_path)
        except HTTPException:
            shutil.rmtree(jdir, ignore_errors=True)
            raise
        except AudioValidationError as exc:
            shutil.rmtree(jdir, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            await file.close()

        active_manager.submit_job(job.job_id, upload_path)
        return AnalyzeResponse(
            job_id=job.job_id,
            status="queued",
            stage="uploading",
            message="File accepted and queued for analysis.",
        )

    @app.get("/api/jobs/{job_id}", response_model=JobState)
    def get_job_status(job_id: str) -> JobState:
        state = active_manager.get_job(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return state

    @app.patch("/api/jobs/{job_id}/characters/{character_id}", response_model=CharacterResult)
    def rename_character_endpoint(
        job_id: str,
        character_id: str,
        body: RenameCharacterRequest,
    ) -> CharacterResult:
        updated = active_manager.rename_character(job_id, character_id, body.name)
        if updated is None:
            raise HTTPException(status_code=404, detail="Job or character not found.")
        return updated

    def _resolve_clip_wav(job_id: str, character_id: str, clip: str) -> tuple[CharacterResult, Path]:
        state = active_manager.get_job(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        char = next((c for c in state.characters if c.id == character_id), None)
        if char is None:
            raise HTTPException(status_code=404, detail="Character not found.")

        suffix = "clean" if clip == "clean" else "combined"
        wav_path = active_manager.job_dir(job_id) / "clips" / f"{character_id}_{suffix}.wav"
        if not wav_path.exists():
            raise HTTPException(status_code=404, detail="Audio clip file not found.")
        return char, wav_path

    @app.get("/api/jobs/{job_id}/audio/{character_id}")
    def stream_character_audio(
        job_id: str,
        character_id: str,
        clip: Literal["combined", "clean"] = Query(default="combined"),
    ) -> FileResponse:
        _, wav_path = _resolve_clip_wav(job_id, character_id, clip)
        return FileResponse(path=str(wav_path), media_type="audio/wav")

    @app.get("/api/jobs/{job_id}/download/{character_id}")
    def download_character_audio(
        job_id: str,
        character_id: str,
        format: Literal["wav", "mp3"] = Query(default="wav"),
        clip: Literal["combined", "clean"] = Query(default="combined"),
        name: str | None = Query(default=None),
    ) -> FileResponse:
        char, wav_path = _resolve_clip_wav(job_id, character_id, clip)
        effective_name = name.strip() if name and name.strip() else char.name
        download_filename = build_character_filename(
            name=effective_name,
            gender=char.gender,
            age_group=char.age_group,
            ext=format,
            clip_kind=clip,
        )

        if format == "mp3":
            mp3_path = wav_path.with_suffix(".mp3")
            if not mp3_path.exists():
                transcode_wav_to_mp3(wav_path, mp3_path)
            target_path = mp3_path
            media_type = "audio/mpeg"
        else:
            target_path = wav_path
            media_type = "audio/wav"

        return FileResponse(
            path=str(target_path),
            media_type=media_type,
            filename=download_filename,
            headers={
                "Content-Disposition": f'attachment; filename="{download_filename}"',
            },
        )

    @app.get("/api/jobs/{job_id}/download-all")
    def download_all_characters(
        job_id: str,
        format: Literal["wav", "mp3"] = Query(default="wav"),
        clip: Literal["combined", "clean"] = Query(default="combined"),
    ) -> FileResponse:
        state = active_manager.get_job(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if not state.characters:
            raise HTTPException(status_code=400, detail="No character clips available to download.")

        files_to_zip: list[tuple[Path, str]] = []
        suffix = "clean" if clip == "clean" else "combined"
        clips_dir = active_manager.job_dir(job_id) / "clips"

        for char in state.characters:
            wav_path = clips_dir / f"{char.id}_{suffix}.wav"
            if not wav_path.exists():
                continue
            if format == "mp3":
                mp3_path = wav_path.with_suffix(".mp3")
                if not mp3_path.exists():
                    transcode_wav_to_mp3(wav_path, mp3_path)
                file_path = mp3_path
            else:
                file_path = wav_path

            arcname = build_character_filename(
                name=char.name,
                gender=char.gender,
                age_group=char.age_group,
                ext=format,
                clip_kind=clip,
            )
            files_to_zip.append((file_path, arcname))

        if not files_to_zip:
            raise HTTPException(status_code=404, detail="No character clip files found on disk.")

        zip_name = f"voicecast_characters_{job_id}_{format}.zip"
        zip_path = active_manager.job_dir(job_id) / zip_name
        create_characters_zip(files_to_zip, zip_path)

        return FileResponse(
            path=str(zip_path),
            media_type="application/zip",
            filename=zip_name,
            headers={
                "Content-Disposition": f'attachment; filename="{zip_name}"',
            },
        )

    return app


app = create_app()
