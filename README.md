# VoiceCast — Local Audio Character Detector

A local, privacy-first full-stack web application that analyzes uploaded audio files to detect distinct speakers ("characters"), classifies each character's **Gender (`Male` / `Female`)** and **Age Group (`Young` / `Adult` / `Old`)**, lets you preview each character's isolated audio with an interactive waveform player, and downloads individual `.wav`/`.mp3` clips (or a `.zip` bundle of all characters) straight to your local `Downloads` folder.

---

## Architecture & ML Models

- **Speaker Diarization (Primary):** [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1) + [`pyannote/segmentation-3.0`](https://huggingface.co/pyannote/segmentation-3.0)
  - Automatic local fallback (**Silero VAD + SpeechBrain ECAPA-TDNN + Agglomerative Clustering**) if Hugging Face gated access is unavailable.
- **Gender & Age Classification:** [`audeering/wav2vec2-large-robust-24-ft-age-gender`](https://huggingface.co/audeering/wav2vec2-large-robust-24-ft-age-gender)
  - Runs across multiple speech segments per speaker using duration-weighted averaging.
  - Maps continuous age (0–100) + child probability into `Young` (child prediction or age < 22), `Adult` (22–54), and `Old` (≥ 55).
- **Voice Clip Extraction:**
  - Produces both a **Full Combined Voice** clip (smoothly padded/faded across all turns) and a **Cleanest 5–15s Clip** (non-overlapping, high-clarity window ideal for voice cloning tools like VoxCPM2).
- **16 GB RAM / Apple Silicon Optimization:**
  - Accelerates inference on Apple Silicon **MPS** with automatic safe CPU fallback.
  - Runs diarization and age/gender classification **sequentially** with explicit `del`, `gc.collect()`, and `torch.mps.empty_cache()` between stages.

---

## What Was Installed on This Machine

- **System Audited:** macOS 27.2.1 (`arm64` Apple Silicon), 16 GB Unified Memory, 67 GiB free disk space.
- **Already Present:** Homebrew 6.0.10, `ffmpeg` 9.0.2, `ffprobe`, `git` 2.54.0, Node.js v25.9.0, `npm` 11.12.1 (system Python was 3.14.4).
- **Installed During Setup:**
  - **`uv` 0.13.8** via Homebrew (`brew install uv`).
  - **Python 3.11.18** pinned inside project-local `.venv/` via `uv python install 3.11 && uv venv --python 3.11 .venv` (system Python untouched).
  - **Backend Python dependencies** in `.venv/` (`torch 2.7.1`, `torchaudio 2.7.1`, `pyannote-audio 3.4.0`, `transformers 4.57.6`, `speechbrain 1.1.2`, `silero-vad 6.3`, `fastapi`, `uvicorn`, `soundfile`, `pytest`, etc.).
  - **Frontend Node dependencies** in `frontend/node_modules/` (React 18, Vite 6, TypeScript 5.7, Tailwind CSS 3.4, `lucide-react`).
  - **Pre-downloaded Hugging Face Models** cached locally:
    - `pyannote/speaker-diarization-3.1` & `pyannote/segmentation-3.0`
    - `audeering/wav2vec2-large-robust-24-ft-age-gender`

---

## Configuration

Copy `.env.example` to `.env` if setting up on another machine and set your Hugging Face read token:

```bash
cp .env.example .env
# Edit .env and set HF_TOKEN=hf_...
```

> `.env` is excluded via `.gitignore` and never committed or printed to logs.

---

## Running the App

Launch both the FastAPI backend (`http://127.0.0.1:8000`) and the React frontend (`http://127.0.0.1:5173`) with a single command:

```bash
./start.sh
```

Then open **http://127.0.0.1:5173** in your browser.

---

## Running Tests & Verification

### Backend Unit, API, and Real Multi-Speaker E2E Tests

```bash
# Run all tests (including real multi-speaker .mp3 & .m4a inference via macOS `say` + ffmpeg)
.venv/bin/pytest -v -s

# Run fast unit & API tests only
.venv/bin/pytest tests/test_audio_utils.py tests/test_classifier.py tests/test_api.py -v
```

### Frontend Typecheck & Production Build

```bash
cd frontend
npm run typecheck
npm run build
```

---

## Project Structure

```text
.
├── .env                  # Local Hugging Face token (gitignored)
├── .env.example          # Template environment file
├── .gitignore
├── pyproject.toml        # Python 3.11+ uv project & dependency specification
├── start.sh              # One-command launcher for backend + frontend
├── README.md
├── backend/
│   ├── config.py         # Env loading, device selection, memory release helpers
│   ├── models.py         # Pydantic API schemas
│   ├── audio_utils.py    # ffprobe validation, ffmpeg conversion, clip & ZIP utilities
│   ├── diarizer.py       # pyannote 3.1 primary + Silero VAD/SpeechBrain fallback
│   ├── classifier.py     # audeering wav2vec2 gender/age model + bucketing logic
│   ├── pipeline.py       # Sequential memory-managed job runner & persistence
│   └── main.py           # FastAPI routes, streaming, attachment downloads, CORS
├── frontend/
│   └── src/
│       ├── api.ts
│       ├── types.ts
│       ├── App.tsx
│       └── components/   # DropZone, ProcessingPipeline, SummaryBar, CharacterCard, WaveformPlayer
└── tests/
    ├── conftest.py
    ├── test_audio_utils.py
    ├── test_classifier.py
    ├── test_api.py
    └── test_e2e_real_audio.py
```
