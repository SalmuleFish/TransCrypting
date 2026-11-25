# TransCrypting — 100% Local Multi-Source Study Workspace, Audio Transcription, Podcast Studio & Exam Prep

A privacy-first Windows desktop workspace built to run **100% offline on your own PC**. Neither your class recordings, PDFs, slides, images, nor AI conversations ever leave your machine. Ideal for organizing courses into local notebooks: transcribe lectures, index documents and whiteboard photos in a persistent local search index, chat with exact cross-source citations, generate two-host study podcasts, and practice with interactive exams.

---

## Screenshots

### Editorial Studio & Unified General Summary (Light Mode)
Distraction-free top command bar, collapsible Conversations and Sources rails, unified gradient summary across active sources, and clickable `▶ MM:SS` audio & source citations.

![Unified Editorial Studio summary in Light Mode](docs/screenshots/summary-light.png)

### Contextual Chat & Cross-Source Citations (Graphite Dark Mode)
Source-grounded Q&A with clickable timestamp/page/image pills, dynamic AI follow-up suggestions, and an on-demand GPU/RAM/Context telemetry popover in the top bar.

![Contextual chat with interactive citations in Dark Mode](docs/screenshots/chat-dark.png)

### Customizable Exam Studio
Pick exact sources and topic focus, choose question count (`3–15`), format (`Mixed`, `MCQ`, `True / False`, `Open`), difficulty, question style, hints, countdown timer, and Real Exam Mode.

![Customizable Exam Studio in Dark Mode](docs/screenshots/exam-dark.png)

### 100% Local Podcast Studio (Kokoro-82M ONNX)
Generate two-host study dialogues completely offline with custom focus/style, exclusive resource locking with instant cancellation, audio player, and full episode script.

![Local Podcast Studio with Kokoro-82M ONNX in Dark Mode](docs/screenshots/podcast-dark.png)

### Modular Local Models & Built-In Guide
Flexible compatibility with any Ollama LLM, Whisper / CTranslate2 model, and Kokoro-82M / Piper local TTS weights directly from **Settings**.

![Built-in guide for modular local models](docs/screenshots/models-guide.png)

---

## Key Features

- **Multi-Source Local Context (`SQLite FTS5`)**:
  - Combine **Audio recordings** (`.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.mp4`, etc.), **Documents** (`.pdf`, `.pptx` slides + speaker notes, `.docx`, `.txt`, `.md`), and **Images** (`.png`, `.jpg`, `.webp`, `.bmp`) inside the same conversation.
  - CPU document extraction (`pypdf`, `python-pptx`, `python-docx`) consumes **0 MB of VRAM** and preserves page/slide locators (`p. 4`, `diap. 2`, `sec. 1`).
  - Images are analyzed **once on upload** using local Ollama vision (or automatic local Windows CPU OCR fallback) and stored in the persistent `SQLite FTS5` index so follow-up chats never reload raw images into VRAM.
  - Every item in the **Sources** panel has a checkbox to enable or exclude it from context at any time.
- **100% Local Podcast Studio (Exclusive Performance Mode)**:
  - Turn selected sources into a two-host **study podcast** (`Conversación`, `Repaso de examen`, `Debate`, or `Paso a paso`, in Spanish or English) with custom topic/focus instructions.
  - Synthesizes expressive speech **100% locally** using **Kokoro-82M ONNX** neural voices (with automatic **Piper ONNX** and Windows `System.Speech` offline fallbacks) and saves replayable `.wav` episodes + full dialogue scripts.
  - **Exclusive Resource Lock & Instant Cancel**: generating a podcast automatically unloads Ollama from VRAM after writing the script before loading TTS voices, locks tab switching/competing AI tasks until finished, and supports instant **Cancel** at any moment.
- **Interactive Cross-Source Citations & Viewer**:
  - `▶ MM:SS` pills jump straight to the exact second of the matching audio recording.
  - `📄 file.pdf · p. 4` and `🖼 image.png` pills open the integrated **Source Viewer** at that exact page, slide, or whiteboard photo.
- **Local Audio Transcription (`faster-whisper`)**:
  - NVIDIA CUDA acceleration (`int8_float16`) with automatic CPU fallback, bilingual segment detection, and optional `ffmpeg` classroom noise leveling.
- **Interactive Exam Mode**: Multiple-choice questions with instant grading and open-ended responses evaluated locally against indexed source excerpts.
- **Smart Memory Management**: Whisper, Ollama, and Podcast TTS take exclusive turns cleanly, with configurable idle VRAM unloading and an instant **Liberar VRAM** button.
- **Editorial Studio UI in Light & Graphite Dark Modes**: Collapsible Conversations & Sources panels, compact top-bar Telemetry popover, single-card gradient general summary, dedicated verbatim Audio Transcripts tab, surgical DOM updates without UI flicker, native `.md` export, and **Borrar historial** that preserves all sources and summaries.

---

## Requirements

- **Windows 10 / 11** (64-bit)
- **Python 3.11 or 3.12**
- **Ollama** installed locally ([ollama.com](https://ollama.com)) with at least one local model pulled (e.g., `ollama pull gemma4:e4b` or `ollama pull qwen3:8b`)
- **ffmpeg** available on system `PATH`
- *(Optional)* NVIDIA CUDA-compatible GPU for accelerated Whisper and Ollama inference

---

## Quick Setup

1. Clone the repository and enter the project folder:
   ```powershell
   git clone https://github.com/SalmuleFish/TransCrypting.git
   cd TransCrypting
   ```
2. Run **`setup.bat`** (double-click or run from PowerShell/CMD). It creates `.venv`, installs dependencies from `requirements.txt`, and compiles **`TransCrypting.exe`** with the `TC` icon.
3. Pull an Ollama model if you have not yet:
   ```powershell
   ollama pull gemma4:e4b
   ```

---

## Usage

### Launching the App
- Double-click **`TransCrypting.exe`** (uses `pythonw.exe` + Windows AppUserModelID for clean taskbar integration).
- Or run via terminal:
  - Desktop window (WebView2): `.\.venv\Scripts\python.exe run.py`
  - Local browser: `.\.venv\Scripts\python.exe run.py --browser`
  - Headless API only: `.\.venv\Scripts\python.exe run.py --headless`

### Modular Models (Bring Your Own Models)
Open **Settings** (gear icon) and click **`¿Cómo instalar modelos?`** for the built-in copyable command guide:
- **Ollama LLMs (Summaries / Chat / Exams / Podcast Scripts)**: pull any model with `ollama pull <model>` (`llama3.2:3b`, `gemma4:e4b`, `qwen3:8b`, `qwen3.5:9b`, etc.) and click **Actualizar lista**. For image vision analysis, choose a vision-capable model such as `gemma4:e4b`.
- **Whisper (Audio Transcription)**: switch between `tiny`, `base`, `small`, `medium`, `large-v3-turbo`, and `large-v3`, or drop any custom **CTranslate2** folder (`model.bin` + `config.json`) inside `data/models/whisper/<your-model>/`.
- **Kokoro-82M / Piper TTS Voices (Podcasts)**: **Kokoro-82M ONNX** weights (`kokoro-v1.0.onnx` and `voices-v1.0.bin`) are downloaded once on demand into `data/models/tts/kokoro/` (with Piper fallback in `data/models/tts/`).

---

## Project Structure & Git Privacy

```text
TransCrypting/
├── backend/                # FastAPI API, exclusive GPU/RAM lock, Whisper, Ollama, FTS5 & Podcast TTS
├── docs/screenshots/       # README showcase screenshots (demo data only)
├── frontend/               # 100% offline Bento UI (HTML/CSS/JS + icon.ico)
├── launcher/               # C# source compiled into TransCrypting.exe
├── tests/                  # Automated pytest suite
├── data/                   # LOCAL ONLY (ignored by Git; only .gitkeep is tracked)
│   ├── audio/              # Uploaded audio recordings
│   ├── sources/            # Uploaded PDFs, slides, docs & images
│   ├── podcasts/           # Locally generated podcast .wav episodes
│   ├── models/             # Whisper & Kokoro-82M / Piper TTS local model caches
│   └── transcript.db       # SQLite database + FTS5 persistent search index
├── .gitignore              # Excludes data/, .venv/, caches, weights & local executables
├── requirements.txt
├── setup.bat               # Windows 1-click setup & launcher compiler
└── run.py
```

All user recordings, uploaded documents, images, generated podcasts, model caches, databases, and `.venv/` stay 100% on your machine and are excluded via `.gitignore`.
