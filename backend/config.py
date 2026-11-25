"""Central configuration for TransCrypting (100% local execution)."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
SOURCES_DIR = DATA_DIR / "sources"
PODCASTS_DIR = DATA_DIR / "podcasts"
MODELS_DIR = DATA_DIR / "models" / "whisper"
TTS_MODELS_DIR = DATA_DIR / "models" / "tts"
KOKORO_MODELS_DIR = TTS_MODELS_DIR / "kokoro"
DB_PATH = DATA_DIR / "transcript.db"
FRONTEND_DIR = BASE_DIR / "frontend"

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

DEFAULT_WHISPER_MODEL = "large-v3-turbo"
WHISPER_MODELS = [
    {
        "id": "tiny",
        "label": "Whisper Tiny (~0.3 GB VRAM, ultra rápido)",
        "vram_gb": 0.3,
    },
    {
        "id": "base",
        "label": "Whisper Base (~0.4 GB VRAM, ligero)",
        "vram_gb": 0.4,
    },
    {
        "id": "small",
        "label": "Whisper Small (~0.6 GB VRAM, muy rápido)",
        "vram_gb": 0.6,
    },
    {
        "id": "medium",
        "label": "Whisper Medium (~1.6 GB VRAM, equilibrado)",
        "vram_gb": 1.6,
    },
    {
        "id": "large-v3-turbo",
        "label": "Whisper Large-v3-Turbo (~1.8 GB VRAM, rápido y preciso)",
        "vram_gb": 1.8,
    },
    {
        "id": "large-v3",
        "label": "Whisper Large-v3 (~3.2 GB VRAM, máxima precisión clases/ruido)",
        "vram_gb": 3.2,
    },
]


def list_available_whisper_models(models_dir: Path = MODELS_DIR) -> list[dict]:
    """Return built-in Whisper presets plus any custom CTranslate2 folders placed in data/models/whisper/."""
    models = list(WHISPER_MODELS)
    if models_dir.is_dir():
        for entry in sorted(models_dir.iterdir(), key=lambda p: p.name.lower()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("models--") or entry.name.startswith("."):
                continue
            if (entry / "model.bin").is_file():
                models.append(
                    {
                        "id": f"local:{entry.name}",
                        "label": f"Local: {entry.name} (CTranslate2 personalizado)",
                        "vram_gb": 0.0,
                    }
                )
    return models

DEFAULT_WHISPER_DEVICE = "auto"
DEFAULT_WHISPER_LANGUAGE = "auto"
DEFAULT_NUM_CTX = 8192
DEFAULT_IDLE_UNLOAD_SEC = 180

SUPPORTED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".webm",
    ".aac",
    ".wma",
    ".mp4",
    ".mkv",
    ".mov",
    ".opus",
}

HTML5_PLAYABLE_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".ogg",
    ".webm",
    ".m4a",
    ".aac",
    ".flac",
    ".opus",
}

SUPPORTED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt",
    ".md",
    ".pptx",
    ".docx",
}

SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
}

SUPPORTED_SOURCE_EXTENSIONS = SUPPORTED_DOCUMENT_EXTENSIONS | SUPPORTED_IMAGE_EXTENSIONS

PODCAST_VOICE_PRESETS = {
    "es": {
        "A": {
            "name": "Sofía",
            "key": "es_MX-claude-high",
            "rel_path": "es/es_MX/claude/high/es_MX-claude-high",
        },
        "B": {
            "name": "Mateo",
            "key": "es_MX-ald-medium",
            "rel_path": "es/es_MX/ald/medium/es_MX-ald-medium",
        },
    },
    "en": {
        "A": {
            "name": "Claire",
            "key": "en_US-amy-medium",
            "rel_path": "en/en_US/amy/medium/en_US-amy-medium",
        },
        "B": {
            "name": "Ryan",
            "key": "en_US-ryan-medium",
            "rel_path": "en/en_US/ryan/medium/en_US-ryan-medium",
        },
    },
}


def ensure_directories() -> None:
    """Create local data directories if they do not exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)
    PODCASTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)


KOKORO_MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
KOKORO_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

KOKORO_VOICE_PRESETS = {
    "es": {
        "lang": "es",
        "A": {"name": "Sofía", "voice": "ef_dora", "speed": 1.02},
        "B": {"name": "Mateo", "voice": "em_alex", "speed": 1.0},
    },
    "en": {
        "lang": "en-us",
        "A": {"name": "Claire", "voice": "af_heart", "speed": 1.02},
        "B": {"name": "Ryan", "voice": "am_michael", "speed": 1.0},
    },
}
