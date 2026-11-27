"""Local audio transcription and ffmpeg audio preparation."""

from __future__ import annotations

import gc
import json
import logging
import os
import re
import shutil
import site
import subprocess
import sysconfig
import tempfile
import threading
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

import httpx

from backend.config import (
    ASR_MODELS_DIR,
    CANARY_MODEL_ID,
    CANARY_ONNX_SUPPORTED,
    FASTER_WHISPER_SUPPORTED,
    HTML5_PLAYABLE_EXTENSIONS,
    MLX_ASR_MODELS,
    MLX_ASR_SUPPORTED,
    MODELS_DIR,
    NEMOTRON_MODEL_ID,
    PARAKEET_MODEL_ID,
    PARAKEET_MODEL_REPO,
    PARAKEET_MODELS_DIR,
    PARAKEET_ONNX_SUPPORTED,
    PARAKEET_SUPPORTED,
    QWEN_TORCH_MODELS,
    QWEN_TORCH_SUPPORTED,
    WHISPER_METAL_SUPPORTED,
    WHISPER_MODELS,
    _default_hf_cache_dir,
    list_available_whisper_models,
)

logger = logging.getLogger(__name__)

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DLL_HANDLES: list[Any] = []
_CUDA_DIRS_REGISTERED = False


class WhisperCancelledError(RuntimeError):
    """Raised when an active Whisper transcription is stopped by the user."""


def _parakeet_word_timings(tokens: list[Any]) -> list[dict[str, float]]:
    words = []
    offset = 0
    last_word_end = -1
    for token in tokens:
        for piece in re.finditer(r"\S+", token.text):
            end = round(token.end, 2)
            if offset + piece.start() == last_word_end:
                words[-1]["end"] = max(words[-1]["end"], end)
            else:
                words.append({"start": round(token.start, 2), "end": end})
            last_word_end = offset + piece.end()
        offset += len(token.text)
    return words


def register_cuda_dlls() -> list[str]:
    """Register CUDA 12 / cuDNN 9 DLL directories from Python packages on Windows."""
    global _CUDA_DIRS_REGISTERED
    added: list[str] = []
    if _CUDA_DIRS_REGISTERED:
        return added

    search_roots: list[Path] = []
    for p in site.getsitepackages():
        search_roots.append(Path(p))
    purelib = sysconfig.get_paths().get("purelib")
    if purelib:
        search_roots.append(Path(purelib))

    rel_bins = [
        Path("nvidia") / "cublas" / "bin",
        Path("nvidia") / "cudnn" / "bin",
        Path("nvidia") / "cuda_runtime" / "bin",
        Path("torch") / "lib",
    ]

    current_path = os.environ.get("PATH", "")
    path_parts = current_path.split(os.pathsep) if current_path else []

    for root in search_roots:
        for rel in rel_bins:
            bin_dir = root / rel
            if bin_dir.is_dir():
                bin_str = str(bin_dir)
                if bin_str not in path_parts:
                    path_parts.insert(0, bin_str)
                if hasattr(os, "add_dll_directory"):
                    try:
                        _DLL_HANDLES.append(os.add_dll_directory(bin_str))
                    except OSError:
                        pass
                if bin_str not in added:
                    added.append(bin_str)

    os.environ["PATH"] = os.pathsep.join(path_parts)
    _CUDA_DIRS_REGISTERED = True
    return added


def probe_duration_sec(file_path: Path) -> float:
    """Return media duration in seconds using local ffprobe."""
    try:
        res = subprocess.run(
            [
                _find_ffprobe_exe(),
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=_NO_WINDOW,
        )
        if res.returncode == 0 and res.stdout.strip():
            return max(0.0, float(res.stdout.strip()))
    except Exception:
        pass
    return 0.0


def prepare_audio_file(src_path: Path, dest_dir: Path) -> tuple[float, Path]:
    """Probe audio duration and create an HTML5-playable MP3 copy if container is unsupported."""
    duration = probe_duration_sec(src_path)
    ext = src_path.suffix.lower()
    if ext in HTML5_PLAYABLE_EXTENSIONS:
        return duration, src_path

    playable_path = dest_dir / f"{src_path.stem}_playable.mp3"
    try:
        res = subprocess.run(
            [
                _find_ffmpeg_exe(),
                "-y",
                "-i",
                str(src_path),
                "-vn",
                "-c:a",
                "libmp3lame",
                "-q:a",
                "4",
                str(playable_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            creationflags=_NO_WINDOW,
        )
        if res.returncode == 0 and playable_path.exists():
            if duration <= 0.0:
                duration = probe_duration_sec(playable_path)
            return duration, playable_path
    except Exception:
        pass
    return duration, src_path


def enhance_audio_for_speech(src_path: Path) -> Path | None:
    """Create a temporary 16kHz mono WAV with noise reduction & dynamic leveling for classroom audio."""
    enhanced_path = src_path.with_name(f"{src_path.stem}_enhanced_tmp.wav")
    try:
        res = subprocess.run(
            [
                _find_ffmpeg_exe(),
                "-y",
                "-i",
                str(src_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-af",
                "highpass=f=80,lowpass=f=7800,afftdn=nf=-25,dynaudnorm=p=0.9:m=15:s=12",
                str(enhanced_path),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            creationflags=_NO_WINDOW,
        )
        if res.returncode == 0 and enhanced_path.exists() and enhanced_path.stat().st_size > 0:
            return enhanced_path
    except Exception:
        pass
    try:
        enhanced_path.unlink(missing_ok=True)
    except OSError:
        pass
    return None


_HF_WHISPER_CPP_BASE = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main"
_ONNX_WHISPER_MAP = {
    "tiny": "onnx-community/whisper-tiny",
    "base": "whisper-base",
    "small": "onnx-community/whisper-small",
    "medium": "onnx-community/whisper-medium",
    "large-v3-turbo": "onnx-community/whisper-large-v3-turbo",
    "large-v3": "onnx-community/whisper-large-v3",
    PARAKEET_MODEL_ID: "nemo-parakeet-tdt-0.6b-v3",
    CANARY_MODEL_ID: "nemo-canary-1b-v2",
}


@lru_cache(maxsize=1)
def _cuda_available() -> bool:
    if WHISPER_METAL_SUPPORTED or not FASTER_WHISPER_SUPPORTED:
        return False
    try:
        register_cuda_dlls()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0 and (
            "int8_float16" in ctranslate2.get_supported_compute_types("cuda")
        )
    except (ImportError, RuntimeError, OSError):
        return False


@lru_cache(maxsize=1)
def _whisper_metal_available() -> bool:
    if not WHISPER_METAL_SUPPORTED:
        return False
    try:
        from pywhispercpp.model import Model

        return "MTL :" in Model.system_info()
    except (ImportError, RuntimeError, OSError):
        return False


@lru_cache(maxsize=1)
def _torch_cuda_available() -> bool:
    if not QWEN_TORCH_SUPPORTED:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except (ImportError, RuntimeError, OSError):
        return False


def transcription_options() -> list[dict]:
    models = list_available_whisper_models()
    for model in models:
        model_id = model["id"]
        devices = ["auto", "cpu"]
        if model_id in MLX_ASR_MODELS and MLX_ASR_SUPPORTED:
            devices.insert(1, "metal")
        elif model_id in QWEN_TORCH_MODELS:
            if _torch_cuda_available():
                devices.insert(1, "cuda")
        elif model_id == PARAKEET_MODEL_ID:
            if PARAKEET_SUPPORTED:
                devices.insert(1, "metal")
        elif model_id != CANARY_MODEL_ID:
            if not model_id.startswith("local:") and _whisper_metal_available():
                devices.insert(1, "metal")
            elif _cuda_available():
                devices.insert(1, "cuda")
        model["devices"] = devices
    return models


def _find_ffmpeg_exe() -> str:
    """Locate ffmpeg executable on PATH or WinGet Links."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    winget_link = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    if winget_link.is_file():
        return str(winget_link)
    return "ffmpeg"


def _find_ffprobe_exe() -> str:
    """Locate ffprobe executable on PATH or WinGet Links."""
    found = shutil.which("ffprobe")
    if found:
        return found
    winget_link = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffprobe.exe"
    if winget_link.is_file():
        return str(winget_link)
    return "ffprobe"


def _find_whisper_cpp_exe() -> str | None:
    """Locate whisper.cpp CLI binary in data/models/whisper/bin or system PATH."""
    bin_dir = MODELS_DIR / "bin"
    for name in ("whisper-cli.exe", "whisper-cli", "main.exe"):
        candidate = bin_dir / name
        if candidate.is_file():
            return str(candidate)
    for name in ("whisper-cli", "whisper-cpp"):
        found = shutil.which(name)
        if found:
            return found
    return None


def _ensure_ggml_model(model_name: str, eco_mode: bool = False) -> Path:
    """Resolve or download quantized GGML model for whisper.cpp into data/models/whisper/ggml/."""
    ggml_dir = MODELS_DIR / "ggml"
    ggml_dir.mkdir(parents=True, exist_ok=True)
    clean_name = model_name.replace("local:", "").strip()

    # Check if user placed a custom .bin directly
    custom = ggml_dir / f"ggml-{clean_name}.bin"
    if custom.is_file() and custom.stat().st_size > 1_000_000:
        return custom

    quant_candidates = (
        [f"ggml-{clean_name}-q5_1.bin", f"ggml-{clean_name}.bin"]
        if eco_mode and clean_name in ("tiny", "base", "small")
        else [f"ggml-{clean_name}.bin", f"ggml-{clean_name}-q5_1.bin"]
    )
    for fname in quant_candidates:
        target = ggml_dir / fname
        if target.is_file() and target.stat().st_size > 1_000_000:
            return target

    for fname in quant_candidates:
        url = f"{_HF_WHISPER_CPP_BASE}/{fname}"
        target = ggml_dir / fname
        tmp = target.with_suffix(".bin.part")
        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=180.0) as resp:
                if resp.status_code >= 400:
                    continue
                with tmp.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            tmp.replace(target)
            return target
        except Exception:
            tmp.unlink(missing_ok=True)

    raise RuntimeError(
        f"No se pudo descargar el modelo GGML '{clean_name}' para whisper.cpp en {ggml_dir}."
    )


def _convert_to_wav_16k(src_path: Path, dest_wav: Path) -> Path:
    """Convert any audio file to 16kHz mono PCM WAV via ffmpeg."""
    res = subprocess.run(
        [
            _find_ffmpeg_exe(),
            "-y",
            "-i",
            str(src_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dest_wav),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        creationflags=_NO_WINDOW,
    )
    if res.returncode != 0 or not dest_wav.is_file():
        raise RuntimeError(
            "ffmpeg es necesario para convertir el audio a WAV 16kHz para whisper.cpp."
        )
    return dest_wav


@contextmanager
def _hf_online_context() -> Iterator[None]:
    """Temporarily allow Hugging Face Hub downloads on first model use, restoring prior offline flags."""
    from huggingface_hub import constants as hf_constants

    prev_env = os.environ.get("HF_HUB_OFFLINE")
    prev_const = getattr(hf_constants, "HF_HUB_OFFLINE", False)
    try:
        os.environ["HF_HUB_OFFLINE"] = "0"
        hf_constants.HF_HUB_OFFLINE = False
        yield
    finally:
        if prev_env is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_env
        hf_constants.HF_HUB_OFFLINE = prev_const


class WhisperService:
    """Manage local Whisper / Parakeet / ONNX transcription models and release them between workloads."""

    def __init__(self) -> None:
        self._model: Any = None
        self._onnx_model: Any = None
        self._onnx_vad: Any = None
        self._qwen_model: Any = None
        self._qwen_processor: Any = None
        self._model_name: str = ""
        self._backend: str = "faster-whisper"
        self._cpp_exe: str | None = None
        self._ggml_path: Path | None = None
        self.eco_cpu_threads: int = 0
        self.active_device: str = ""
        self.active_compute_type: str = ""

    @property
    def is_loaded(self) -> bool:
        return (
            self._model is not None
            or self._onnx_model is not None
            or self._ggml_path is not None
            or self._qwen_model is not None
        )

    def _resolve_model_source(self, model_name: str) -> str:
        """Use local CTranslate2 folder or cached Whisper weights; otherwise download once to MODELS_DIR."""
        if model_name.startswith("local:"):
            folder_name = model_name[len("local:") :].strip()
            custom_dir = MODELS_DIR / folder_name
            if not (custom_dir / "model.bin").is_file():
                raise RuntimeError(
                    f"No se encontró el modelo local '{folder_name}' (falta model.bin en {custom_dir})."
                )
            return str(custom_dir)

        direct_dir = MODELS_DIR / model_name
        if direct_dir.is_dir() and (direct_dir / "model.bin").is_file():
            return str(direct_dir)

        from faster_whisper import download_model

        try:
            return download_model(
                model_name,
                cache_dir=str(MODELS_DIR),
                local_files_only=True,
            )
        except Exception:
            pass

        with _hf_online_context():
            return download_model(
                model_name,
                cache_dir=str(MODELS_DIR),
                local_files_only=False,
            )

    def _load_onnx_asr(self, model_name: str) -> str:
        """Load Whisper, Parakeet, or Canary via onnx-asr with CPU thread cap."""
        import onnx_asr
        import onnxruntime as ort

        mapped = _ONNX_WHISPER_MAP.get(model_name.replace("local:", "").strip(), "whisper-base")
        threads = self.eco_cpu_threads if self.eco_cpu_threads > 0 else max(2, min(6, (os.cpu_count() or 4) // 2))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        try:
            self._onnx_model = onnx_asr.load_model(
                mapped,
                quantization="int8",
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._onnx_vad = onnx_asr.load_vad(
                "silero",
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        except Exception:
            with _hf_online_context():
                self._onnx_model = onnx_asr.load_model(
                    mapped,
                    quantization="int8",
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._onnx_vad = onnx_asr.load_vad(
                    "silero",
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
        self._backend = "onnx-asr"
        self._model_name = model_name
        self.active_device = "cpu"
        self.active_compute_type = "onnx-int8"
        return "cpu"

    def load(self, model_name: str, device_pref: str = "auto") -> str:
        """Load a transcription model on Metal, CUDA, or CPU (faster-whisper / onnx-asr / whisper.cpp)."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pref = (device_pref or "auto").lower()

        if self.is_loaded and self._model_name == model_name:
            if pref == "auto" or pref == self.active_device:
                return self.active_device

        self.unload()
        if model_name in QWEN_TORCH_MODELS and not MLX_ASR_SUPPORTED:
            preset = QWEN_TORCH_MODELS[model_name]
            if not QWEN_TORCH_SUPPORTED:
                raise RuntimeError(
                    f"{preset['engine']} en Windows/Linux requiere PyTorch y transformers>=5.13."
                )

            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

            target_dir = ASR_MODELS_DIR / model_name
            target_dir.mkdir(parents=True, exist_ok=True)
            has_weights = (target_dir / "config.json").is_file() and (
                (target_dir / "model.safetensors").is_file()
                or any(target_dir.glob("model-*.safetensors"))
            )
            if not has_weights:
                with _hf_online_context():
                    snapshot_download(
                        preset["repo"],
                        local_dir=str(target_dir),
                        local_files_only=False,
                    )

            use_cuda = pref in ("auto", "cuda") and torch.cuda.is_available()
            if pref == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("No se encontró GPU CUDA disponible para Qwen3-ASR.")
            device = "cuda" if use_cuda else "cpu"
            if use_cuda:
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32

            self._qwen_processor = AutoProcessor.from_pretrained(str(target_dir))
            self._qwen_model = (
                Qwen3ASRForConditionalGeneration.from_pretrained(str(target_dir), dtype=dtype)
                .to(device)
                .eval()
            )
            self._backend = "qwen-torch"
            self._model_name = model_name
            self.active_device = device
            self.active_compute_type = str(dtype).replace("torch.", "")
            return device

        if model_name in MLX_ASR_MODELS:
            preset = MLX_ASR_MODELS[model_name]
            if not MLX_ASR_SUPPORTED:
                raise RuntimeError(f"{preset['engine']} requiere Apple Silicon y el paquete mlx-audio.")
            if pref == "cuda":
                raise RuntimeError(f"Selecciona Automático o CPU para usar {preset['engine']}.")

            import mlx.core as mx
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
            from mlx_audio.stt import load

            patterns = ["*.json", "*.safetensors", "*.txt", "*.model"]
            try:
                snapshot = Path(snapshot_download(preset["repo"], local_files_only=True, allow_patterns=patterns))
            except LocalEntryNotFoundError:
                snapshot = None
            files = ("config.json", "model.safetensors", "tokenizer.model") if model_name == NEMOTRON_MODEL_ID else (
                "config.json", "model.safetensors", "vocab.json", "merges.txt", "preprocessor_config.json", "tokenizer_config.json",
            )
            if snapshot is None or not all((snapshot / name).is_file() for name in files):
                snapshot = Path(snapshot_download(preset["repo"], allow_patterns=patterns))
            device = "metal" if pref != "cpu" and mx.metal.is_available() else "cpu"
            with mx.stream(mx.gpu if device == "metal" else mx.cpu):
                self._model = load(str(snapshot))
            self._model_name = model_name
            self.active_device = device
            return device

        if model_name in (PARAKEET_MODEL_ID, CANARY_MODEL_ID):
            if model_name == CANARY_MODEL_ID or not PARAKEET_SUPPORTED:
                try:
                    return self._load_onnx_asr(model_name)
                except Exception as exc:
                    raise RuntimeError(
                        f"{model_name} requiere 'onnx-asr' instalado."
                    ) from exc
            if pref == "cuda":
                raise RuntimeError("Selecciona Automático o CPU para usar Parakeet v3.")

            import mlx.core as mx
            from huggingface_hub import snapshot_download
            from parakeet_mlx import from_pretrained

            if not all((PARAKEET_MODELS_DIR / name).is_file() for name in ("config.json", "model.safetensors")):
                snapshot_download(
                    PARAKEET_MODEL_REPO,
                    local_dir=PARAKEET_MODELS_DIR,
                    allow_patterns=["config.json", "model.safetensors"],
                )
            device = "metal" if pref != "cpu" and mx.metal.is_available() else "cpu"
            with mx.stream(mx.gpu if device == "metal" else mx.cpu):
                self._model = from_pretrained(str(PARAKEET_MODELS_DIR))
            self._model_name = model_name
            self.active_device = device
            return self.active_device

        if (
            pref in ("auto", "metal")
            and WHISPER_METAL_SUPPORTED
            and any(model["id"] == model_name for model in WHISPER_MODELS)
        ):
            try:
                self._model = self._load_metal(model_name)
            except Exception as exc:
                if pref == "metal":
                    raise
                logger.warning("No se pudo cargar Whisper con Metal. Usando CPU: %s", exc)
            else:
                self._model_name = model_name
                self.active_device = "metal"
                return self.active_device

        try:
            register_cuda_dlls()
            from faster_whisper import WhisperModel
        except ImportError:
            try:
                return self._load_onnx_asr(model_name)
            except Exception as onnx_exc:
                cpp_exe = _find_whisper_cpp_exe()
                if not cpp_exe:
                    raise RuntimeError(
                        "Transcripción de audio no disponible: instala 'onnx-asr[hub]' (ARM64) "
                        f"o coloca 'whisper-cli.exe' en '{MODELS_DIR / 'bin'}'. Detalle: {onnx_exc}"
                    ) from onnx_exc
                self._backend = "whisper.cpp"
                self._cpp_exe = cpp_exe
                self._ggml_path = _ensure_ggml_model(model_name, eco_mode=self.eco_cpu_threads > 0)
                self._model_name = model_name
                self.active_device = "cpu"
                self.active_compute_type = "ggml"
                return "cpu"

        self._backend = "faster-whisper"
        model_source = self._resolve_model_source(model_name)

        if pref == "cuda" or (pref == "auto" and not WHISPER_METAL_SUPPORTED):
            try:
                self._model = WhisperModel(
                    model_source,
                    device="cuda",
                    compute_type="int8_float16",
                    local_files_only=True,
                )
                self._model_name = model_name
                self.active_device = "cuda"
                self.active_compute_type = "int8_float16"
                return "cuda"
            except Exception as exc:
                if pref == "cuda":
                    raise RuntimeError(
                        f"No se pudo cargar Whisper en GPU CUDA: {exc}"
                    ) from exc

        threads = self.eco_cpu_threads if self.eco_cpu_threads > 0 else max(4, min(12, (os.cpu_count() or 8) // 2))
        self._model = WhisperModel(
            model_source,
            device="cpu",
            compute_type="int8",
            cpu_threads=threads,
            local_files_only=True,
        )
        self._model_name = model_name
        self.active_device = "cpu"
        self.active_compute_type = "int8"
        return "cpu"

    def _load_metal(self, model_name: str) -> Any:
        from huggingface_hub import hf_hub_download
        from pywhispercpp.model import Model

        if "MTL :" not in Model.system_info():
            raise RuntimeError("La instalación de whisper.cpp no incluye Metal.")

        model_dir = MODELS_DIR / "ggml"
        variant = "large-v3-turbo-q5_0" if model_name == "large-v3-turbo" else model_name
        model_path = model_dir / f"ggml-{variant}.bin"
        if not model_path.is_file():
            model_path = Path(
                hf_hub_download("ggerganov/whisper.cpp", model_path.name, local_dir=model_dir)
            )
        model = Model(
            str(model_path),
            context_params={"use_gpu": True},
            params_sampling_strategy=1,
            beam_search={"beam_size": 5, "patience": -1.0},
            print_progress=False,
        )
        if model._ctx is None:
            raise RuntimeError("whisper.cpp no pudo inicializar el modelo.")
        return model

    def _transcribe_whisper_cpp(
        self,
        audio_path: Path,
        language: str = "auto",
        progress_callback: Callable[[float, str], None] | None = None,
        classroom_enhance: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if not self._cpp_exe or not self._ggml_path:
            raise RuntimeError("whisper.cpp no está cargado.")

        threads = self.eco_cpu_threads if self.eco_cpu_threads > 0 else max(2, min(6, (os.cpu_count() or 4) // 2))
        lang_arg = "auto" if not language or language == "auto" else language
        duration_sec = probe_duration_sec(audio_path)

        with tempfile.TemporaryDirectory(prefix="wcpp_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            wav_path = tmp_root / "input_16k.wav"
            enhanced_tmp = enhance_audio_for_speech(audio_path) if classroom_enhance else None
            try:
                src_wav = enhanced_tmp if enhanced_tmp is not None else _convert_to_wav_16k(audio_path, wav_path)
                out_prefix = tmp_root / "result"
                cmd = [
                    self._cpp_exe,
                    "-m",
                    str(self._ggml_path),
                    "-f",
                    str(src_wav),
                    "-l",
                    lang_arg,
                    "-t",
                    str(threads),
                    "-oj",
                    "-of",
                    str(out_prefix),
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    creationflags=_NO_WINDOW,
                )
                while proc.poll() is None:
                    if cancel_event is not None and cancel_event.is_set():
                        proc.terminate()
                        raise WhisperCancelledError("Transcripción detenida por el usuario.")
                    import time as _time
                    _time.sleep(0.25)

                if proc.returncode != 0:
                    err = (proc.stderr.read() if proc.stderr else b"").decode("utf-8", errors="ignore")
                    raise RuntimeError(f"whisper.cpp falló ({proc.returncode}): {err.strip()}")

                json_file = Path(f"{out_prefix}.json")
                data = json.loads(json_file.read_text(encoding="utf-8")) if json_file.is_file() else {}
                detected_lang = str((data.get("result") or {}).get("language") or ("" if lang_arg == "auto" else lang_arg))
                raw_items = data.get("transcription") or []

                segments: list[dict[str, Any]] = []
                text_parts: list[str] = []
                last_norm = ""
                for item in raw_items:
                    clean = str(item.get("text") or "").strip()
                    offsets = item.get("offsets") or {}
                    start_s = round(float(offsets.get("from", 0)) / 1000.0, 2)
                    end_s = round(float(offsets.get("to", 0)) / 1000.0, 2)
                    if clean:
                        norm = clean.lower()
                        if norm != last_norm:
                            segments.append({"start": start_s, "end": end_s, "text": clean})
                            text_parts.append(clean)
                            last_norm = norm

                if progress_callback:
                    progress_callback(0.99, detected_lang)

                full_text = "\n".join(text_parts).strip() or "[No se detectó voz comprensible en la grabación]"
                final_dur = duration_sec or (segments[-1]["end"] if segments else 0.0)
                return {
                    "text": full_text,
                    "segments": segments,
                    "language": detected_lang,
                    "duration_sec": round(float(final_dur), 2),
                }
            finally:
                if enhanced_tmp is not None:
                    enhanced_tmp.unlink(missing_ok=True)

    def _transcribe_onnx_asr(
        self,
        audio_path: Path,
        language: str = "auto",
        progress_callback: Callable[[float, str], None] | None = None,
        classroom_enhance: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        if self._onnx_model is None or self._onnx_vad is None:
            raise RuntimeError("Modelo onnx-asr no está cargado.")

        lang_arg = None if not language or language == "auto" else language
        duration_sec = probe_duration_sec(audio_path)

        with tempfile.TemporaryDirectory(prefix="onnx_asr_") as tmp_dir:
            tmp_root = Path(tmp_dir)
            wav_path = tmp_root / "input_16k.wav"
            enhanced_tmp = enhance_audio_for_speech(audio_path) if classroom_enhance else None
            try:
                src_wav = enhanced_tmp if enhanced_tmp is not None else _convert_to_wav_16k(audio_path, wav_path)
                adapter = self._onnx_model.with_vad(self._onnx_vad)
                kwargs: dict[str, Any] = {}
                if self._model_name == CANARY_MODEL_ID:
                    kwargs["language"] = lang_arg or "es"
                elif lang_arg and self._model_name != PARAKEET_MODEL_ID:
                    kwargs["language"] = lang_arg
                seg_iter = adapter.recognize(str(src_wav), **kwargs)

                segments: list[dict[str, Any]] = []
                text_parts: list[str] = []
                last_norm = ""
                for seg in seg_iter:
                    if cancel_event is not None and cancel_event.is_set():
                        raise WhisperCancelledError("Transcripción detenida por el usuario.")
                    clean = str(getattr(seg, "text", "") or "").strip()
                    start_s = round(float(getattr(seg, "start", 0.0) or 0.0), 2)
                    end_s = round(float(getattr(seg, "end", 0.0) or 0.0), 2)
                    if clean:
                        norm = clean.lower()
                        if norm != last_norm:
                            segments.append({"start": start_s, "end": end_s, "text": clean})
                            text_parts.append(clean)
                            last_norm = norm
                    if progress_callback and duration_sec > 0:
                        ratio = min(0.99, max(0.01, end_s / duration_sec))
                        progress_callback(ratio, lang_arg or "")

                if not text_parts and duration_sec <= 20.0:
                    direct_text = str(self._onnx_model.recognize(str(src_wav), **kwargs) or "").strip()
                    if direct_text:
                        text_parts.append(direct_text)
                        segments.append({"start": 0.0, "end": round(float(duration_sec), 2), "text": direct_text})

                if progress_callback and duration_sec > 0:
                    progress_callback(0.99, lang_arg or "")

                full_text = "\n".join(text_parts).strip() or "[No se detectó voz comprensible en la grabación]"
                final_dur = duration_sec or (segments[-1]["end"] if segments else 0.0)
                resolved_lang = (
                    (lang_arg or "es") if self._model_name == CANARY_MODEL_ID else (lang_arg or "auto")
                )
                return {
                    "text": full_text,
                    "segments": segments,
                    "language": resolved_lang,
                    "duration_sec": round(float(final_dur), 2),
                }
            finally:
                if enhanced_tmp is not None:
                    enhanced_tmp.unlink(missing_ok=True)

    @staticmethod
    def _build_qwen_chunks(samples: Any, sample_rate: int = 16000) -> list[tuple[float, float]]:
        """Group audio into ~26s speech-aligned windows using energy-valley splitting."""
        import numpy as np

        total_sec = len(samples) / float(sample_rate)
        if total_sec <= 0.0:
            return []
        if total_sec <= 28.0:
            return [(0.0, total_sec)]

        # Energy-valley splitter targeting 20–28s chunks without cutting mid-word
        windows: list[tuple[float, float]] = []
        frame_len = int(0.05 * sample_rate)
        cursor_s = 0.0
        while cursor_s < total_sec - 0.15:
            if total_sec - cursor_s <= 28.0:
                windows.append((cursor_s, total_sec))
                break
            search_start = int((cursor_s + 18.0) * sample_rate)
            search_end = min(len(samples), int((cursor_s + 28.0) * sample_rate))
            best_sample = int((cursor_s + 25.0) * sample_rate)
            best_energy = float("inf")
            for pos in range(search_start, max(search_start + 1, search_end - frame_len), frame_len):
                frame = samples[pos : pos + frame_len]
                energy = float(np.mean(frame * frame))
                if energy < best_energy:
                    best_energy = energy
                    best_sample = pos + frame_len // 2
            split_s = min(total_sec, best_sample / float(sample_rate))
            windows.append((cursor_s, split_s))
            cursor_s = split_s
        return windows

    def _transcribe_qwen_torch(
        self,
        audio_path: Path,
        language: str = "auto",
        progress_callback: Callable[[float, str], None] | None = None,
        classroom_enhance: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Transcribe audio using Qwen3ASRForConditionalGeneration on CUDA or CPU."""
        if self._qwen_model is None or self._qwen_processor is None:
            raise RuntimeError("Qwen3-ASR no está cargado.")

        import numpy as np
        import soundfile as sf
        import torch

        lang_arg = None if not language or language == "auto" else language
        lang_name_to_code = {
            "spanish": "es", "english": "en", "portuguese": "pt",
            "french": "fr", "german": "de", "italian": "it",
            "chinese": "zh", "japanese": "ja", "korean": "ko",
        }

        with tempfile.TemporaryDirectory(prefix="qwen_asr_") as tmp_dir:
            wav_path = Path(tmp_dir) / "input_16k.wav"
            enhanced_tmp = enhance_audio_for_speech(audio_path) if classroom_enhance else None
            try:
                src_wav = enhanced_tmp if enhanced_tmp is not None else _convert_to_wav_16k(audio_path, wav_path)
                samples, sr = sf.read(str(src_wav), dtype="float32")
                if getattr(samples, "ndim", 1) > 1:
                    samples = samples.mean(axis=1)
                if sr != 16000:
                    src_wav = _convert_to_wav_16k(src_wav, wav_path)
                    samples, sr = sf.read(str(src_wav), dtype="float32")
                    if getattr(samples, "ndim", 1) > 1:
                        samples = samples.mean(axis=1)

                total_duration = len(samples) / float(sr or 16000)
                windows = self._build_qwen_chunks(samples, sr)
                segments: list[dict[str, Any]] = []
                text_parts: list[str] = []
                detected_lang = lang_arg or ""

                model = self._qwen_model
                processor = self._qwen_processor
                model_dtype = next(model.parameters()).dtype

                for seg_start, seg_end in windows:
                    if cancel_event is not None and cancel_event.is_set():
                        raise WhisperCancelledError("Transcripción detenida por el usuario.")

                    s_idx = int(seg_start * sr)
                    e_idx = min(len(samples), int(seg_end * sr))
                    chunk_np = np.ascontiguousarray(samples[s_idx:e_idx], dtype=np.float32)
                    if chunk_np.size < int(0.15 * sr):
                        continue
                    if float(np.max(np.abs(chunk_np))) < 1e-4:
                        continue

                    inputs = processor.apply_transcription_request(audio=chunk_np, language=lang_arg)
                    moved: dict[str, Any] = {}
                    for k, v in inputs.items():
                        if hasattr(v, "to"):
                            if hasattr(v, "is_floating_point") and v.is_floating_point():
                                moved[k] = v.to(device=model.device, dtype=model_dtype)
                            else:
                                moved[k] = v.to(device=model.device)
                        else:
                            moved[k] = v

                    with torch.inference_mode():
                        output_ids = model.generate(**moved, max_new_tokens=512, do_sample=False)

                    prompt_len = moved["input_ids"].shape[1]
                    gen_ids = output_ids[0, prompt_len:]
                    parsed = processor.decode(gen_ids, return_format="parsed")
                    clean = str((parsed or {}).get("transcription") or "").strip()
                    raw_det = str((parsed or {}).get("language") or "").strip()
                    if raw_det and not detected_lang:
                        detected_lang = lang_name_to_code.get(raw_det.lower(), raw_det.lower())

                    if clean:
                        segments.append({
                            "start": round(float(seg_start), 2),
                            "end": round(float(seg_end), 2),
                            "text": clean,
                        })
                        text_parts.append(clean)

                    if progress_callback and total_duration > 0:
                        ratio = min(0.99, max(0.01, float(seg_end) / total_duration))
                        progress_callback(ratio, detected_lang)

                full_text = "\n".join(text_parts).strip() or "[No se detectó voz comprensible en la grabación]"
                return {
                    "text": full_text,
                    "segments": segments,
                    "language": detected_lang or (lang_arg or "auto"),
                    "duration_sec": round(float(total_duration), 2),
                }
            finally:
                if enhanced_tmp is not None:
                    enhanced_tmp.unlink(missing_ok=True)

    def transcribe(
        self,
        audio_path: Path,
        language: str = "auto",
        progress_callback: Callable[[float, str], None] | None = None,
        classroom_enhance: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Transcribe an audio file synchronously and emit progress callbacks."""
        if self._backend == "qwen-torch":
            return self._transcribe_qwen_torch(
                audio_path=audio_path,
                language=language,
                progress_callback=progress_callback,
                classroom_enhance=classroom_enhance,
                cancel_event=cancel_event,
            )
        if self._backend == "onnx-asr":
            return self._transcribe_onnx_asr(
                audio_path=audio_path,
                language=language,
                progress_callback=progress_callback,
                classroom_enhance=classroom_enhance,
                cancel_event=cancel_event,
            )
        if self._backend == "whisper.cpp":
            return self._transcribe_whisper_cpp(
                audio_path=audio_path,
                language=language,
                progress_callback=progress_callback,
                classroom_enhance=classroom_enhance,
                cancel_event=cancel_event,
            )

        model = self._model
        if model is None:
            raise RuntimeError("El modelo de transcripción no está cargado.")
        use_parakeet = self._model_name == PARAKEET_MODEL_ID
        use_mlx_asr = self._model_name in MLX_ASR_MODELS
        use_nemotron = self._model_name == NEMOTRON_MODEL_ID
        use_whisper_metal = not (use_parakeet or use_mlx_asr) and self.active_device == "metal"

        if cancel_event is not None and cancel_event.is_set():
            raise WhisperCancelledError("Transcripción detenida por el usuario.")

        lang_arg = None if not language or language == "auto" else language
        if use_nemotron and lang_arg is None:
            raise RuntimeError("Selecciona el idioma del audio en Ajustes para usar Nemotron.")
        enhanced_tmp: Path | None = None
        target_audio = audio_path
        if classroom_enhance:
            enhanced_tmp = enhance_audio_for_speech(audio_path)
            if enhanced_tmp is not None:
                target_audio = enhanced_tmp

        try:
            if cancel_event is not None and cancel_event.is_set():
                raise WhisperCancelledError("Transcripción detenida por el usuario.")

            if use_parakeet:
                import mlx.core as mx

                completed_samples = 0

                def on_chunk(end: int, total: int) -> None:
                    nonlocal completed_samples
                    if cancel_event is not None and cancel_event.is_set():
                        raise WhisperCancelledError("Transcripción detenida por el usuario.")
                    if progress_callback and completed_samples:
                        progress_callback(min(0.99, completed_samples / total), "")
                    completed_samples = end

                total_duration = probe_duration_sec(target_audio)
                detected_lang = ""
                with mx.stream(mx.gpu if self.active_device == "metal" else mx.cpu):
                    result = model.transcribe(
                        target_audio,
                        chunk_duration=45.0,
                        overlap_duration=5.0,
                        chunk_callback=on_chunk,
                    )
                if cancel_event is not None and cancel_event.is_set():
                    raise WhisperCancelledError("Transcripción detenida por el usuario.")
                segments_iter = ((seg.start, seg.end, seg.text, seg.tokens) for seg in result.sentences)
            elif use_mlx_asr:
                import mlx.core as mx
                from faster_whisper.audio import decode_audio

                audio = decode_audio(str(target_audio), sampling_rate=16000)
                total_duration = audio.size / 16000
                if cancel_event is not None and cancel_event.is_set():
                    raise WhisperCancelledError("Transcripción detenida por el usuario.")
                detected_lang = lang_arg or ""
                with mx.stream(mx.gpu if self.active_device == "metal" else mx.cpu):
                    if use_nemotron:
                        result = model.generate(audio, language=lang_arg)
                        segments_iter = ((seg.start, seg.end, seg.text, seg.tokens) for seg in result.sentences)
                    else:
                        result = model.generate(audio, language=lang_arg, chunk_duration=45.0)
                        detected_lang = detected_lang or next((lang for lang in result.language if lang), "")
                        segments_iter = ((seg["start"], seg["end"], seg["text"], None) for seg in result.segments)
                if cancel_event is not None and cancel_event.is_set():
                    raise WhisperCancelledError("Transcripción detenida por el usuario.")
            elif use_whisper_metal:
                from _pywhispercpp import whisper_full_lang_id, whisper_lang_str
                from faster_whisper.audio import decode_audio

                audio = decode_audio(str(target_audio), sampling_rate=16000)
                total_duration = audio.size / 16000
                if cancel_event is not None and cancel_event.is_set():
                    raise WhisperCancelledError("Transcripción detenida por el usuario.")

                def on_segment(segment: Any) -> None:
                    if progress_callback is not None and total_duration > 0:
                        ratio = min(0.99, max(0.01, segment.t1 / 100 / total_duration))
                        progress_callback(ratio, lang_arg or "")

                raw_segments = model.transcribe(
                    audio,
                    language=lang_arg or "auto",
                    new_segment_callback=on_segment if progress_callback else None,
                    abort_callback=cancel_event.is_set if cancel_event is not None else None,
                )
                if cancel_event is not None and cancel_event.is_set():
                    raise WhisperCancelledError("Transcripción detenida por el usuario.")
                detected_lang = whisper_lang_str(whisper_full_lang_id(model._ctx)) or ""
                segments_iter = ((seg.t0 / 100, seg.t1 / 100, seg.text, None) for seg in raw_segments)
            else:
                raw_segments, info = model.transcribe(
                    str(target_audio),
                    language=lang_arg,
                    beam_size=5,
                    multilingual=(lang_arg is None),
                    condition_on_previous_text=False,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    hallucination_silence_threshold=2.0,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.35,
                        "min_silence_duration_ms": 700,
                        "speech_pad_ms": 400,
                    },
                )
                total_duration = float(getattr(info, "duration", 0.0) or 0.0)
                detected_lang = getattr(info, "language", "") or (lang_arg or "")
                segments_iter = ((seg.start, seg.end, seg.text, None) for seg in raw_segments)

            segments: list[dict[str, Any]] = []
            text_parts: list[str] = []
            last_norm = ""

            for start, end, text, tokens in segments_iter:
                if cancel_event is not None and cancel_event.is_set():
                    raise WhisperCancelledError("Transcripción detenida por el usuario.")
                clean = (text or "").strip()
                if clean:
                    norm = clean.lower()
                    if use_parakeet or norm != last_norm:
                        segment = {
                            "start": round(float(start), 2),
                            "end": round(float(end), 2),
                            "text": clean,
                        }
                        if tokens is not None:
                            segment["words"] = _parakeet_word_timings(tokens)
                        segments.append(segment)
                        text_parts.append(clean)
                        last_norm = norm
                if not use_parakeet and not use_whisper_metal and progress_callback and total_duration > 0:
                    ratio = min(0.99, max(0.01, float(end) / total_duration))
                    progress_callback(ratio, detected_lang)

            full_text = "\n".join(text_parts).strip()
            if not full_text:
                full_text = "[No se detectó voz comprensible en la grabación]"

            return {
                "text": full_text,
                "segments": segments,
                "language": detected_lang,
                "duration_sec": round(total_duration, 2),
            }
        finally:
            if enhanced_tmp is not None:
                try:
                    enhanced_tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def unload(self) -> None:
        """Release transcription weights and any cached device memory."""
        mlx_model = self._model_name == PARAKEET_MODEL_ID or self._model_name in MLX_ASR_MODELS
        clear_mlx_cache = mlx_model and self.active_device == "metal"
        clear_cuda_cache = self._backend == "qwen-torch" or self.active_device == "cuda"
        if self._model is not None:
            del self._model
            self._model = None
        if self._onnx_model is not None:
            del self._onnx_model
            self._onnx_model = None
        if self._onnx_vad is not None:
            del self._onnx_vad
            self._onnx_vad = None
        if self._qwen_model is not None:
            del self._qwen_model
            self._qwen_model = None
        if self._qwen_processor is not None:
            del self._qwen_processor
            self._qwen_processor = None
        self._backend = "faster-whisper"
        self._cpp_exe = None
        self._ggml_path = None
        self._model_name = ""
        self.active_device = ""
        self.active_compute_type = ""
        gc.collect()
        if clear_cuda_cache and QWEN_TORCH_SUPPORTED:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        if clear_mlx_cache:
            import mlx.core as mx

            mx.clear_cache()


whisper_service = WhisperService()


def download_asr_model(model_name: str) -> dict[str, Any]:
    """Explicitly download weights for a single user-selected ASR model preset."""
    mid = (model_name or "").strip()
    if not mid or mid.startswith("local:") or ".." in mid or "/" in mid or "\\" in mid:
        raise ValueError("Solo se pueden descargar presets oficiales integrados.")

    if mid in QWEN_TORCH_MODELS and not MLX_ASR_SUPPORTED:
        if not QWEN_TORCH_SUPPORTED:
            raise RuntimeError("Qwen3-ASR requiere PyTorch y transformers>=5.13 instalados.")
        from huggingface_hub import snapshot_download

        preset = QWEN_TORCH_MODELS[mid]
        target_dir = ASR_MODELS_DIR / mid
        target_dir.mkdir(parents=True, exist_ok=True)
        with _hf_online_context():
            snapshot_download(preset["repo"], local_dir=str(target_dir), local_files_only=False)
        return {"id": mid, "installed": True}

    if mid in MLX_ASR_MODELS and MLX_ASR_SUPPORTED:
        from huggingface_hub import snapshot_download

        preset = MLX_ASR_MODELS[mid]
        with _hf_online_context():
            snapshot_download(preset["repo"], local_files_only=False)
        return {"id": mid, "installed": True}

    if mid in (PARAKEET_MODEL_ID, CANARY_MODEL_ID):
        if mid == PARAKEET_MODEL_ID and PARAKEET_SUPPORTED:
            from huggingface_hub import snapshot_download

            PARAKEET_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            with _hf_online_context():
                snapshot_download(
                    PARAKEET_MODEL_REPO,
                    local_dir=PARAKEET_MODELS_DIR,
                    allow_patterns=["config.json", "model.safetensors"],
                )
            return {"id": mid, "installed": True}
        if not (PARAKEET_ONNX_SUPPORTED or CANARY_ONNX_SUPPORTED):
            raise RuntimeError("Este modelo requiere 'onnx-asr' instalado en .venv.")
        import onnx_asr

        mapped = _ONNX_WHISPER_MAP[mid]
        with _hf_online_context():
            m = onnx_asr.load_model(mapped, quantization="int8", providers=["CPUExecutionProvider"])
            v = onnx_asr.load_vad("silero", providers=["CPUExecutionProvider"])
            del m, v
        return {"id": mid, "installed": True}

    if any(m["id"] == mid for m in WHISPER_MODELS):
        if not FASTER_WHISPER_SUPPORTED:
            raise RuntimeError(
                "faster-whisper no está instalado en este perfil. Reinstala con setup.bat (perfil 2 o 3)."
            )
        from faster_whisper import download_model

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with _hf_online_context():
            download_model(mid, cache_dir=str(MODELS_DIR), local_files_only=False)
        return {"id": mid, "installed": True}

    raise ValueError(f"Modelo de transcripción desconocido: {mid}")


def delete_asr_model(
    model_name: str,
    models_dir: Path = MODELS_DIR,
    asr_dir: Path = ASR_MODELS_DIR,
    hf_cache_dir: Path | None = None,
) -> bool:
    """Delete locally cached weights for a specific ASR preset without touching other models."""
    mid = (model_name or "").strip()
    if not mid or mid.startswith("local:") or ".." in mid or "/" in mid or "\\" in mid:
        raise ValueError("No se puede eliminar un modelo personalizado desde este botón.")

    removed = False
    hf_cache = hf_cache_dir if hf_cache_dir is not None else _default_hf_cache_dir()

    if mid in QWEN_TORCH_MODELS:
        target = asr_dir / mid
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed = True

    if mid == PARAKEET_MODEL_ID:
        if PARAKEET_SUPPORTED and PARAKEET_MODELS_DIR.exists():
            shutil.rmtree(PARAKEET_MODELS_DIR, ignore_errors=True)
            removed = True
        onnx_dir = hf_cache / "models--istupakov--parakeet-tdt-0.6b-v3-onnx"
        if onnx_dir.exists():
            shutil.rmtree(onnx_dir, ignore_errors=True)
            removed = True

    if mid == CANARY_MODEL_ID:
        onnx_dir = hf_cache / "models--istupakov--canary-1b-v2-onnx"
        if onnx_dir.exists():
            shutil.rmtree(onnx_dir, ignore_errors=True)
            removed = True

    if mid in MLX_ASR_MODELS:
        slug = "models--" + MLX_ASR_MODELS[mid]["repo"].replace("/", "--")
        mlx_dir = hf_cache / slug
        if mlx_dir.exists():
            shutil.rmtree(mlx_dir, ignore_errors=True)
            removed = True

    if any(model["id"] == mid for model in WHISPER_MODELS):
        for path in (models_dir / "ggml").glob(f"ggml-{mid}*.bin"):
            if path.stem.removeprefix(f"ggml-{mid}") in ("", "-q5_0", "-q5_1", "-q8_0"):
                path.unlink()
                removed = True

    direct = models_dir / mid
    if direct.is_dir() and (direct / "model.bin").is_file():
        shutil.rmtree(direct, ignore_errors=True)
        removed = True

    if models_dir.is_dir():
        suffix = f"--faster-whisper-{mid.lower()}"
        for entry in list(models_dir.iterdir()):
            if entry.is_dir() and entry.name.lower().endswith(suffix):
                shutil.rmtree(entry, ignore_errors=True)
                removed = True
                lock_dir = models_dir / ".locks" / entry.name
                if lock_dir.exists():
                    shutil.rmtree(lock_dir, ignore_errors=True)

    return removed
