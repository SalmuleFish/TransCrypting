"""Local faster-whisper transcription and ffmpeg audio preparation."""

from __future__ import annotations

import gc
import os
import site
import subprocess
import sysconfig
import threading
from pathlib import Path
from typing import Any, Callable

from backend.config import HTML5_PLAYABLE_EXTENSIONS, MODELS_DIR

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DLL_HANDLES: list[Any] = []
_CUDA_DIRS_REGISTERED = False


class WhisperCancelledError(RuntimeError):
    """Raised when an active Whisper transcription is stopped by the user."""


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
                "ffprobe",
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
                "ffmpeg",
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
                "ffmpeg",
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


class WhisperService:
    """Manages local faster-whisper loading, transcription, and explicit VRAM cleanup."""

    def __init__(self) -> None:
        self._model: Any = None
        self._model_name: str = ""
        self.active_device: str = ""
        self.active_compute_type: str = ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

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
        from huggingface_hub import constants as hf_constants

        try:
            return download_model(
                model_name,
                cache_dir=str(MODELS_DIR),
                local_files_only=True,
            )
        except Exception:
            pass

        prev_env = os.environ.get("HF_HUB_OFFLINE")
        prev_const = getattr(hf_constants, "HF_HUB_OFFLINE", False)
        try:
            os.environ["HF_HUB_OFFLINE"] = "0"
            hf_constants.HF_HUB_OFFLINE = False
            return download_model(
                model_name,
                cache_dir=str(MODELS_DIR),
                local_files_only=False,
            )
        finally:
            if prev_env is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = prev_env
            hf_constants.HF_HUB_OFFLINE = prev_const

    def load(self, model_name: str, device_pref: str = "auto") -> str:
        """Load WhisperModel onto GPU (int8_float16) or CPU (int8) with automatic fallback."""
        register_cuda_dlls()
        from faster_whisper import WhisperModel

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pref = (device_pref or "auto").lower()

        if self._model is not None and self._model_name == model_name:
            if pref == "auto" or pref == self.active_device:
                return self.active_device

        self.unload()
        model_source = self._resolve_model_source(model_name)

        if pref in ("auto", "cuda"):
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

        # CPU fallback scaled dynamically to host cores
        threads = max(4, min(12, (os.cpu_count() or 8) // 2))
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

    def transcribe(
        self,
        audio_path: Path,
        language: str = "auto",
        progress_callback: Callable[[float, str], None] | None = None,
        classroom_enhance: bool = True,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        """Transcribe an audio file synchronously and emit progress callbacks."""
        model = self._model
        if model is None:
            raise RuntimeError("WhisperModel no está cargado.")

        if cancel_event is not None and cancel_event.is_set():
            raise WhisperCancelledError("Transcripción detenida por el usuario.")

        lang_arg = None if not language or language == "auto" else language
        enhanced_tmp: Path | None = None
        target_audio = audio_path
        if classroom_enhance:
            enhanced_tmp = enhance_audio_for_speech(audio_path)
            if enhanced_tmp is not None:
                target_audio = enhanced_tmp

        try:
            if cancel_event is not None and cancel_event.is_set():
                raise WhisperCancelledError("Transcripción detenida por el usuario.")

            transcribe_kwargs: dict[str, Any] = {
                "language": lang_arg,
                "beam_size": 5,
                "multilingual": (lang_arg is None),
                "condition_on_previous_text": False,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 3,
                "hallucination_silence_threshold": 2.0,
                "vad_filter": True,
                "vad_parameters": {
                    "threshold": 0.35,
                    "min_silence_duration_ms": 700,
                    "speech_pad_ms": 400,
                },
            }
            segments_iter, info = model.transcribe(str(target_audio), **transcribe_kwargs)

            total_duration = float(getattr(info, "duration", 0.0) or 0.0)
            detected_lang = getattr(info, "language", "") or (lang_arg or "")

            segments: list[dict[str, Any]] = []
            text_parts: list[str] = []
            last_norm = ""

            for seg in segments_iter:
                if cancel_event is not None and cancel_event.is_set():
                    raise WhisperCancelledError("Transcripción detenida por el usuario.")
                clean = (seg.text or "").strip()
                if clean:
                    norm = clean.lower()
                    if norm != last_norm:
                        segments.append(
                            {
                                "start": round(float(seg.start), 2),
                                "end": round(float(seg.end), 2),
                                "text": clean,
                            }
                        )
                        text_parts.append(clean)
                        last_norm = norm
                if progress_callback and total_duration > 0:
                    ratio = min(0.99, max(0.01, float(seg.end) / total_duration))
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
        """Immediately release Whisper model weights from VRAM/RAM."""
        if self._model is not None:
            del self._model
            self._model = None
        self._model_name = ""
        self.active_device = ""
        self.active_compute_type = ""
        gc.collect()


whisper_service = WhisperService()
