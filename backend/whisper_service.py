"""Local Whisper and Parakeet transcription and ffmpeg audio preparation."""

from __future__ import annotations

import gc
import logging
import os
import re
import site
import subprocess
import sysconfig
import threading
from pathlib import Path
from typing import Any, Callable

from backend.config import (
    HTML5_PLAYABLE_EXTENSIONS,
    MODELS_DIR,
    PARAKEET_MODEL_ID,
    PARAKEET_MODEL_REPO,
    PARAKEET_MODELS_DIR,
    PARAKEET_SUPPORTED,
    WHISPER_METAL_SUPPORTED,
    WHISPER_MODELS,
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
    """Manage local transcription models and release them between workloads."""

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
        """Load a transcription model on Metal, CUDA, or CPU."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        pref = (device_pref or "auto").lower()

        if self._model is not None and self._model_name == model_name:
            if pref == "auto" or pref == self.active_device:
                return self.active_device

        self.unload()
        if model_name == PARAKEET_MODEL_ID:
            if not PARAKEET_SUPPORTED:
                raise RuntimeError("Parakeet v3 requiere Apple Silicon y el paquete parakeet-mlx.")
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
            pref == "auto"
            and WHISPER_METAL_SUPPORTED
            and any(model["id"] == model_name for model in WHISPER_MODELS)
        ):
            try:
                self._model = self._load_metal(model_name)
            except Exception as exc:
                logger.warning("No se pudo cargar Whisper con Metal. Usando CPU: %s", exc)
            else:
                self._model_name = model_name
                self.active_device = "metal"
                return self.active_device

        register_cuda_dlls()
        from faster_whisper import WhisperModel

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
            raise RuntimeError("El modelo de transcripción no está cargado.")
        use_parakeet = self._model_name == PARAKEET_MODEL_ID
        use_whisper_metal = not use_parakeet and self.active_device == "metal"

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
        clear_mlx_cache = self._model_name == PARAKEET_MODEL_ID and self.active_device == "metal"
        if self._model is not None:
            del self._model
            self._model = None
        self._model_name = ""
        self.active_device = ""
        self.active_compute_type = ""
        gc.collect()
        if clear_mlx_cache:
            import mlx.core as mx

            mx.clear_cache()


whisper_service = WhisperService()
