"""Exclusive GPU/RAM resource orchestrator with idle background unloading."""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable
from uuid import uuid4

from backend.config import MLX_ASR_MODELS, PARAKEET_MODEL_ID
from backend.database import db
from backend.ollama_service import ollama_service
from backend.whisper_service import WhisperCancelledError, whisper_service


class ResourceManager:
    """
    Coordinates exclusive access to the 8 GB RTX 4060 VRAM & 16 GB RAM.

    Guarantees that Whisper and Ollama (Gemma 4) never stay loaded simultaneously,
    and automatically unloads idle models in the background.
    """

    def __init__(self) -> None:
        self._gpu_lock = asyncio.Lock()
        self.phase: str = "idle"  # idle | unloading | whisper | llm | podcast
        self.active_model: str = ""
        self.active_device: str = ""
        self.podcast_active: bool = False
        self.podcast_id: str = ""
        self.podcast_stage: str = ""
        self.podcast_progress: float = 0.0
        self.last_activity_ts: float = time.monotonic()
        self._watchdog_task: asyncio.Task[None] | None = None
        self._active_tasks: set[asyncio.Task[Any]] = set()
        self._active_cancel_events: set[threading.Event] = set()
        self._operation: tuple[dict[str, str], asyncio.Task[Any], threading.Event | None] | None = None

    @asynccontextmanager
    async def _run_operation(
        self,
        kind: str,
        *,
        label: str = "",
        cancel_event: threading.Event | None = None,
    ) -> AsyncIterator[None]:
        async with self._gpu_lock:
            task = asyncio.current_task()
            if task is not None:
                self._operation = ({"id": uuid4().hex, "kind": kind, "label": label}, task, cancel_event)
            try:
                yield
            finally:
                self._operation = None

    def stop_operation(self, operation_id: str) -> bool:
        if self._operation is None:
            return False
        info, task, cancel_event = self._operation
        if info["id"] != operation_id or task.done() or task.cancelling():
            return False
        if cancel_event is not None:
            cancel_event.set()
        return task.cancel()

    def _register_current_task(self) -> asyncio.Task[Any] | None:
        task = asyncio.current_task()
        if task is not None:
            self._active_tasks.add(task)
        return task

    def _unregister_task(self, task: asyncio.Task[Any] | None) -> None:
        if task is not None:
            self._active_tasks.discard(task)

    def touch_activity(self) -> None:
        self.last_activity_ts = time.monotonic()

    def is_podcast_busy(self) -> bool:
        return bool(self.podcast_active)

    def get_state(self) -> dict[str, Any]:
        operation = None
        if self._operation is not None:
            info, task, _ = self._operation
            operation = {**info, "stopping": bool(task.cancelling())}
        return {
            "phase": self.phase,
            "active_model": self.active_model,
            "active_device": self.active_device,
            "active_operation": operation,
            "whisper_loaded": whisper_service.is_loaded,
            "idle_seconds": int(max(0.0, time.monotonic() - self.last_activity_ts)),
            "podcast_active": self.podcast_active,
            "podcast_id": self.podcast_id,
            "podcast_stage": self.podcast_stage,
            "podcast_progress": self.podcast_progress,
        }

    async def transcribe_exclusive(
        self,
        audio_path: Path,
        model_name: str,
        device_pref: str = "auto",
        language: str = "auto",
        progress_callback: Callable[[float, str], None] | None = None,
        classroom_enhance: bool = True,
        keep_loaded: bool = False,
    ) -> dict[str, Any]:
        """Run transcription with exclusive access to memory and release the model afterwards."""
        task = self._register_current_task()
        cancel_ev = threading.Event()
        self._active_cancel_events.add(cancel_ev)
        try:
            async with self._run_operation("audio", cancel_event=cancel_ev):
                if cancel_ev.is_set():
                    raise WhisperCancelledError("Transcripción detenida por el usuario.")
                try:
                    self.phase = "unloading"
                    self.active_model = model_name if model_name == PARAKEET_MODEL_ID or model_name in MLX_ASR_MODELS else f"whisper-{model_name}"
                    self.active_device = ""
                    await ollama_service.unload_all()

                    if cancel_ev.is_set():
                        raise WhisperCancelledError("Transcripción detenida por el usuario.")

                    self.phase = "whisper"

                    def _run_transcription() -> dict[str, Any]:
                        if cancel_ev.is_set():
                            raise WhisperCancelledError("Transcripción detenida por el usuario.")
                        settings = db.get_settings()
                        eco_mode = bool(settings.get("eco_mode", False))
                        eco_threads = int(settings.get("eco_cpu_threads") or 2)
                        whisper_service.eco_cpu_threads = eco_threads if eco_mode else 0
                        used_device = whisper_service.load(model_name=model_name, device_pref=device_pref)
                        self.active_device = used_device
                        return whisper_service.transcribe(
                            audio_path=audio_path,
                            language=language,
                            progress_callback=progress_callback,
                            classroom_enhance=classroom_enhance,
                            cancel_event=cancel_ev,
                        )

                    worker = asyncio.create_task(asyncio.to_thread(_run_transcription))
                    try:
                        result = await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        cancel_ev.set()
                        await asyncio.gather(worker, return_exceptions=True)
                        raise
                    self.touch_activity()
                    return result
                finally:
                    if not keep_loaded:
                        whisper_service.unload()
                        gc.collect()
                    self.phase = "idle"
                    self.active_model = ""
                    self.active_device = ""
        finally:
            self._active_cancel_events.discard(cancel_ev)
            self._unregister_task(task)

    async def summarize_exclusive(
        self,
        messages: list[dict[str, Any]],
        model_name: str,
        num_ctx: int,
        idle_unload_sec: int,
        num_predict: int | None = None,
        operation_label: str = "",
    ) -> str:
        """Ensure Whisper is unloaded, then run Ollama summary generation."""
        task = self._register_current_task()
        try:
            async with self._run_operation("text", label=operation_label):
                try:
                    if whisper_service.is_loaded:
                        self.phase = "unloading"
                        whisper_service.unload()

                    self.phase = "llm"
                    self.active_model = model_name
                    self.active_device = "gpu/ollama"
                    keep_alive = f"{max(30, int(idle_unload_sec))}s"
                    settings = db.get_settings()
                    eco_mode = bool(settings.get("eco_mode", False))
                    eco_cpu_threads = int(settings.get("eco_cpu_threads") or 2)
                    if eco_mode and num_predict is not None:
                        num_predict = min(768, num_predict)
                    text = await ollama_service.generate_complete(
                        model=model_name,
                        messages=messages,
                        num_ctx=num_ctx,
                        keep_alive=keep_alive,
                        eco_mode=eco_mode,
                        eco_cpu_threads=eco_cpu_threads,
                        num_predict=num_predict,
                    )
                    self.touch_activity()
                    return text
                finally:
                    self.phase = "idle"
                    self.active_model = ""
                    self.active_device = ""
        finally:
            self._unregister_task(task)

    async def chat_events_exclusive(
        self,
        messages: list[dict[str, Any]],
        model_name: str,
        num_ctx: int,
        idle_unload_sec: int,
        think: bool = False,
    ) -> AsyncIterator[tuple[str, str]]:
        """Acquire GPU lock, ensure Whisper is unloaded, and stream ('thinking'|'token', text)."""
        task = self._register_current_task()
        try:
            async with self._run_operation("text", label="Chat"):
                try:
                    if whisper_service.is_loaded:
                        self.phase = "unloading"
                        whisper_service.unload()

                    self.phase = "llm"
                    self.active_model = model_name
                    self.active_device = "gpu/ollama"
                    keep_alive = f"{max(30, int(idle_unload_sec))}s"
                    settings = db.get_settings()
                    eco_mode = bool(settings.get("eco_mode", False))
                    eco_cpu_threads = int(settings.get("eco_cpu_threads") or 2)

                    async for kind, text in ollama_service.stream_chat_events(
                        model=model_name,
                        messages=messages,
                        num_ctx=num_ctx,
                        keep_alive=keep_alive,
                        think=think,
                        eco_mode=eco_mode,
                        eco_cpu_threads=eco_cpu_threads,
                    ):
                        self.touch_activity()
                        yield (kind, text)
                finally:
                    self.touch_activity()
                    self.phase = "idle"
                    self.active_model = ""
                    self.active_device = ""
        finally:
            self._unregister_task(task)

    async def chat_stream_exclusive(
        self,
        messages: list[dict[str, Any]],
        model_name: str,
        num_ctx: int,
        idle_unload_sec: int,
    ) -> AsyncIterator[str]:
        """Acquire GPU lock, ensure Whisper is unloaded, and stream Ollama chat tokens."""
        async for kind, text in self.chat_events_exclusive(
            messages=messages,
            model_name=model_name,
            num_ctx=num_ctx,
            idle_unload_sec=idle_unload_sec,
            think=False,
        ):
            if kind == "token" and text:
                yield text

    async def run_podcast_exclusive(
        self,
        podcast_id: str,
        script_coro_factory: Callable[[], Any],
        synth_fn: Callable[[Any], Any],
        on_stage: Callable[[str, float, str], None] | None = None,
        cancel_event: Any = None,
    ) -> Any:
        """
        Run Podcast Studio in strict exclusive mode:
        1. Acquire GPU/RAM lock and unload Whisper.
        2. Generate script with Ollama LLM.
        3. Completely unload Ollama from VRAM/RAM + gc.collect().
        4. Run local TTS synthesis cleanly with free resources, then unload.
        """
        task = self._register_current_task()
        if cancel_event is not None:
            self._active_cancel_events.add(cancel_event)
        self.podcast_active = True
        self.podcast_id = podcast_id
        self.podcast_stage = "Esperando turno exclusivo..."
        self.podcast_progress = 0.05
        if on_stage:
            on_stage("queued", 0.05, self.podcast_stage)

        try:
            async with self._run_operation("podcast", cancel_event=cancel_event):
                try:
                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError()
                    self.phase = "podcast"
                    self.active_model = "Podcast Studio · Guion LLM"
                    self.active_device = "exclusivo"
                    whisper_service.unload()

                    self.podcast_stage = "Escribiendo guion de podcast con modelo local..."
                    self.podcast_progress = 0.15
                    if on_stage:
                        on_stage("scripting", 0.15, self.podcast_stage)

                    script_turns = await script_coro_factory()
                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError()

                    self.podcast_stage = "Liberando VRAM de Ollama antes de cargar voces..."
                    self.podcast_progress = 0.38
                    if on_stage:
                        on_stage("synthesizing", 0.38, self.podcast_stage)

                    await ollama_service.unload_all()
                    gc.collect()
                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError()

                    self.active_model = "Podcast Studio · Voces Kokoro-82M"
                    self.active_device = "local/cpu"
                    self.podcast_stage = "Sintetizando voces locales (Kokoro-82M)..."
                    self.podcast_progress = 0.45

                    result = await asyncio.to_thread(synth_fn, script_turns)
                    self.touch_activity()
                    return result
                finally:
                    if cancel_event is not None and cancel_event.is_set():
                        try:
                            await ollama_service.unload_all()
                        except Exception:
                            pass
                    whisper_service.unload()
                    gc.collect()
                    self.podcast_active = False
                    self.podcast_id = ""
                    self.podcast_stage = ""
                    self.podcast_progress = 0.0
                    self.phase = "idle"
                    self.active_model = ""
                    self.active_device = ""
        finally:
            if cancel_event is not None:
                self._active_cancel_events.discard(cancel_event)
            self._unregister_task(task)

    async def force_stop_all(self) -> dict[str, Any]:
        """Forcibly stop all active LLM streams, Whisper runs, podcasts, and queued tasks."""
        for ev in list(self._active_cancel_events):
            try:
                ev.set()
            except Exception:
                pass

        aborted_streams = 0
        try:
            aborted_streams = await ollama_service.abort_active()
        except Exception:
            pass

        current = asyncio.current_task()
        cancelled_tasks = 0
        for task in list(self._active_tasks):
            if task is not None and task is not current and not task.done() and not task.cancelling():
                task.cancel()
                cancelled_tasks += 1

        await asyncio.sleep(0.05)

        if not self._gpu_lock.locked():
            whisper_service.unload()
            gc.collect()
            self.phase = "idle"
            self.active_model = ""
            self.active_device = ""
            self.podcast_active = False
            self.podcast_id = ""
            self.podcast_stage = ""
            self.podcast_progress = 0.0

        return {
            "aborted_streams": aborted_streams,
            "cancelled_tasks": cancelled_tasks,
            "resources": self.get_state(),
        }

    async def unload_everything(self) -> None:
        """Force immediate unloading of all AI models from VRAM and RAM."""
        async with self._gpu_lock:
            self.phase = "unloading"
            whisper_service.unload()
            await ollama_service.unload_all()
            gc.collect()
            self.phase = "idle"
            self.active_model = ""
            self.active_device = ""

    def start_idle_watchdog(self) -> None:
        """Start background loop that frees VRAM/RAM after configured idle timeout."""
        if self._watchdog_task is None or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._idle_watchdog_loop())

    async def stop_idle_watchdog(self) -> None:
        if self._watchdog_task is not None and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

    async def _idle_watchdog_loop(self) -> None:
        while True:
            await asyncio.sleep(30)
            try:
                if self._gpu_lock.locked() or self.phase != "idle":
                    continue
                settings = db.get_settings()
                timeout_sec = max(30, int(settings.get("idle_unload_sec", 180)))
                idle_elapsed = time.monotonic() - self.last_activity_ts
                if idle_elapsed < timeout_sec:
                    continue

                running_ollama = await ollama_service.get_running_models()
                if whisper_service.is_loaded or running_ollama:
                    await self.unload_everything()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass


resource_manager = ResourceManager()
