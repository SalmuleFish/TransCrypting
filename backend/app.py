"""FastAPI application and background processing pipeline for TransCrypting."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import (
    AUDIO_DIR,
    DEFAULT_WHISPER_MODEL,
    FRONTEND_DIR,
    PARAKEET_MODEL_ID,
    PODCASTS_DIR,
    SOURCES_DIR,
    SUPPORTED_AUDIO_EXTENSIONS,
    SUPPORTED_SOURCE_EXTENSIONS,
    TTS_MODELS_DIR,
    WHISPER_METAL_SUPPORTED,
    ensure_directories,
    list_available_whisper_models,
)
from backend.context_manager import (
    build_chat_messages,
    build_combined_summary_messages,
    build_open_answer_grading_messages,
    build_quiz_generation_messages,
    build_quiz_rebalance_messages,
    build_recording_summary_messages,
    build_source_summary_messages,
    build_suggestions_messages,
    estimate_session_context_usage,
    extract_relevant_excerpt,
    format_transcript_with_timestamps,
    is_true_false_unbalanced,
    parse_grade_json,
    parse_quiz_json,
    parse_suggestions_json,
    shuffle_quiz_mcq_options,
)
from backend.database import db
from backend.hardware import get_system_stats
from backend.ollama_service import GenerationAbortedError, VisionUnsupportedError, ollama_service
from backend.podcast_service import (
    PodcastCancelledError,
    build_podcast_script_messages,
    parse_podcast_script_json,
    synthesize_podcast_audio,
)
from backend.resource_manager import resource_manager
from backend.source_service import (
    build_image_analysis_messages,
    chunk_pages,
    chunk_recording_for_index,
    detect_source_type,
    extract_document_pages,
    extract_image_text_windows_ocr,
    is_meaningful_page_text,
    pages_to_full_text,
    prepare_image_base64,
)
from backend.whisper_service import WhisperCancelledError, _find_ffmpeg_exe, prepare_audio_file


class SessionCreateRequest(BaseModel):
    title: str = "Nueva conversación"


class SessionUpdateRequest(BaseModel):
    title: str


class SettingsUpdateRequest(BaseModel):
    ollama_model: str | None = None
    whisper_model: str | None = None
    whisper_device: str | None = None
    whisper_language: str | None = None
    num_ctx: int | None = None
    idle_unload_sec: int | None = None
    classroom_audio_enhance: bool | None = None
    enable_thinking: bool | None = None
    eco_mode: bool | None = None
    eco_cpu_threads: int | None = None
    theme: str | None = None


class ChatRequest(BaseModel):
    question: str


class ResummarizeRequest(BaseModel):
    recording_id: str | None = None
    source_id: str | None = None


class EnableToggleRequest(BaseModel):
    enabled: bool


class PodcastCreateRequest(BaseModel):
    title: str = ""
    focus_prompt: str = ""
    style: str = "conversation"
    length: str = "medium"
    language: str = "es"
    recording_ids: list[str] | None = None
    source_ids: list[str] | None = None


_podcast_jobs: dict[str, tuple[asyncio.Task[None], threading.Event]] = {}


def _require_not_podcast_busy() -> None:
    if resource_manager.is_podcast_busy():
        raise HTTPException(
            status_code=409,
            detail=(
                "Se está generando un podcast en modo exclusivo por rendimiento. "
                "Espera a que termine para realizar otras tareas de IA."
            ),
        )


class QuizGenerateRequest(BaseModel):
    num_questions: int = 5
    mode: str = "mixed"
    difficulty: str = "medium"
    question_style: str = "conceptual"
    focus_prompt: str = ""
    include_hints: bool = True
    recording_ids: list[str] | None = None
    source_ids: list[str] | None = None


class QuizGradeRequest(BaseModel):
    question: str
    expected_answer: str = ""
    timestamp: str = ""
    user_answer: str


class ProcessingPipeline:
    """Sequential background worker for audio transcription, document/image indexing, and summaries."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None
        self._current_item_task: asyncio.Task[None] | None = None
        self._current_item: tuple[str, str] | None = None
        self._stopping: bool = False

    def start(self) -> None:
        self._stopping = False
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        self._stopping = True
        if self._current_item_task is not None and not self._current_item_task.done():
            self._current_item_task.cancel()
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def cancel_current_and_clear(self) -> int:
        """Drain pending queue items and cancel the currently running pipeline item without stopping the worker."""
        cancelled = 0
        while not self._queue.empty():
            try:
                kind, item_id = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                if kind == "source":
                    db.update_source_error(item_id, "Procesamiento detenido manualmente.")
                else:
                    db.update_recording_error(item_id, "Procesamiento detenido manualmente.")
                cancelled += 1
            finally:
                self._queue.task_done()

        if self._current_item_task is not None and not self._current_item_task.done():
            self._current_item_task.cancel()
            cancelled += 1
        return cancelled

    def enqueue(self, recording_id: str) -> None:
        self._queue.put_nowait(("recording", recording_id))

    def enqueue_source(self, source_id: str) -> None:
        self._queue.put_nowait(("source", source_id))

    async def _worker_loop(self) -> None:
        while True:
            kind, item_id = await self._queue.get()
            self._current_item = (kind, item_id)
            try:
                coro = (
                    self._process_source(item_id)
                    if kind == "source"
                    else self._process_recording(item_id)
                )
                self._current_item_task = asyncio.create_task(coro)
                await self._current_item_task
            except asyncio.CancelledError:
                if self._stopping:
                    raise
                if kind == "source":
                    src = db.get_source(item_id)
                    if src and src.get("status") in ("queued", "extracting", "indexing", "summarizing"):
                        if src.get("extracted_text") and db.count_session_chunks(src["session_id"], enabled_only=False) > 0:
                            db.update_source_summary(
                                item_id,
                                "",
                                status="ready",
                                progress=1.0,
                                stage="Indexado (resumen detenido)",
                            )
                        else:
                            db.update_source_error(item_id, "Procesamiento detenido manualmente.")
                else:
                    rec = db.get_recording(item_id)
                    if rec and rec.get("status") in ("queued", "preparing", "transcribing", "summarizing"):
                        if rec.get("transcript_text"):
                            db.update_recording_summary(
                                item_id,
                                "",
                                status="ready",
                                progress=1.0,
                                stage="Transcripción lista (resumen detenido)",
                            )
                        else:
                            db.update_recording_error(item_id, "Transcripción detenida manualmente.")
            except Exception as exc:
                if kind == "source":
                    db.update_source_error(item_id, f"Error inesperado: {exc}")
                else:
                    db.update_recording_error(item_id, f"Error inesperado: {exc}")
            finally:
                self._current_item_task = None
                self._current_item = None
                self._queue.task_done()

    async def _process_recording(self, rec_id: str) -> None:
        rec = db.get_recording(rec_id)
        if not rec:
            return

        session_id = rec["session_id"]
        src_path = Path(rec["stored_path"])
        if not src_path.exists():
            db.update_recording_error(rec_id, "No se encontró el archivo de audio guardado.")
            return

        settings = db.get_settings()
        whisper_model = settings.get("whisper_model") or DEFAULT_WHISPER_MODEL
        engine_label = "Parakeet v3" if whisper_model == PARAKEET_MODEL_ID else f"Whisper ({whisper_model})"
        whisper_device = settings.get("whisper_device") or "auto"
        whisper_lang = settings.get("whisper_language") or "auto"
        classroom_enhance = bool(settings.get("classroom_audio_enhance", True))
        num_ctx = int(settings.get("num_ctx") or 8192)
        idle_unload_sec = int(settings.get("idle_unload_sec") or 180)

        # 1. Prepare audio & probe duration
        db.update_recording_progress(rec_id, "preparing", 0.05, "Preparando audio...")
        duration_sec, playable_path = await asyncio.to_thread(
            prepare_audio_file, src_path, src_path.parent
        )
        db.update_recording_progress(
            rec_id,
            "transcribing",
            0.08,
            f"Cargando {engine_label}...",
            duration_sec=duration_sec,
            playable_path=str(playable_path),
        )

        # 2. Exclusive transcription
        last_reported = 0.08

        def _on_progress(ratio: float, detected_lang: str) -> None:
            nonlocal last_reported
            mapped = 0.10 + ratio * 0.65
            if mapped - last_reported >= 0.03:
                last_reported = mapped
                pct = int(ratio * 100)
                db.update_recording_progress(
                    rec_id,
                    "transcribing",
                    mapped,
                    f"Transcribiendo audio ({pct}%)...",
                )

        try:
            transcript_result = await resource_manager.transcribe_exclusive(
                audio_path=src_path,
                model_name=whisper_model,
                device_pref=whisper_device,
                language=whisper_lang,
                progress_callback=_on_progress,
                classroom_enhance=classroom_enhance,
            )
        except (WhisperCancelledError, asyncio.CancelledError):
            db.update_recording_error(rec_id, "Transcripción detenida manualmente.")
            return
        except Exception as exc:
            db.update_recording_error(rec_id, f"Error al transcribir con {engine_label}: {exc}")
            return

        # Re-check recording still exists (user might have deleted it while processing)
        if not db.get_recording(rec_id):
            return

        transcript_text = transcript_result["text"]
        segments = transcript_result["segments"]
        detected_lang = transcript_result["language"]
        final_duration = transcript_result["duration_sec"] or duration_sec

        db.update_recording_transcript(
            rec_id=rec_id,
            transcript_text=transcript_text,
            segments=segments,
            language=detected_lang,
            duration_sec=final_duration,
            whisper_model=whisper_model,
        )

        rec_indexed = db.get_recording(rec_id)
        if rec_indexed:
            rec_chunks = chunk_recording_for_index(rec_indexed)
            db.replace_item_chunks(
                session_id=session_id,
                item_kind="recording",
                item_id=rec_id,
                filename=rec["filename"],
                source_type="audio",
                chunks=rec_chunks,
            )

        if bool(settings.get("eco_mode", False)):
            db.update_recording_summary(
                rec_id=rec_id,
                summary_text="",
                status="ready",
                progress=1.0,
                stage="Indexado (Modo Eco)",
            )
            return

        # 3. Summarize with local Ollama LLM
        db.update_recording_progress(
            rec_id,
            "summarizing",
            0.80,
            "Generando resumen con modelo local...",
        )

        try:
            ollama_model = await ollama_service.resolve_default_model(
                settings.get("ollama_model", "")
            )
            if not settings.get("ollama_model"):
                db.update_settings({"ollama_model": ollama_model})

            summary_msgs = build_recording_summary_messages(
                filename=rec["filename"],
                transcript_text=transcript_text,
                duration_sec=final_duration,
                max_input_chars=max(8000, int(num_ctx * 2.2)),
            )
            summary_text = await resource_manager.summarize_exclusive(
                messages=summary_msgs,
                model_name=ollama_model,
                num_ctx=num_ctx,
                idle_unload_sec=idle_unload_sec,
            )
        except (GenerationAbortedError, asyncio.CancelledError):
            db.update_recording_summary(
                rec_id=rec_id,
                summary_text="",
                status="ready",
                progress=1.0,
                stage="Transcripción lista (resumen detenido)",
            )
            return
        except Exception as exc:
            db.update_recording_error(
                rec_id,
                f"Transcripción lista, pero falló el resumen con Ollama: {exc}",
            )
            return

        if not db.get_recording(rec_id):
            return

        other_ready_recs = [
            r for r in db.list_recordings(session_id)
            if r.get("id") != rec_id and r.get("status") == "ready" and r.get("enabled", True)
        ]
        other_ready_srcs = [
            s for s in db.list_sources(session_id)
            if s.get("status") == "ready" and s.get("enabled", True)
        ]
        if other_ready_recs or other_ready_srcs:
            db.update_recording_summary(
                rec_id=rec_id,
                summary_text=summary_text,
                status="summarizing",
                progress=0.92,
                stage="Actualizando resumen integrado...",
            )
            try:
                await regenerate_combined_summary_if_needed(session_id, extra_rec_id=rec_id)
            except (GenerationAbortedError, asyncio.CancelledError):
                db.update_recording_summary(
                    rec_id=rec_id,
                    summary_text=summary_text,
                    status="ready",
                    progress=1.0,
                    stage="Completado (resumen integrado detenido)",
                )
                return

        db.update_recording_summary(
            rec_id=rec_id,
            summary_text=summary_text,
            status="ready",
            progress=1.0,
            stage="Completado",
        )

        try:
            await generate_and_save_suggestions(session_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _process_source(self, source_id: str) -> None:
        src = db.get_source(source_id)
        if not src:
            return

        session_id = src["session_id"]
        src_path = Path(src["stored_path"])
        if not src_path.exists():
            db.update_source_error(source_id, "No se encontró el archivo guardado en disco.")
            return

        settings = db.get_settings()
        num_ctx = int(settings.get("num_ctx") or 8192)
        idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
        eco_mode = bool(settings.get("eco_mode", False))
        source_type = src.get("source_type") or detect_source_type(src["filename"])

        ollama_model = ""
        if not eco_mode:
            ollama_model = await ollama_service.resolve_default_model(
                settings.get("ollama_model", "")
            )
            if not settings.get("ollama_model") and ollama_model:
                db.update_settings({"ollama_model": ollama_model})

        if source_type == "image":
            db.update_source_progress(
                source_id, "extracting", 0.18, "Analizando imagen una sola vez..."
            )
            try:
                image_b64 = await asyncio.to_thread(prepare_image_base64, src_path)
            except Exception as exc:
                db.update_source_error(source_id, f"No se pudo leer la imagen: {exc}")
                return

            extracted_text = ""
            used_vision = False
            if not eco_mode and ollama_model:
                try:
                    vision_msgs = build_image_analysis_messages(src["filename"], image_b64)
                    extracted_text = await resource_manager.summarize_exclusive(
                        messages=vision_msgs,
                        model_name=ollama_model,
                        num_ctx=min(num_ctx, 8192),
                        idle_unload_sec=idle_unload_sec,
                    )
                    used_vision = bool(extracted_text.strip())
                except (GenerationAbortedError, asyncio.CancelledError):
                    db.update_source_error(source_id, "Procesamiento detenido manualmente.")
                    return
                except VisionUnsupportedError:
                    extracted_text = ""
                except Exception:
                    extracted_text = ""

            if not used_vision:
                db.update_source_progress(
                    source_id,
                    "extracting",
                    0.42,
                    "Extrayendo texto con OCR local (CPU)...",
                )
                ocr_text = await asyncio.to_thread(extract_image_text_windows_ocr, src_path)
                if is_meaningful_page_text(ocr_text):
                    extracted_text = f"Texto extraído mediante OCR local:\n{ocr_text}"
                else:
                    db.delete_item_chunks("source", source_id)
                    db.update_source_content(source_id, "", [], page_count=1)
                    db.update_source_error(
                        source_id,
                        "No se detectó texto legible por OCR y el modelo actual no admite visión multimodal. Selecciona un modelo con visión (por ejemplo gemma4:e4b) y pulsa Reprocesar.",
                    )
                    return

            pages = [{"locator": src["filename"], "label": "Imagen", "text": extracted_text}]
            db.update_source_content(source_id, extracted_text, pages, page_count=1)
            db.update_source_progress(source_id, "indexing", 0.65, "Indexando contenido en SQLite FTS5...")
            chunks = chunk_pages(pages)
            db.replace_item_chunks(
                session_id=session_id,
                item_kind="source",
                item_id=source_id,
                filename=src["filename"],
                source_type="image",
                chunks=chunks,
            )
        else:
            db.update_source_progress(
                source_id, "extracting", 0.22, "Extrayendo páginas/texto en CPU..."
            )
            try:
                pages = await asyncio.to_thread(extract_document_pages, src_path, source_type)
                extracted_text = pages_to_full_text(pages)
            except Exception as exc:
                db.update_source_error(source_id, f"Error al leer el documento: {exc}")
                return

            if not db.get_source(source_id):
                return

            chunks = chunk_pages(pages)
            if not chunks or not is_meaningful_page_text(extracted_text):
                db.delete_item_chunks("source", source_id)
                db.update_source_content(source_id, "", pages, page_count=len(pages))
                db.update_source_error(
                    source_id,
                    f"No se pudo extraer texto legible de este archivo ({len(pages)} pág./secc. sin texto seleccionable ni OCR). Si es un documento escaneado, prueba exportarlo con OCR o subir una captura .png/.jpg con un modelo de visión.",
                )
                return

            db.update_source_content(
                source_id, extracted_text, pages, page_count=len(pages)
            )
            db.update_source_progress(
                source_id, "indexing", 0.50, "Construyendo índice FTS5 local...", page_count=len(pages)
            )
            db.replace_item_chunks(
                session_id=session_id,
                item_kind="source",
                item_id=source_id,
                filename=src["filename"],
                source_type=source_type,
                chunks=chunks,
            )

        if not db.get_source(source_id):
            return

        if eco_mode:
            db.update_source_summary(
                source_id=source_id,
                summary_text="",
                status="ready",
                progress=1.0,
                stage="Indexado (Modo Eco)",
            )
            return

        # Generate concise Markdown summary for this source
        db.update_source_progress(source_id, "summarizing", 0.78, "Resumiendo fuente con modelo local...")
        try:
            if source_type == "image" and used_vision and len(extracted_text) <= 2200:
                summary_text = extracted_text
            else:
                summary_msgs = build_source_summary_messages(
                    filename=src["filename"],
                    source_type=source_type,
                    extracted_text=extracted_text,
                    page_count=len(pages),
                    max_input_chars=max(8000, int(num_ctx * 2.2)),
                )
                summary_text = await resource_manager.summarize_exclusive(
                    messages=summary_msgs,
                    model_name=ollama_model,
                    num_ctx=num_ctx,
                    idle_unload_sec=idle_unload_sec,
                )
        except (GenerationAbortedError, asyncio.CancelledError):
            db.update_source_summary(
                source_id=source_id,
                summary_text="",
                status="ready",
                progress=1.0,
                stage="Indexado (resumen detenido)",
            )
            return
        except Exception as exc:
            db.update_source_error(
                source_id,
                f"Índice FTS5 creado, pero falló el resumen con Ollama: {exc}",
            )
            return

        if not db.get_source(source_id):
            return

        other_ready_recs = [
            r for r in db.list_recordings(session_id)
            if r.get("status") == "ready" and r.get("enabled", True)
        ]
        other_ready_srcs = [
            s for s in db.list_sources(session_id)
            if s.get("id") != source_id and s.get("status") == "ready" and s.get("enabled", True)
        ]
        if other_ready_recs or other_ready_srcs:
            db.update_source_summary(
                source_id=source_id,
                summary_text=summary_text,
                status="summarizing",
                progress=0.92,
                stage="Actualizando resumen integrado...",
            )
            try:
                await regenerate_combined_summary_if_needed(session_id, extra_source_id=source_id)
            except (GenerationAbortedError, asyncio.CancelledError):
                db.update_source_summary(
                    source_id=source_id,
                    summary_text=summary_text,
                    status="ready",
                    progress=1.0,
                    stage="Indexado (resumen integrado detenido)",
                )
                return

        db.update_source_summary(
            source_id=source_id,
            summary_text=summary_text,
            status="ready",
            progress=1.0,
            stage="Indexado y listo",
        )

        try:
            await generate_and_save_suggestions(session_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass


pipeline = ProcessingPipeline()


async def generate_and_save_suggestions(session_id: str) -> list[dict[str, str]]:
    """Ask the local LLM to generate 3 tailored quick-prompt suggestions across active sources."""
    detail = db.get_session_detail(session_id)
    if not detail:
        return []
    ready_recs = [
        r for r in detail.get("recordings", [])
        if r.get("status") == "ready" and r.get("enabled", True)
    ]
    ready_srcs = [
        s for s in detail.get("sources", [])
        if s.get("status") == "ready" and s.get("enabled", True)
    ]
    total_ready = len(ready_recs) + len(ready_srcs)
    if total_ready == 0:
        return []

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    msgs = build_suggestions_messages(
        ready_recs,
        detail.get("combined_summary", ""),
        sources=ready_srcs,
    )
    raw = await resource_manager.summarize_exclusive(
        messages=msgs,
        model_name=ollama_model,
        num_ctx=min(num_ctx, 4096),
        idle_unload_sec=idle_unload_sec,
    )
    prompts = parse_suggestions_json(raw, recording_count=total_ready)
    db.update_session_suggestions(session_id, prompts)
    return prompts


_combined_summary_tasks: dict[str, asyncio.Task[None]] = {}


def _schedule_combined_summary_refresh(session_id: str) -> None:
    """Refresh or clear combined session summary without blocking HTTP responses."""
    prior = _combined_summary_tasks.pop(session_id, None)
    if prior and not prior.done():
        prior.cancel()

    ready_recs = [
        r for r in db.list_recordings(session_id)
        if r.get("enabled", True) and r.get("status") == "ready"
    ]
    ready_srcs = [
        s for s in db.list_sources(session_id)
        if s.get("enabled", True) and s.get("status") == "ready"
    ]
    if len(ready_recs) + len(ready_srcs) < 2:
        db.update_session_summary(session_id, "")
        return

    if bool(db.get_settings().get("eco_mode", False)):
        return

    async def _runner() -> None:
        try:
            await regenerate_combined_summary_if_needed(session_id)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        finally:
            if _combined_summary_tasks.get(session_id) is asyncio.current_task():
                _combined_summary_tasks.pop(session_id, None)

    _combined_summary_tasks[session_id] = asyncio.create_task(_runner())


async def regenerate_combined_summary_if_needed(
    session_id: str,
    extra_rec_id: str | None = None,
    extra_source_id: str | None = None,
) -> str:
    """Build or clear the integrated summary across enabled recordings and sources."""
    recordings = db.list_recordings(session_id)
    sources = db.list_sources(session_id)
    ready_recs = [
        r for r in recordings
        if r.get("enabled", True)
        and (
            r.get("status") == "ready"
            or (extra_rec_id and r.get("id") == extra_rec_id and r.get("summary_text"))
        )
    ]
    ready_srcs = [
        s for s in sources
        if s.get("enabled", True)
        and (
            s.get("status") == "ready"
            or (extra_source_id and s.get("id") == extra_source_id and s.get("summary_text"))
        )
    ]
    if len(ready_recs) + len(ready_srcs) < 2:
        db.update_session_summary(session_id, "")
        return ""

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    msgs = build_combined_summary_messages(ready_recs, sources=ready_srcs)
    combined = await resource_manager.summarize_exclusive(
        messages=msgs,
        model_name=ollama_model,
        num_ctx=num_ctx,
        idle_unload_sec=idle_unload_sec,
    )
    db.update_session_summary(session_id, combined)
    return combined


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_directories()
    db.init_db()
    pipeline.start()
    resource_manager.start_idle_watchdog()
    yield
    for task in list(_combined_summary_tasks.values()):
        task.cancel()
    for task, cancel_ev in list(_podcast_jobs.values()):
        cancel_ev.set()
        task.cancel()
    await pipeline.stop()
    await resource_manager.stop_idle_watchdog()
    await resource_manager.unload_everything()
    await ollama_service.shutdown()
    db.close()


app = FastAPI(title="TransCrypting Local", lifespan=lifespan)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "mode": "100% local"}


@app.get("/api/system/status")
async def system_status(refresh_models: bool = False) -> dict[str, Any]:
    stats = await asyncio.to_thread(get_system_stats)
    ollama_ok = await ollama_service.is_running()
    ollama_models: list[dict[str, Any]] = []
    running_ollama: list[dict[str, Any]] = []
    if ollama_ok:
        ollama_models = await ollama_service.list_models(force_refresh=refresh_models)
        running_ollama = await ollama_service.get_running_models()

    settings = db.get_settings()
    if not settings.get("ollama_model") and ollama_models:
        resolved = await ollama_service.resolve_default_model("")
        settings = db.update_settings({"ollama_model": resolved})

    return {
        "hardware": stats,
        "resources": resource_manager.get_state(),
        "ollama_running": ollama_ok,
        "ollama_models": ollama_models,
        "running_ollama_models": running_ollama,
        "whisper_models": list_available_whisper_models(),
        "whisper_metal_supported": WHISPER_METAL_SUPPORTED,
        "settings": settings,
    }


@app.post("/api/system/unload")
async def system_unload() -> dict[str, Any]:
    await resource_manager.unload_everything()
    stats = await asyncio.to_thread(get_system_stats, True)
    return {"status": "unloaded", "hardware": stats, "resources": resource_manager.get_state()}


@app.post("/api/system/stop")
async def system_stop() -> dict[str, Any]:
    """Forcibly stop all running AI tasks (LLM, Whisper, Podcast, background pipeline)."""
    cancelled_podcasts = 0
    for _pod_id, (task, cancel_ev) in list(_podcast_jobs.items()):
        cancel_ev.set()
        if not task.done():
            task.cancel()
            cancelled_podcasts += 1

    cancelled_combined = 0
    for task in list(_combined_summary_tasks.values()):
        if not task.done():
            task.cancel()
            cancelled_combined += 1

    cancelled_pipeline = pipeline.cancel_current_and_clear()
    stop_info = await resource_manager.force_stop_all()
    return {
        "stopped": True,
        "cancelled_podcasts": cancelled_podcasts,
        "cancelled_combined": cancelled_combined,
        "cancelled_pipeline": cancelled_pipeline,
        **stop_info,
    }


@app.get("/api/settings")
async def get_settings() -> dict[str, Any]:
    settings = db.get_settings()
    if not settings.get("ollama_model"):
        try:
            resolved = await ollama_service.resolve_default_model("")
            settings = db.update_settings({"ollama_model": resolved})
        except Exception:
            pass
    return settings


@app.put("/api/settings")
async def update_settings(payload: SettingsUpdateRequest) -> dict[str, Any]:
    updates = payload.model_dump(exclude_none=True)
    if "num_ctx" in updates:
        updates["num_ctx"] = max(2048, min(32768, int(updates["num_ctx"])))
    if "idle_unload_sec" in updates:
        updates["idle_unload_sec"] = max(30, min(3600, int(updates["idle_unload_sec"])))
    if "eco_cpu_threads" in updates:
        updates["eco_cpu_threads"] = max(1, min(os.cpu_count() or 16, int(updates["eco_cpu_threads"])))
    return db.update_settings(updates)


# --- Sessions ---

@app.get("/api/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    return db.list_sessions()


@app.post("/api/sessions")
async def create_session(payload: SessionCreateRequest) -> dict[str, Any]:
    session = db.create_session(payload.title)
    return db.get_session_detail(session["id"]) or session


def _build_session_response(session_id: str) -> dict[str, Any]:
    db.ensure_recording_chunks_indexed(session_id)
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    settings = db.get_settings()
    detail["context_metrics"] = estimate_session_context_usage(
        recordings=detail.get("recordings", []),
        combined_summary=detail.get("combined_summary", ""),
        messages=detail.get("messages", []),
        num_ctx=int(settings.get("num_ctx") or 8192),
        sources=detail.get("sources", []),
    )
    return detail


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    return _build_session_response(session_id)


@app.post("/api/sessions/{session_id}/suggestions")
async def create_session_suggestions(session_id: str) -> dict[str, Any]:
    _require_not_podcast_busy()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    prompts = await generate_and_save_suggestions(session_id)
    return {"suggested_prompts": prompts}


@app.post("/api/sessions/{session_id}/quiz/generate")
async def generate_session_quiz(
    session_id: str,
    payload: QuizGenerateRequest | None = None,
) -> dict[str, Any]:
    _require_not_podcast_busy()
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    db.ensure_recording_chunks_indexed(session_id)
    all_ready_recs = [
        r for r in detail["recordings"]
        if r.get("status") == "ready" and r.get("transcript_text")
    ]
    all_ready_srcs = [
        s for s in detail.get("sources", [])
        if s.get("status") == "ready" and s.get("extracted_text")
    ]

    req = payload or QuizGenerateRequest()
    if req.recording_ids is not None or req.source_ids is not None:
        rec_filter = set(req.recording_ids or [])
        src_filter = set(req.source_ids or [])
        ready_recs = [r for r in all_ready_recs if r["id"] in rec_filter]
        ready_srcs = [s for s in all_ready_srcs if s["id"] in src_filter]
    else:
        ready_recs = [r for r in all_ready_recs if r.get("enabled", True)]
        ready_srcs = [s for s in all_ready_srcs if s.get("enabled", True)]

    if not ready_recs and not ready_srcs:
        raise HTTPException(
            status_code=400,
            detail="Selecciona al menos una grabación o fuente lista antes de generar el examen.",
        )

    num_questions = max(2, min(15, int(req.num_questions)))
    mode = req.mode if req.mode in ("mixed", "mcq", "true_false", "open") else "mixed"
    difficulty = req.difficulty if req.difficulty in ("easy", "medium", "hard") else "medium"
    question_style = req.question_style if req.question_style in ("conceptual", "practical", "tricky") else "conceptual"
    focus_prompt = (req.focus_prompt or "").strip()
    include_hints = bool(req.include_hints)

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))
    search_query = focus_prompt or "conceptos definiciones examen resumen conclusiones puntos clave"
    indexed_hits = db.search_session_chunks(
        session_id,
        search_query,
        limit=14,
        enabled_only=(req.recording_ids is None and req.source_ids is None),
    )

    msgs = build_quiz_generation_messages(
        recordings=ready_recs,
        combined_summary=detail.get("combined_summary", ""),
        num_questions=num_questions,
        mode=mode,
        num_ctx=num_ctx,
        sources=ready_srcs,
        indexed_chunks=indexed_hits,
        difficulty=difficulty,
        question_style=question_style,
        focus_prompt=focus_prompt,
        include_hints=include_hints,
    )
    try:
        raw = await resource_manager.summarize_exclusive(
            messages=msgs,
            model_name=ollama_model,
            num_ctx=num_ctx,
            idle_unload_sec=idle_unload_sec,
        )
    except (asyncio.CancelledError, GenerationAbortedError) as exc:
        raise HTTPException(
            status_code=499,
            detail="Generación detenida por el usuario.",
        ) from exc
    questions = parse_quiz_json(raw, mode=mode)
    if questions and is_true_false_unbalanced(questions):
        try:
            rebalance_msgs = build_quiz_rebalance_messages(msgs, raw, questions)
            rebalanced_raw = await resource_manager.summarize_exclusive(
                messages=rebalance_msgs,
                model_name=ollama_model,
                num_ctx=num_ctx,
                idle_unload_sec=idle_unload_sec,
            )
            rebalanced_questions = parse_quiz_json(rebalanced_raw, mode=mode)
            if rebalanced_questions and len(rebalanced_questions) >= len(questions) // 2:
                orig_tf = [q for q in questions if q.get("type") == "true_false"]
                new_tf = [q for q in rebalanced_questions if q.get("type") == "true_false"]
                orig_min = min(
                    sum(1 for q in orig_tf if q.get("correct_index") == 0),
                    sum(1 for q in orig_tf if q.get("correct_index") == 1),
                ) if orig_tf else 0
                new_min = min(
                    sum(1 for q in new_tf if q.get("correct_index") == 0),
                    sum(1 for q in new_tf if q.get("correct_index") == 1),
                ) if new_tf else 0
                if new_min > orig_min:
                    questions = rebalanced_questions
        except (asyncio.CancelledError, GenerationAbortedError) as exc:
            raise HTTPException(
                status_code=499,
                detail="Generación detenida por el usuario.",
            ) from exc
        except Exception:
            pass

    if questions:
        questions = shuffle_quiz_mcq_options(questions)

    if not questions:
        raise HTTPException(
            status_code=502,
            detail="El modelo no devolvió preguntas válidas. Intenta generar de nuevo.",
        )
    return {
        "questions": questions,
        "model": ollama_model,
        "config": {
            "num_questions": num_questions,
            "mode": mode,
            "difficulty": difficulty,
            "question_style": question_style,
            "focus_prompt": focus_prompt,
            "include_hints": include_hints,
        },
    }


@app.post("/api/sessions/{session_id}/quiz/grade")
async def grade_session_quiz_answer(
    session_id: str,
    payload: QuizGradeRequest,
) -> dict[str, Any]:
    _require_not_podcast_busy()
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    user_answer = (payload.user_answer or "").strip()
    if not user_answer:
        raise HTTPException(status_code=400, detail="Escribe una respuesta antes de evaluar.")

    db.ensure_recording_chunks_indexed(session_id)
    ready_recs = [
        r for r in detail["recordings"]
        if r.get("status") == "ready" and r.get("enabled", True) and r.get("transcript_text")
    ]
    ready_srcs = [
        s for s in detail.get("sources", [])
        if s.get("status") == "ready" and s.get("enabled", True) and s.get("extracted_text")
    ]
    if not ready_recs and not ready_srcs:
        raise HTTPException(status_code=400, detail="No hay fuentes o grabaciones activas para evaluar.")

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    query_text = f"{payload.question} {payload.expected_answer} {user_answer}"
    indexed_hits = db.search_session_chunks(session_id, query_text, limit=6, enabled_only=True)
    excerpt = extract_relevant_excerpt(
        ready_recs,
        query_text,
        sources=ready_srcs,
        indexed_chunks=indexed_hits,
    )
    msgs = build_open_answer_grading_messages(
        question=payload.question,
        expected_answer=payload.expected_answer,
        timestamp=payload.timestamp,
        user_answer=user_answer,
        relevant_excerpt=excerpt,
    )
    try:
        raw = await resource_manager.summarize_exclusive(
            messages=msgs,
            model_name=ollama_model,
            num_ctx=min(num_ctx, 8192),
            idle_unload_sec=idle_unload_sec,
        )
    except (asyncio.CancelledError, GenerationAbortedError) as exc:
        raise HTTPException(
            status_code=499,
            detail="Evaluación detenida por el usuario.",
        ) from exc
    return parse_grade_json(raw, fallback_timestamp=payload.timestamp)


@app.patch("/api/sessions/{session_id}")
async def update_session(session_id: str, payload: SessionUpdateRequest) -> dict[str, Any]:
    if not db.update_session_title(session_id, payload.title):
        raise HTTPException(status_code=404, detail="Conversación no encontrada o título inválido.")
    return db.get_session_detail(session_id)  # type: ignore[return-value]


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, bool]:
    if not db.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    return {"deleted": True}


# --- Recordings ---

@app.post("/api/recordings/live-transcript")
async def transcribe_live_recording(file: UploadFile = File(...)) -> dict[str, str]:
    _require_not_podcast_busy()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Formato de audio no soportado.")

    limit = 3 * 1024 * 1024
    try:
        data = await file.read(limit + 1)
    finally:
        await file.close()
    if not data or len(data) > limit:
        raise HTTPException(status_code=400, detail="El fragmento debe tener entre 1 byte y 3 MB.")

    with tempfile.TemporaryDirectory(prefix="transcrypting-live-") as directory:
        source = Path(directory) / f"fragment{ext}"
        audio = Path(directory) / "fragment.wav"
        source.write_bytes(data)
        try:
            await asyncio.to_thread(
                subprocess.run,
                [_find_ffmpeg_exe(), "-nostdin", "-loglevel", "error", "-i", str(source),
                 "-t", "15", "-vn", "-ar", "16000", "-ac", "1", str(audio)],
                capture_output=True,
                check=True,
                timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(status_code=422, detail="No se pudo leer el fragmento del micrófono.") from exc

        if resource_manager.phase != "idle":
            raise HTTPException(status_code=409, detail="La IA está ocupada. El micrófono sigue grabando.")

        settings = db.get_settings()
        try:
            result = await resource_manager.transcribe_exclusive(
                audio_path=audio,
                model_name=settings.get("whisper_model") or DEFAULT_WHISPER_MODEL,
                device_pref=settings.get("whisper_device") or "auto",
                language=settings.get("whisper_language") or "auto",
                classroom_enhance=False,
                keep_loaded=True,
            )
        except WhisperCancelledError as exc:
            raise HTTPException(status_code=499, detail="Vista previa detenida.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="No se pudo transcribir el fragmento del micrófono.") from exc
    return {"text": result["text"]}


def _sanitize_filename(name: str) -> str:
    base = Path(name or "audio.wav").name
    cleaned = re.sub(r"[^A-Za-z0-9áéíóúÁÉÍÓÚñÑüÜ._\- ]+", "_", base).strip()
    return cleaned or "grabacion.wav"


@app.post("/api/sessions/{session_id}/recordings")
async def upload_recordings(
    session_id: str,
    files: list[UploadFile] = File(...),
) -> list[dict[str, Any]]:
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    if not files:
        raise HTTPException(status_code=400, detail="Debes adjuntar al menos un archivo de audio.")

    session_audio_dir = AUDIO_DIR / session_id
    session_audio_dir.mkdir(parents=True, exist_ok=True)
    settings = db.get_settings()
    whisper_model = settings.get("whisper_model") or DEFAULT_WHISPER_MODEL

    created: list[dict[str, Any]] = []
    first_stem: str | None = None

    for upload in files:
        safe_name = _sanitize_filename(upload.filename or "grabacion.wav")
        ext = Path(safe_name).suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Formato no soportado ({ext or 'desconocido'}) en '{safe_name}'.",
            )

        rec_id = uuid.uuid4().hex[:12]
        dest_path = session_audio_dir / f"{rec_id}{ext}"

        with dest_path.open("wb") as out_file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                out_file.write(chunk)
        await upload.close()

        rec = db.create_recording(
            session_id=session_id,
            filename=safe_name,
            stored_path=str(dest_path),
            playable_path=str(dest_path),
            whisper_model=whisper_model,
            rec_id=rec_id,
        )
        pipeline.enqueue(rec_id)
        created.append(rec)
        if first_stem is None:
            first_stem = Path(safe_name).stem

    if session.get("title") == "Nueva conversación" and first_stem:
        db.update_session_title(session_id, first_stem[:80])

    return created


@app.post("/api/sessions/{session_id}/recordings/{rec_id}/retranscribe")
async def retranscribe_recording(session_id: str, rec_id: str) -> dict[str, Any]:
    """Re-run transcription and summarization with current settings."""
    _require_not_podcast_busy()
    rec = db.get_recording(rec_id)
    if not rec or rec["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Grabación no encontrada.")
    if rec.get("status") in ("queued", "preparing", "transcribing", "summarizing"):
        raise HTTPException(status_code=400, detail="Esta grabación ya se está procesando.")
    db.update_recording_progress(rec_id, "queued", 0.0, "En cola para re-transcribir...")
    pipeline.enqueue(rec_id)
    return db.get_recording(rec_id)  # type: ignore[return-value]


@app.patch("/api/sessions/{session_id}/recordings/{rec_id}")
async def toggle_recording_enabled(
    session_id: str,
    rec_id: str,
    payload: EnableToggleRequest,
) -> dict[str, Any]:
    rec = db.get_recording(rec_id)
    if not rec or rec["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Grabación no encontrada.")
    db.set_recording_enabled(rec_id, payload.enabled)
    _schedule_combined_summary_refresh(session_id)
    return _build_session_response(session_id)


@app.delete("/api/sessions/{session_id}/recordings/{rec_id}")
async def delete_recording(session_id: str, rec_id: str) -> dict[str, Any]:
    rec = db.get_recording(rec_id)
    if not rec or rec["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Grabación no encontrada.")
    db.delete_recording(rec_id)
    _schedule_combined_summary_refresh(session_id)
    return {"deleted": True}


# --- Sources (PDF / PPTX / DOCX / TXT / Image) ---

@app.post("/api/sessions/{session_id}/sources")
async def upload_sources(
    session_id: str,
    files: list[UploadFile] = File(...),
) -> list[dict[str, Any]]:
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    if not files:
        raise HTTPException(status_code=400, detail="Debes adjuntar al menos un documento o imagen.")

    session_sources_dir = SOURCES_DIR / session_id
    session_sources_dir.mkdir(parents=True, exist_ok=True)

    created: list[dict[str, Any]] = []
    first_stem: str | None = None

    for upload in files:
        safe_name = _sanitize_filename(upload.filename or "documento.pdf")
        ext = Path(safe_name).suffix.lower()
        if ext not in SUPPORTED_SOURCE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Formato de fuente no soportado ({ext or 'desconocido'}) en '{safe_name}'.",
            )

        src_id = uuid.uuid4().hex[:12]
        dest_path = session_sources_dir / f"{src_id}{ext}"
        size_bytes = 0
        with dest_path.open("wb") as out_file:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size_bytes += len(chunk)
                out_file.write(chunk)
        mime_type = upload.content_type or ""
        await upload.close()

        source_type = detect_source_type(safe_name)
        src = db.create_source(
            session_id=session_id,
            filename=safe_name,
            source_type=source_type,
            stored_path=str(dest_path),
            mime_type=mime_type,
            size_bytes=size_bytes,
            source_id=src_id,
        )
        pipeline.enqueue_source(src_id)
        created.append(src)
        if first_stem is None:
            first_stem = Path(safe_name).stem

    if session.get("title") == "Nueva conversación" and first_stem:
        db.update_session_title(session_id, first_stem[:80])

    return created


@app.patch("/api/sessions/{session_id}/sources/{source_id}")
async def toggle_source_enabled(
    session_id: str,
    source_id: str,
    payload: EnableToggleRequest,
) -> dict[str, Any]:
    src = db.get_source(source_id)
    if not src or src["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Fuente no encontrada.")
    db.set_source_enabled(source_id, payload.enabled)
    _schedule_combined_summary_refresh(session_id)
    return _build_session_response(session_id)


@app.post("/api/sessions/{session_id}/sources/{source_id}/reprocess")
async def reprocess_source(session_id: str, source_id: str) -> dict[str, Any]:
    _require_not_podcast_busy()
    src = db.get_source(source_id)
    if not src or src["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Fuente no encontrada.")
    if src.get("status") in ("queued", "extracting", "indexing", "summarizing"):
        raise HTTPException(status_code=400, detail="Esta fuente ya se está procesando.")
    db.update_source_progress(source_id, "queued", 0.0, "En cola para re-indexar...")
    pipeline.enqueue_source(source_id)
    return db.get_source(source_id)  # type: ignore[return-value]


@app.delete("/api/sessions/{session_id}/sources/{source_id}")
async def delete_source(session_id: str, source_id: str) -> dict[str, Any]:
    src = db.get_source(source_id)
    if not src or src["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Fuente no encontrada.")
    db.delete_source(source_id)
    _schedule_combined_summary_refresh(session_id)
    return {"deleted": True}


@app.get("/api/sources/{source_id}/file")
async def serve_source_file(source_id: str) -> FileResponse:
    src = db.get_source(source_id)
    if not src:
        raise HTTPException(status_code=404, detail="Fuente no encontrada.")
    stored = Path(src["stored_path"])
    if not stored.exists():
        raise HTTPException(status_code=404, detail="Archivo fuente no encontrado en disco.")
    return FileResponse(
        path=str(stored),
        filename=src["filename"],
        media_type=src.get("mime_type") or None,
    )


@app.get("/api/sessions/{session_id}/search")
async def search_session_index(session_id: str, q: str = "", limit: int = 12) -> dict[str, Any]:
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    db.ensure_recording_chunks_indexed(session_id)
    hits = db.search_session_chunks(session_id, q, limit=limit, enabled_only=True)
    return {"query": q, "results": hits}


@app.get("/api/recordings/{rec_id}/audio")
async def stream_recording_audio(rec_id: str) -> FileResponse:
    rec = db.get_recording(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Grabación no encontrada.")
    playable = Path(rec["playable_path"])
    if not playable.exists():
        playable = Path(rec["stored_path"])
    if not playable.exists():
        raise HTTPException(status_code=404, detail="Archivo de audio no encontrado en disco.")
    return FileResponse(path=str(playable), filename=rec["filename"])


@app.post("/api/sessions/{session_id}/resummarize")
async def resummarize_session(
    session_id: str,
    payload: ResummarizeRequest | None = None,
) -> dict[str, Any]:
    _require_not_podcast_busy()
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    target_rec_id = payload.recording_id if payload else None
    target_src_id = payload.source_id if payload else None
    recordings = db.list_recordings(session_id)
    sources = db.list_sources(session_id)

    rec_targets = [
        r for r in recordings
        if r.get("transcript_text")
        and r.get("enabled", True)
        and (target_src_id is None)
        and (target_rec_id is None or r["id"] == target_rec_id)
    ]
    src_targets = [
        s for s in sources
        if s.get("extracted_text")
        and s.get("enabled", True)
        and (target_rec_id is None)
        and (target_src_id is None or s["id"] == target_src_id)
    ]
    if not rec_targets and not src_targets:
        raise HTTPException(status_code=400, detail="No hay fuentes ni transcripciones activas para resumir.")

    try:
        for rec in rec_targets:
            msgs = build_recording_summary_messages(
                filename=rec["filename"],
                transcript_text=rec["transcript_text"],
                duration_sec=float(rec.get("duration_sec") or 0.0),
                max_input_chars=max(8000, int(num_ctx * 2.2)),
            )
            summary_text = await resource_manager.summarize_exclusive(
                messages=msgs,
                model_name=ollama_model,
                num_ctx=num_ctx,
                idle_unload_sec=idle_unload_sec,
            )
            db.update_recording_summary(rec["id"], summary_text)

        for src in src_targets:
            msgs = build_source_summary_messages(
                filename=src["filename"],
                source_type=src.get("source_type", "text"),
                extracted_text=src["extracted_text"],
                page_count=int(src.get("page_count") or 0),
                max_input_chars=max(8000, int(num_ctx * 2.2)),
            )
            summary_text = await resource_manager.summarize_exclusive(
                messages=msgs,
                model_name=ollama_model,
                num_ctx=num_ctx,
                idle_unload_sec=idle_unload_sec,
            )
            db.update_source_summary(src["id"], summary_text)

        await regenerate_combined_summary_if_needed(session_id)
    except (asyncio.CancelledError, GenerationAbortedError) as exc:
        raise HTTPException(
            status_code=499,
            detail="Resumen detenido por el usuario.",
        ) from exc
    return _build_session_response(session_id)


@app.get("/api/sessions/{session_id}/export")
async def export_session_markdown(session_id: str) -> Response:
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    lines = [f"# {detail['title']}", ""]
    if detail.get("combined_summary"):
        lines.extend(["## Resumen integrado", "", detail["combined_summary"], ""])

    recs = detail.get("recordings", [])
    if recs:
        lines.append("## Grabaciones")
        lines.append("")
        for idx, rec in enumerate(recs, start=1):
            lines.append(f"### {idx}. {rec['filename']}")
            if rec.get("summary_text"):
                lines.extend(["", "#### Resumen", "", rec["summary_text"], ""])
            if rec.get("transcript_text"):
                lines.extend(
                    ["", "#### Transcripción", "", format_transcript_with_timestamps(rec), ""]
                )

    srcs = detail.get("sources", [])
    if srcs:
        lines.append("## Fuentes documentales e imágenes")
        lines.append("")
        for idx, src in enumerate(srcs, start=1):
            lines.append(f"### {idx}. {src['filename']} ({src.get('source_type', 'doc').upper()})")
            if src.get("summary_text"):
                lines.extend(["", "#### Resumen", "", src["summary_text"], ""])
            if src.get("extracted_text"):
                lines.extend(["", "#### Contenido extraído", "", src["extracted_text"], ""])

    messages = detail.get("messages", [])
    if messages:
        active_model = db.get_settings().get("ollama_model") or "Asistente IA"
        lines.extend(["## Historial de preguntas y respuestas", ""])
        for msg in messages:
            who = "Usuario" if msg["role"] == "user" else active_model
            lines.extend([f"**{who}:**", "", msg["content"], ""])

    safe_title = re.sub(r"[^A-Za-z0-9áéíóúÁÉÍÓÚñÑ._\- ]+", "_", detail["title"]).strip() or "transcript"
    content = "\n".join(lines).encode("utf-8")
    return Response(
        content=content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_title}.md"'},
    )


# --- Chat ---

def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/sessions/{session_id}/chat")
async def chat_with_session(session_id: str, payload: ChatRequest) -> StreamingResponse:
    _require_not_podcast_busy()
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Escribe una pregunta.")

    db.ensure_recording_chunks_indexed(session_id)
    ready_recs = [
        r for r in detail["recordings"]
        if r.get("status") == "ready" and r.get("enabled", True) and r.get("transcript_text")
    ]
    ready_srcs = [
        s for s in detail.get("sources", [])
        if s.get("status") == "ready" and s.get("enabled", True) and s.get("extracted_text")
    ]
    if not ready_recs and not ready_srcs:
        raise HTTPException(
            status_code=400,
            detail="Activa y procesa al menos una grabación o fuente antes de preguntar.",
        )

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    enable_thinking = bool(settings.get("enable_thinking", False))
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    prior_messages = detail["messages"]
    recent_turns = " ".join(m["content"] for m in prior_messages[-4:] if m.get("role") == "user")
    fts_query = f"{question} {recent_turns}".strip()
    indexed_hits = db.search_session_chunks(
        session_id,
        fts_query,
        limit=12,
        enabled_only=True,
        primary_query=question,
    )

    llm_messages, context_mode = build_chat_messages(
        recordings=ready_recs,
        combined_summary=detail.get("combined_summary", ""),
        chat_history=prior_messages,
        user_question=question,
        num_ctx=num_ctx,
        sources=ready_srcs,
        indexed_chunks=indexed_hits,
    )

    user_msg = db.add_message(session_id, "user", question)

    async def event_stream():
        yield _sse(
            {
                "type": "meta",
                "context_mode": context_mode,
                "model": ollama_model,
                "recording_count": len(ready_recs),
                "source_count": len(ready_srcs),
                "user_message": user_msg,
                "thinking_enabled": enable_thinking,
            }
        )
        chunks: list[str] = []
        try:
            async for kind, text in resource_manager.chat_events_exclusive(
                messages=llm_messages,
                model_name=ollama_model,
                num_ctx=num_ctx,
                idle_unload_sec=idle_unload_sec,
                think=enable_thinking,
            ):
                if kind == "thinking":
                    yield _sse({"type": "thinking", "text": text})
                elif kind == "token":
                    chunks.append(text)
                    yield _sse({"type": "token", "text": text})

            full_answer = "".join(chunks).strip()
            if not full_answer:
                full_answer = "No obtuve una respuesta del modelo local."
            saved_assistant = db.add_message(session_id, "assistant", full_answer)
            yield _sse({"type": "done", "message": saved_assistant})
        except (asyncio.CancelledError, GenerationAbortedError):
            partial = "".join(chunks).strip()
            if partial:
                db.add_message(session_id, "assistant", partial)
            yield _sse({"type": "stopped", "partial": bool(partial)})
        except Exception as exc:
            partial = "".join(chunks).strip()
            if partial:
                db.add_message(session_id, "assistant", partial)
            yield _sse({"type": "error", "detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/sessions/{session_id}/messages")
async def clear_chat_history(session_id: str) -> dict[str, Any]:
    """Delete ONLY chat messages; keep recordings, transcripts, and summaries untouched."""
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    deleted_count = db.clear_session_messages(session_id)
    return {"cleared": True, "deleted_messages": deleted_count}


# --- Local Podcast Studio (Exclusive Mode) ---

async def _run_podcast_job(
    session_id: str,
    podcast_id: str,
    selected_recs: list[dict[str, Any]],
    selected_srcs: list[dict[str, Any]],
    focus_prompt: str,
    style: str,
    length: str,
    language: str,
    cancel_event: threading.Event | None = None,
) -> None:
    out_path = PODCASTS_DIR / session_id / f"{podcast_id}.wav"
    try:
        settings = db.get_settings()
        num_ctx = int(settings.get("num_ctx") or 8192)
        idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
        ollama_model = await ollama_service.resolve_default_model(
            settings.get("ollama_model", "")
        )
        if not ollama_model:
            raise RuntimeError("No hay ningún modelo de Ollama disponible para redactar el guion.")

        indexed_hits: list[dict[str, Any]] = []
        if focus_prompt.strip():
            db.ensure_recording_chunks_indexed(session_id)
            indexed_hits = db.search_session_chunks(
                session_id,
                focus_prompt.strip(),
                limit=10,
                enabled_only=False,
            )

        def _on_stage(status: str, progress: float, stage: str) -> None:
            db.update_podcast_progress(podcast_id, status, progress, stage)

        async def _build_script() -> list[dict[str, str]]:
            if cancel_event is not None and cancel_event.is_set():
                raise PodcastCancelledError()
            msgs = build_podcast_script_messages(
                recordings=selected_recs,
                sources=selected_srcs,
                focus_prompt=focus_prompt,
                style=style,
                length=length,
                language=language,
                num_ctx=num_ctx,
                indexed_chunks=indexed_hits,
            )
            keep_alive = f"{max(30, idle_unload_sec)}s"
            eco_mode = bool(settings.get("eco_mode", False))
            eco_cpu_threads = int(settings.get("eco_cpu_threads") or 2)
            raw_script = await ollama_service.generate_complete(
                model=ollama_model,
                messages=msgs,
                num_ctx=num_ctx,
                keep_alive=keep_alive,
                eco_mode=eco_mode,
                eco_cpu_threads=eco_cpu_threads,
                num_predict=1400 if eco_mode else None,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise PodcastCancelledError()
            if not (raw_script or "").strip():
                raise RuntimeError(
                    "El modelo local devolvió una respuesta vacía al redactar el guion. Verifica que el modelo en Ajustes responda correctamente e intenta de nuevo."
                )
            turns = parse_podcast_script_json(raw_script, language=language)
            if not turns:
                raise RuntimeError(
                    "El modelo local no devolvió un guion válido de diálogo. Intenta de nuevo."
                )
            db.update_podcast_script(podcast_id, turns)
            return turns

        def _run_synth(turns: list[dict[str, str]]) -> tuple[float, str]:
            def _synth_progress(pct: float, label: str) -> None:
                resource_manager.podcast_progress = pct
                resource_manager.podcast_stage = label
                db.update_podcast_progress(podcast_id, "synthesizing", pct, label)

            eco_mode = bool(settings.get("eco_mode", False))
            eco_cpu_threads = int(settings.get("eco_cpu_threads") or 2)
            return synthesize_podcast_audio(
                turns=turns,
                output_path=out_path,
                language=language,
                models_dir=TTS_MODELS_DIR,
                progress_callback=_synth_progress,
                cancel_event=cancel_event,
                cpu_threads=eco_cpu_threads if eco_mode else 0,
            )

        duration_sec, tts_engine = await resource_manager.run_podcast_exclusive(
            podcast_id=podcast_id,
            script_coro_factory=_build_script,
            synth_fn=_run_synth,
            on_stage=_on_stage,
            cancel_event=cancel_event,
        )
        db.update_podcast_ready(podcast_id, str(out_path), duration_sec, tts_engine)
    except (asyncio.CancelledError, PodcastCancelledError):
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        db.delete_podcast(podcast_id)
    except Exception as exc:
        db.update_podcast_error(podcast_id, str(exc))
    finally:
        _podcast_jobs.pop(podcast_id, None)


@app.post("/api/sessions/{session_id}/podcasts")
async def create_session_podcast(
    session_id: str,
    payload: PodcastCreateRequest,
) -> dict[str, Any]:
    _require_not_podcast_busy()
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    all_ready_recs = [
        r for r in detail.get("recordings", [])
        if r.get("status") == "ready" and r.get("transcript_text")
    ]
    all_ready_srcs = [
        s for s in detail.get("sources", [])
        if s.get("status") == "ready" and s.get("extracted_text")
    ]

    if payload.recording_ids is not None or payload.source_ids is not None:
        rec_filter = set(payload.recording_ids or [])
        src_filter = set(payload.source_ids or [])
        selected_recs = [r for r in all_ready_recs if r["id"] in rec_filter]
        selected_srcs = [s for s in all_ready_srcs if s["id"] in src_filter]
    else:
        selected_recs = [r for r in all_ready_recs if r.get("enabled", True)]
        selected_srcs = [s for s in all_ready_srcs if s.get("enabled", True)]

    if not selected_recs and not selected_srcs:
        raise HTTPException(
            status_code=400,
            detail="Selecciona al menos una grabación o fuente lista para generar el podcast.",
        )

    style = payload.style if payload.style in ("conversation", "exam_review", "debate", "step_by_step") else "conversation"
    length = payload.length if payload.length in ("short", "medium", "long") else "medium"
    language = "en" if str(payload.language).lower().startswith("en") else "es"
    title = (payload.title or "").strip() or f"Podcast — {detail['title']}"
    refs = [r["filename"] for r in selected_recs] + [s["filename"] for s in selected_srcs]

    pod = db.create_podcast(
        session_id=session_id,
        title=title,
        focus_prompt=payload.focus_prompt,
        style=style,
        length=length,
        language=language,
        source_refs=refs,
    )

    cancel_ev = threading.Event()
    task = asyncio.create_task(
        _run_podcast_job(
            session_id=session_id,
            podcast_id=pod["id"],
            selected_recs=selected_recs,
            selected_srcs=selected_srcs,
            focus_prompt=payload.focus_prompt,
            style=style,
            length=length,
            language=language,
            cancel_event=cancel_ev,
        )
    )
    _podcast_jobs[pod["id"]] = (task, cancel_ev)
    return pod


@app.post("/api/sessions/{session_id}/podcasts/{podcast_id}/cancel")
async def cancel_session_podcast(session_id: str, podcast_id: str) -> dict[str, Any]:
    pod = db.get_podcast(podcast_id)
    if not pod or pod["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Podcast no encontrado.")
    job = _podcast_jobs.get(podcast_id)
    if job:
        task, cancel_ev = job
        cancel_ev.set()
        task.cancel()
    else:
        db.delete_podcast(podcast_id)
    return {"cancelled": True, "podcast_id": podcast_id}


@app.get("/api/podcasts/{podcast_id}/audio")
async def stream_podcast_audio(podcast_id: str) -> FileResponse:
    pod = db.get_podcast(podcast_id)
    if not pod:
        raise HTTPException(status_code=404, detail="Podcast no encontrado.")
    audio_path = Path(pod.get("audio_path") or "")
    if not audio_path.is_file():
        raise HTTPException(status_code=404, detail="Archivo de audio del podcast aún no disponible.")
    return FileResponse(path=str(audio_path), filename=f"{podcast_id}.wav", media_type="audio/wav")


@app.delete("/api/sessions/{session_id}/podcasts/{podcast_id}")
async def delete_session_podcast(session_id: str, podcast_id: str) -> dict[str, bool]:
    pod = db.get_podcast(podcast_id)
    if not pod or pod["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Podcast no encontrado.")
    if resource_manager.is_podcast_busy() and resource_manager.podcast_id == podcast_id:
        raise HTTPException(
            status_code=409,
            detail="No puedes eliminar este podcast mientras se está generando.",
        )
    db.delete_podcast(podcast_id)
    return {"deleted": True}


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
