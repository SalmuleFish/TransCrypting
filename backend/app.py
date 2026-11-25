"""FastAPI application and background processing pipeline for TransCrypting."""

from __future__ import annotations

import asyncio
import json
import re
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
    FRONTEND_DIR,
    SUPPORTED_AUDIO_EXTENSIONS,
    ensure_directories,
    list_available_whisper_models,
)
from backend.context_manager import (
    build_chat_messages,
    build_combined_summary_messages,
    build_open_answer_grading_messages,
    build_quiz_generation_messages,
    build_recording_summary_messages,
    build_suggestions_messages,
    estimate_session_context_usage,
    extract_relevant_excerpt,
    format_transcript_with_timestamps,
    parse_grade_json,
    parse_quiz_json,
    parse_suggestions_json,
)
from backend.database import db
from backend.hardware import get_system_stats
from backend.ollama_service import ollama_service
from backend.resource_manager import resource_manager
from backend.whisper_service import prepare_audio_file


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
    theme: str | None = None


class ChatRequest(BaseModel):
    question: str


class ResummarizeRequest(BaseModel):
    recording_id: str | None = None


class QuizGenerateRequest(BaseModel):
    num_questions: int = 5
    mode: str = "mixed"


class QuizGradeRequest(BaseModel):
    question: str
    expected_answer: str = ""
    timestamp: str = ""
    user_answer: str


class ProcessingPipeline:
    """Sequential background worker for audio preparation, transcription, and summarization."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

    def enqueue(self, recording_id: str) -> None:
        self._queue.put_nowait(recording_id)

    async def _worker_loop(self) -> None:
        while True:
            rec_id = await self._queue.get()
            try:
                await self._process_recording(rec_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                db.update_recording_error(rec_id, f"Error inesperado: {exc}")
            finally:
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
        whisper_model = settings.get("whisper_model") or "large-v3-turbo"
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
            f"Cargando Whisper ({whisper_model})...",
            duration_sec=duration_sec,
            playable_path=str(playable_path),
        )

        # 2. Exclusive transcription with faster-whisper
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
        except Exception as exc:
            db.update_recording_error(rec_id, f"Error al transcribir con Whisper: {exc}")
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

        # 3. Summarize with local Gemma 4 / Ollama
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
        except Exception as exc:
            db.update_recording_error(
                rec_id,
                f"Transcripción lista, pero falló el resumen con Ollama: {exc}",
            )
            return

        if not db.get_recording(rec_id):
            return

        existing_ready = [
            r for r in db.list_recordings(session_id)
            if r.get("id") != rec_id and r.get("status") == "ready"
        ]
        if existing_ready:
            db.update_recording_summary(
                rec_id=rec_id,
                summary_text=summary_text,
                status="summarizing",
                progress=0.92,
                stage="Actualizando resumen integrado...",
            )
            await regenerate_combined_summary_if_needed(session_id, extra_rec_id=rec_id)

        db.update_recording_summary(
            rec_id=rec_id,
            summary_text=summary_text,
            status="ready",
            progress=1.0,
            stage="Completado",
        )

        try:
            await generate_and_save_suggestions(session_id)
        except Exception:
            pass


pipeline = ProcessingPipeline()


async def generate_and_save_suggestions(session_id: str) -> list[dict[str, str]]:
    """Ask Gemma 4 to generate 3 tailored quick-prompt suggestions for this session."""
    detail = db.get_session_detail(session_id)
    if not detail:
        return []
    ready_recs = [r for r in detail["recordings"] if r.get("status") == "ready"]
    if not ready_recs:
        return []

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    msgs = build_suggestions_messages(ready_recs, detail.get("combined_summary", ""))
    raw = await resource_manager.summarize_exclusive(
        messages=msgs,
        model_name=ollama_model,
        num_ctx=min(num_ctx, 4096),
        idle_unload_sec=idle_unload_sec,
    )
    prompts = parse_suggestions_json(raw, recording_count=len(ready_recs))
    db.update_session_suggestions(session_id, prompts)
    return prompts


async def regenerate_combined_summary_if_needed(
    session_id: str,
    extra_rec_id: str | None = None,
) -> str:
    """Build or clear the integrated summary when a session has multiple ready recordings."""
    recordings = db.list_recordings(session_id)
    ready_recs = [
        r for r in recordings
        if r.get("status") == "ready" or (extra_rec_id and r.get("id") == extra_rec_id and r.get("summary_text"))
    ]
    if len(ready_recs) < 2:
        db.update_session_summary(session_id, "")
        return ""

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    msgs = build_combined_summary_messages(ready_recs)
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
        "settings": settings,
    }


@app.post("/api/system/unload")
async def system_unload() -> dict[str, Any]:
    await resource_manager.unload_everything()
    stats = await asyncio.to_thread(get_system_stats, True)
    return {"status": "unloaded", "hardware": stats, "resources": resource_manager.get_state()}


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
    return db.update_settings(updates)


# --- Sessions ---

@app.get("/api/sessions")
async def list_sessions() -> list[dict[str, Any]]:
    return db.list_sessions()


@app.post("/api/sessions")
async def create_session(payload: SessionCreateRequest) -> dict[str, Any]:
    session = db.create_session(payload.title)
    return db.get_session_detail(session["id"]) or session


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str) -> dict[str, Any]:
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    settings = db.get_settings()
    detail["context_metrics"] = estimate_session_context_usage(
        recordings=detail.get("recordings", []),
        combined_summary=detail.get("combined_summary", ""),
        messages=detail.get("messages", []),
        num_ctx=int(settings.get("num_ctx") or 8192),
    )
    return detail


@app.post("/api/sessions/{session_id}/suggestions")
async def create_session_suggestions(session_id: str) -> dict[str, Any]:
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
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    ready_recs = [r for r in detail["recordings"] if r.get("status") == "ready"]
    if not ready_recs:
        raise HTTPException(
            status_code=400,
            detail="Espera a que al menos una grabación termine de procesarse.",
        )

    req = payload or QuizGenerateRequest()
    num_questions = max(2, min(10, int(req.num_questions)))
    mode = req.mode if req.mode in ("mixed", "mcq", "open") else "mixed"

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    msgs = build_quiz_generation_messages(
        recordings=ready_recs,
        combined_summary=detail.get("combined_summary", ""),
        num_questions=num_questions,
        mode=mode,
        num_ctx=num_ctx,
    )
    raw = await resource_manager.summarize_exclusive(
        messages=msgs,
        model_name=ollama_model,
        num_ctx=num_ctx,
        idle_unload_sec=idle_unload_sec,
    )
    questions = parse_quiz_json(raw, mode=mode)
    if not questions:
        raise HTTPException(
            status_code=502,
            detail="El modelo no devolvió preguntas válidas. Intenta generar de nuevo.",
        )
    return {"questions": questions, "model": ollama_model}


@app.post("/api/sessions/{session_id}/quiz/grade")
async def grade_session_quiz_answer(
    session_id: str,
    payload: QuizGradeRequest,
) -> dict[str, Any]:
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    user_answer = (payload.user_answer or "").strip()
    if not user_answer:
        raise HTTPException(status_code=400, detail="Escribe una respuesta antes de evaluar.")

    ready_recs = [r for r in detail["recordings"] if r.get("status") == "ready"]
    if not ready_recs:
        raise HTTPException(status_code=400, detail="No hay grabaciones listas para evaluar.")

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    excerpt = extract_relevant_excerpt(
        ready_recs,
        f"{payload.question} {payload.expected_answer} {user_answer}",
    )
    msgs = build_open_answer_grading_messages(
        question=payload.question,
        expected_answer=payload.expected_answer,
        timestamp=payload.timestamp,
        user_answer=user_answer,
        relevant_excerpt=excerpt,
    )
    raw = await resource_manager.summarize_exclusive(
        messages=msgs,
        model_name=ollama_model,
        num_ctx=min(num_ctx, 8192),
        idle_unload_sec=idle_unload_sec,
    )
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
    whisper_model = settings.get("whisper_model") or "large-v3-turbo"

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
    """Re-run Whisper transcription and summarization with current settings."""
    rec = db.get_recording(rec_id)
    if not rec or rec["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Grabación no encontrada.")
    if rec.get("status") in ("queued", "preparing", "transcribing", "summarizing"):
        raise HTTPException(status_code=400, detail="Esta grabación ya se está procesando.")
    db.update_recording_progress(rec_id, "queued", 0.0, "En cola para re-transcribir...")
    pipeline.enqueue(rec_id)
    return db.get_recording(rec_id)  # type: ignore[return-value]


@app.delete("/api/sessions/{session_id}/recordings/{rec_id}")
async def delete_recording(session_id: str, rec_id: str) -> dict[str, Any]:
    rec = db.get_recording(rec_id)
    if not rec or rec["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="Grabación no encontrada.")
    db.delete_recording(rec_id)

    remaining = [r for r in db.list_recordings(session_id) if r.get("status") == "ready"]
    if len(remaining) < 2:
        db.update_session_summary(session_id, "")
    else:
        try:
            await regenerate_combined_summary_if_needed(session_id)
        except Exception:
            pass
    return {"deleted": True}


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
    session = db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    target_id = payload.recording_id if payload else None
    recordings = db.list_recordings(session_id)
    targets = [
        r
        for r in recordings
        if r.get("transcript_text") and (target_id is None or r["id"] == target_id)
    ]
    if not targets:
        raise HTTPException(status_code=400, detail="No hay transcripciones listas para resumir.")

    for rec in targets:
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

    await regenerate_combined_summary_if_needed(session_id)
    return db.get_session_detail(session_id)  # type: ignore[return-value]


@app.get("/api/sessions/{session_id}/export")
async def export_session_markdown(session_id: str) -> Response:
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    lines = [f"# {detail['title']}", ""]
    if detail.get("combined_summary"):
        lines.extend(["## Resumen integrado", "", detail["combined_summary"], ""])

    lines.append("## Grabaciones")
    lines.append("")
    for idx, rec in enumerate(detail.get("recordings", []), start=1):
        lines.append(f"### {idx}. {rec['filename']}")
        if rec.get("summary_text"):
            lines.extend(["", "#### Resumen", "", rec["summary_text"], ""])
        if rec.get("transcript_text"):
            lines.extend(
                ["", "#### Transcripción", "", format_transcript_with_timestamps(rec), ""]
            )

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
    detail = db.get_session_detail(session_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")

    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Escribe una pregunta.")

    ready_recs = [r for r in detail["recordings"] if r.get("status") == "ready"]
    if not ready_recs:
        raise HTTPException(
            status_code=400,
            detail="Espera a que al menos una grabación termine de transcribirse y resumirse.",
        )

    settings = db.get_settings()
    num_ctx = int(settings.get("num_ctx") or 8192)
    idle_unload_sec = int(settings.get("idle_unload_sec") or 180)
    enable_thinking = bool(settings.get("enable_thinking", False))
    ollama_model = await ollama_service.resolve_default_model(settings.get("ollama_model", ""))

    prior_messages = detail["messages"]
    llm_messages, context_mode = build_chat_messages(
        recordings=ready_recs,
        combined_summary=detail.get("combined_summary", ""),
        chat_history=prior_messages,
        user_question=question,
        num_ctx=num_ctx,
    )

    user_msg = db.add_message(session_id, "user", question)

    async def event_stream():
        yield _sse(
            {
                "type": "meta",
                "context_mode": context_mode,
                "model": ollama_model,
                "recording_count": len(ready_recs),
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


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
