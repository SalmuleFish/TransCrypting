"""Automated core tests for TransCrypting database, context manager, resource lock, and API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.config import MLX_ASR_SUPPORTED, NEMOTRON_MODEL_ID, list_available_whisper_models
from backend.context_manager import (
    build_chat_messages,
    build_combined_summary_messages,
    estimate_session_context_usage,
    format_transcript_for_llm,
    parse_grade_json,
    parse_quiz_json,
    parse_suggestions_json,
)
from backend.database import Database
from backend.resource_manager import ResourceManager


def test_combined_summary_limits_large_inputs_and_keeps_each_active_source() -> None:
    recordings = [
        {"filename": "clase.wav", "summary_text": "Inicio " + " detalle " * 3000 + " Cierre"},
        {"filename": "reunion.wav", "transcript_text": "Acuerdo de la reunión"},
    ]
    sources = [
        {"filename": "notas.pdf", "extracted_text": "Notas complementarias"},
        {"filename": "oculto.pdf", "summary_text": "Fuente desactivada", "enabled": False},
    ]
    prompt = build_combined_summary_messages(recordings, sources, max_input_chars=600)[-1]["content"]
    assert "clase.wav" in prompt and "Inicio " in prompt and " Cierre" in prompt
    excerpt = prompt.split(" ===\n", 1)[1].split("\n\n=== Grabación 2", 1)[0]
    assert len(excerpt) == 200
    assert "reunion.wav" in prompt and "Acuerdo de la reunión" in prompt
    assert "notas.pdf" in prompt and "Notas complementarias" in prompt
    assert "oculto.pdf" not in prompt


def test_clear_chat_history_preserves_recordings_transcripts_and_summaries(tmp_path: Path) -> None:
    test_db = Database(db_path=tmp_path / "test.db", audio_dir=tmp_path / "audio")
    test_db.init_db()

    session = test_db.create_session("Reunión de prueba")
    sid = session["id"]

    audio_file = tmp_path / "audio" / sid / "rec1.wav"
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"RIFFdemo")

    rec = test_db.create_recording(
        session_id=sid,
        filename="reunion_1.wav",
        stored_path=str(audio_file),
        playable_path=str(audio_file),
        whisper_model="large-v3-turbo",
    )
    test_db.update_recording_transcript(
        rec_id=rec["id"],
        transcript_text="El presupuesto aprobado es de cinco mil dólares.",
        segments=[{"start": 0.0, "end": 4.2, "text": "El presupuesto aprobado es de cinco mil dólares."}],
        language="es",
        duration_sec=4.2,
        whisper_model="large-v3-turbo",
    )
    test_db.update_recording_summary(rec["id"], "### Resumen\n- Presupuesto de 5000 dólares.")
    test_db.update_session_summary(sid, "Resumen combinado de prueba")

    test_db.add_message(sid, "user", "¿Cuál es el presupuesto?")
    test_db.add_message(sid, "assistant", "El presupuesto aprobado es de 5000 dólares.")

    deleted = test_db.clear_session_messages(sid)
    assert deleted == 2

    detail = test_db.get_session_detail(sid)
    assert detail is not None
    assert detail["messages"] == []
    assert detail["combined_summary"] == "Resumen combinado de prueba"
    assert len(detail["recordings"]) == 1
    assert detail["recordings"][0]["transcript_text"] == "El presupuesto aprobado es de cinco mil dólares."
    assert detail["recordings"][0]["summary_text"] == "### Resumen\n- Presupuesto de 5000 dólares."
    assert audio_file.exists()
    test_db.close()


def test_context_manager_full_and_bm25_retrieval() -> None:
    rec1 = {
        "filename": "parte1.mp3",
        "duration_sec": 30.0,
        "status": "ready",
        "transcript_text": "La fecha límite del proyecto es el 15 de diciembre.",
        "segments": [{"start": 2.0, "end": 7.0, "text": "La fecha límite del proyecto es el 15 de diciembre."}],
        "summary_text": "Define fecha límite 15 de diciembre.",
    }
    rec2 = {
        "filename": "parte2.mp3",
        "duration_sec": 40.0,
        "status": "ready",
        "transcript_text": "El proveedor elegido es Tecnología Norte.",
        "segments": [{"start": 10.0, "end": 16.0, "text": "El proveedor elegido es Tecnología Norte."}],
        "summary_text": "Selecciona proveedor Tecnología Norte.",
    }

    msgs_full, mode_full = build_chat_messages(
        recordings=[rec1, rec2],
        combined_summary="Resumen combinado.",
        chat_history=[],
        user_question="¿Cuál es el proveedor y la fecha límite?",
        num_ctx=8192,
    )
    assert mode_full == "full"
    assert "parte1.mp3" in msgs_full[0]["content"]
    assert "parte2.mp3" in msgs_full[0]["content"]

    # Force retrieval mode using large transcripts
    long_segments = [
        {"start": float(i * 10), "end": float(i * 10 + 9), "text": f"Bloque {i} relleno general de conversación."}
        for i in range(450)
    ]
    long_segments[200] = {
        "start": 2000.0,
        "end": 2009.0,
        "text": "La clave secreta del servidor local es Alfa-99.",
    }
    long_rec = {
        "filename": "larga.mp3",
        "duration_sec": 4500.0,
        "status": "ready",
        "transcript_text": "\n".join(s["text"] for s in long_segments),
        "segments": long_segments,
        "summary_text": "Grabación extensa sobre infraestructura.",
    }

    msgs_ret, mode_ret = build_chat_messages(
        recordings=[long_rec],
        combined_summary="",
        chat_history=[],
        user_question="¿Cuál es la clave secreta del servidor local?",
        num_ctx=4096,
    )
    assert mode_ret == "retrieval"
    assert "Alfa-99" in msgs_ret[0]["content"]


def test_resource_manager_unloads_ollama_before_whisper_and_frees_whisper() -> None:
    rm = ResourceManager()
    events: list[str] = []

    async def fake_unload_ollama() -> None:
        events.append("ollama_unloaded")

    def fake_load(model_name: str, device_pref: str = "auto") -> str:
        assert rm.get_state()["active_operation"]["kind"] == "audio"
        events.append(f"whisper_loaded:{model_name}:{device_pref}")
        return "cuda"

    def fake_transcribe(
        audio_path: Path,
        language: str = "auto",
        progress_callback=None,
        classroom_enhance: bool = True,
        cancel_event=None,
    ):
        events.append("whisper_transcribed")
        return {"text": "hola", "segments": [], "language": "es", "duration_sec": 1.0}

    def fake_whisper_unload() -> None:
        events.append("whisper_unloaded")

    with (
        patch("backend.resource_manager.db.get_settings", return_value={}),
        patch("backend.resource_manager.ollama_service.unload_all", side_effect=fake_unload_ollama),
        patch("backend.resource_manager.whisper_service.load", side_effect=fake_load),
        patch("backend.resource_manager.whisper_service.transcribe", side_effect=fake_transcribe),
        patch("backend.resource_manager.whisper_service.unload", side_effect=fake_whisper_unload),
    ):
        res = asyncio.run(rm.transcribe_exclusive(Path("dummy.wav"), "large-v3-turbo", "auto", "es"))
        assert res["text"] == "hola"
        assert events == [
            "ollama_unloaded",
            "whisper_loaded:large-v3-turbo:auto",
            "whisper_transcribed",
            "whisper_unloaded",
        ]
        assert rm.phase == "idle"


def test_stop_active_operation_keeps_waiting_generation_and_rejects_stale_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert app_module._auto_summaries_enabled({}) is True
    assert app_module._auto_summaries_enabled({"eco_mode": True}) is False

    manager = ResourceManager()
    abort_mock = AsyncMock(return_value=1)
    monkeypatch.setattr(app_module, "resource_manager", manager)
    monkeypatch.setattr(app_module.ollama_service, "abort_active", abort_mock)
    monkeypatch.setattr("backend.resource_manager.db.get_settings", lambda: {})

    async def run() -> None:
        first_started = asyncio.Event()
        second_started = asyncio.Event()
        finish_second = asyncio.Event()

        async def generate(model, **_kwargs):
            if model == "first":
                first_started.set()
                await asyncio.Future()
            second_started.set()
            await finish_second.wait()
            return "Segundo resumen"

        monkeypatch.setattr(app_module.ollama_service, "generate_complete", generate)
        first = asyncio.create_task(manager.summarize_exclusive([], "first", 8192, 180))
        await first_started.wait()
        operation = manager.get_state()["active_operation"]
        assert operation["kind"] == "text"

        second = asyncio.create_task(manager.summarize_exclusive([], "second", 8192, 180))
        await asyncio.sleep(0)
        assert not second_started.is_set()
        assert await app_module.system_stop_active(operation["id"]) == {"stopped": True}
        assert abort_mock.await_count == 1
        with pytest.raises(asyncio.CancelledError):
            await first

        await second_started.wait()
        assert await app_module.system_stop_active(operation["id"]) == {"stopped": False}
        finish_second.set()
        assert await second == "Segundo resumen"
        assert manager.get_state()["active_operation"] is None

    asyncio.run(asyncio.wait_for(run(), 2))


def test_api_clear_messages_and_multi_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_db = Database(db_path=tmp_path / "api.db", audio_dir=tmp_path / "audio")
    test_db.init_db()

    monkeypatch.setattr(app_module, "db", test_db)
    monkeypatch.setattr("backend.resource_manager.db", test_db)
    monkeypatch.setattr(app_module, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(app_module.pipeline, "enqueue", lambda _rec_id: None)
    monkeypatch.setattr(app_module.pipeline, "start", lambda: None)
    monkeypatch.setattr(app_module.resource_manager, "start_idle_watchdog", lambda: None)
    monkeypatch.setattr(app_module.ollama_service, "shutdown", AsyncMock())

    with TestClient(app_module.app) as client:
        s_resp = client.post("/api/sessions", json={"title": "Nueva conversación"})
        assert s_resp.status_code == 200
        sid = s_resp.json()["id"]

        up_resp = client.post(
            f"/api/sessions/{sid}/recordings",
            files=[
                ("files", ("audio_uno.wav", b"RIFF0001", "audio/wav")),
                ("files", ("audio_dos.mp3", b"ID3demo2", "audio/mpeg")),
            ],
        )
        assert up_resp.status_code == 200
        recs = up_resp.json()
        assert len(recs) == 2

        test_db.update_recording_transcript(recs[0]["id"], "Texto uno", [], "es", 5.0, "small")
        test_db.update_recording_summary(recs[0]["id"], "Resumen uno")
        test_db.add_message(sid, "user", "Hola")
        test_db.add_message(sid, "assistant", "Respuesta")

        clr_resp = client.delete(f"/api/sessions/{sid}/messages")
        assert clr_resp.status_code == 200
        assert clr_resp.json()["deleted_messages"] == 2

        detail = client.get(f"/api/sessions/{sid}").json()
        assert detail["messages"] == []
        assert len(detail["recordings"]) == 2
        assert detail["recordings"][0]["summary_text"] == "Resumen uno"
        assert detail["recordings"][0]["transcript_text"] == "Texto uno"

        # Test retranscribe endpoint
        retrans = client.post(f"/api/sessions/{sid}/recordings/{recs[0]['id']}/retranscribe")
        assert retrans.status_code == 200
        assert retrans.json()["status"] == "queued"

        # Test new boolean settings persistence
        upd = client.put(
            "/api/settings",
            json={"classroom_audio_enhance": False, "enable_thinking": True},
        )
        assert upd.status_code == 200
        assert upd.json()["classroom_audio_enhance"] is False
        assert upd.json()["enable_thinking"] is True

    test_db.close()


def test_paragraph_grouping_and_spotlight_in_full_context() -> None:
    segs = [
        {"start": float(i * 4), "end": float(i * 4 + 3), "text": f"Comentario general número {i} de la clase sobre ética empresarial."}
        for i in range(110)
    ]
    segs[55] = {
        "start": 220.0,
        "end": 224.0,
        "text": "Warren Buffett explicó la regla del fair test para decisiones corporativas.",
    }
    rec = {
        "filename": "clase_etica.m4a",
        "duration_sec": 440.0,
        "status": "ready",
        "transcript_text": "\n".join(s["text"] for s in segs),
        "segments": segs,
        "summary_text": "Clase sobre ética.",
    }
    grouped = format_transcript_for_llm(rec)
    assert grouped.count("[") < len(segs)

    msgs, mode = build_chat_messages(
        recordings=[rec],
        combined_summary="",
        chat_history=[],
        user_question="¿Qué dijo sobre la regla del fair test?",
        num_ctx=8192,
    )
    assert mode == "full"
    assert "FRAGMENTOS MÁS RELEVANTES DETECTADOS PARA ESTA PREGUNTA" in msgs[0]["content"]
    assert "fair test" in msgs[0]["content"]


def test_context_metrics_protection_and_suggestions_parser(tmp_path: Path) -> None:
    short_rec = {
        "filename": "clase1.m4a",
        "duration_sec": 60.0,
        "status": "ready",
        "transcript_text": "Repaso de examen de ética.",
        "segments": [{"start": 0.0, "end": 5.0, "text": "Repaso de examen de ética."}],
        "summary_text": "Resumen breve.",
    }
    m_opt = estimate_session_context_usage([short_rec], "", [], num_ctx=8192)
    assert m_opt["mode"] == "full"
    assert m_opt["health"] == "optimal"

    huge_recs = [
        {
            "filename": f"clase_{i}.m4a",
            "duration_sec": 1200.0,
            "status": "ready",
            "transcript_text": "palabra " * 2500,
            "segments": [],
            "summary_text": "Resumen " * 80,
        }
        for i in range(6)
    ]
    m_prot = estimate_session_context_usage(huge_recs, "Resumen integrado", [], num_ctx=4096)
    assert m_prot["mode"] == "retrieval"
    assert m_prot["health"] == "protected"
    assert m_prot["estimated_tokens"] <= 4096

    parsed = parse_suggestions_json(
        '```json\n[{"label": "Quiz rápido", "prompt": "Hazme 5 preguntas."}]\n```'
    )
    assert parsed == [{"label": "Quiz rápido", "prompt": "Hazme 5 preguntas."}]

    fallback = parse_suggestions_json("no json here", recording_count=2)
    assert len(fallback) == 3

    test_db = Database(db_path=tmp_path / "theme.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    s = test_db.create_session("Clase")
    assert test_db.update_session_suggestions(s["id"], parsed)
    detail = test_db.get_session_detail(s["id"])
    assert detail is not None and detail["suggested_prompts"] == parsed
    updated = test_db.update_settings({"theme": "dark"})
    assert updated["theme"] == "dark"
    test_db.close()


def test_quiz_parsers_and_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw_quiz = """```json
    [
      {
        "id": 1,
        "type": "mcq",
        "question": "¿Cuántas preguntas tendrá el examen?",
        "options": ["10", "20", "30", "40"],
        "correct_index": 2,
        "explanation": "Se dijo que serán 30 preguntas en total.",
        "timestamp": "05:01"
      },
      {
        "id": 2,
        "type": "open",
        "question": "Explica qué es el fair test según la clase.",
        "expected_answer": "Dar acceso y explicar decisiones con motivación y datos.",
        "timestamp": "08:12"
      }
    ]
    ```"""
    parsed_q = parse_quiz_json(raw_quiz, mode="mixed")
    assert len(parsed_q) == 2
    assert parsed_q[0]["type"] == "mcq" and parsed_q[0]["correct_index"] == 2
    assert parsed_q[1]["type"] == "open" and parsed_q[1]["timestamp"] == "08:12"

    grade = parse_grade_json(
        '{"verdict": "correct", "score": 95, "feedback": "Muy bien.", "missing_points": "", "timestamp": "08:12"}'
    )
    assert grade["verdict"] == "correct" and grade["score"] == 95

    test_db = Database(db_path=tmp_path / "quiz.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    monkeypatch.setattr(app_module, "db", test_db)
    monkeypatch.setattr("backend.resource_manager.db", test_db)
    monkeypatch.setattr(app_module.pipeline, "start", lambda: None)
    monkeypatch.setattr(app_module.resource_manager, "start_idle_watchdog", lambda: None)
    monkeypatch.setattr(app_module.ollama_service, "shutdown", AsyncMock())
    monkeypatch.setattr(
        app_module.ollama_service,
        "resolve_default_model",
        AsyncMock(return_value="Gemma-4-E4B-it:latest"),
    )

    s = test_db.create_session("Clase Ética")
    rec = test_db.create_recording(s["id"], "clase.m4a", "a.m4a", "a.m4a")
    test_db.update_recording_transcript(
        rec["id"],
        "El examen tendrá 30 preguntas.",
        [{"start": 301.0, "end": 305.0, "text": "El examen tendrá 30 preguntas."}],
        "es",
        310.0,
        "large-v3",
    )
    test_db.update_recording_summary(rec["id"], "Resumen")

    async def fake_summarize(messages, model_name, num_ctx, idle_unload_sec, **_kwargs):
        content = messages[-1]["content"]
        if "Respuesta del estudiante" in content:
            return '{"verdict": "correct", "score": 90, "feedback": "Exacto.", "missing_points": "", "timestamp": "05:01"}'
        return raw_quiz

    monkeypatch.setattr(app_module.resource_manager, "summarize_exclusive", fake_summarize)

    with TestClient(app_module.app) as client:
        g_resp = client.post(
            f"/api/sessions/{s['id']}/quiz/generate",
            json={
                "num_questions": 2,
                "mode": "mixed",
                "difficulty": "hard",
                "question_style": "practical",
                "focus_prompt": "fair test",
                "include_hints": True,
            },
        )
        assert g_resp.status_code == 200
        assert len(g_resp.json()["questions"]) == 2
        assert g_resp.json()["config"]["difficulty"] == "hard"
        assert g_resp.json()["config"]["question_style"] == "practical"

        tf_parsed = parse_quiz_json(
            '[{"type": "true_false", "question": "El examen tiene 30 preguntas.", "correct_answer": true, "hint": "Mencionado al minuto 5.", "timestamp": "05:01"}]',
            mode="true_false",
        )
        assert len(tf_parsed) == 1
        assert tf_parsed[0]["type"] == "true_false"
        assert tf_parsed[0]["options"] == ["Verdadero", "Falso"]
        assert tf_parsed[0]["correct_index"] == 0
        assert "minuto 5" in tf_parsed[0]["hint"]

        from backend.context_manager import build_quiz_generation_messages

        msgs_custom = build_quiz_generation_messages(
            recordings=[test_db.get_recording(rec["id"])],  # type: ignore[list-item]
            num_questions=4,
            mode="true_false",
            difficulty="hard",
            question_style="tricky",
            focus_prompt="regla del fair test",
            include_hints=True,
        )
        assert "DIFÍCIL" in msgs_custom[1]["content"]
        assert "TRAMPAS DE EXAMEN" in msgs_custom[1]["content"]
        assert "regla del fair test" in msgs_custom[1]["content"]
        assert "true_false" in msgs_custom[1]["content"]

        gr_resp = client.post(
            f"/api/sessions/{s['id']}/quiz/grade",
            json={
                "question": "Explica el fair test",
                "expected_answer": "Acceso y datos",
                "timestamp": "08:12",
                "user_answer": "Es explicar decisiones usando datos.",
            },
        )
        assert gr_resp.status_code == 200
        assert gr_resp.json()["verdict"] == "correct"
        assert gr_resp.json()["score"] == 90

    test_db.close()


def test_modular_whisper_model_discovery(tmp_path: Path) -> None:
    custom_dir = tmp_path / "mi-whisper-aula"
    custom_dir.mkdir(parents=True)
    (custom_dir / "model.bin").write_bytes(b"ct2")
    # HF cache dirs (models--*) and incomplete dirs without model.bin must be ignored
    (tmp_path / "models--Systran--faster-whisper-small").mkdir()
    (tmp_path / "incompleto").mkdir()

    models = list_available_whisper_models(models_dir=tmp_path)
    ids = [m["id"] for m in models]
    assert "large-v3-turbo" in ids
    assert "local:mi-whisper-aula" in ids
    assert "local:incompleto" not in ids
    assert not any(i.startswith("local:models--") for i in ids)


def test_windows_icon_and_taskbar_properties() -> None:
    import ctypes
    import struct
    import sys
    from ctypes import wintypes

    import run as run_module
    from backend.config import FRONTEND_DIR

    icon_path = FRONTEND_DIR / "icon.ico"
    assert icon_path.exists()
    raw = icon_path.read_bytes()
    reserved, ico_type, count = struct.unpack("<HHH", raw[:6])
    assert reserved == 0 and ico_type == 1 and count == 6

    frames: list[tuple[int, int, bytes]] = []
    for i in range(count):
        wb, hb, _, _, _, bpp, size, offset = struct.unpack(
            "<BBBBHHII", raw[6 + i * 16 : 6 + (i + 1) * 16]
        )
        w = 256 if wb == 0 else wb
        h = 256 if hb == 0 else hb
        assert bpp == 32
        frames.append((w, h, raw[offset : offset + size]))

    for w, h, payload in frames:
        if w <= 64:
            assert payload[:4] == struct.pack("<I", 40)
        else:
            assert payload[:8] == b"\x89PNG\r\n\x1a\n"

    if sys.platform != "win32":
        return

    try:
        import clr
    except (ImportError, OSError, RuntimeError):
        pytest.skip("pythonnet/clr no disponible en esta arquitectura")

    clr.AddReference("System.Windows.Forms")
    import System.Windows.Forms as WinForms

    form = WinForms.Form()
    try:
        mock_window = type("MockWindow", (), {"native": form})()
        run_module._apply_windows_titlebar_style(
            mock_window, icon_path, theme="light", apply_icon=True
        )
        hwnd = int(form.Handle.ToInt64())

        user32 = ctypes.windll.user32
        user32.SendMessageW.argtypes = [
            wintypes.HWND,
            wintypes.UINT,
            wintypes.WPARAM,
            wintypes.LPARAM,
        ]
        user32.SendMessageW.restype = ctypes.c_ssize_t
        get_class_long = getattr(user32, "GetClassLongPtrW", None) or getattr(
            user32, "GetClassLongW"
        )
        get_class_long.argtypes = [wintypes.HWND, ctypes.c_int]
        get_class_long.restype = ctypes.c_void_p

        WM_GETICON = 0x007F
        GCLP_HICON = -14
        GCLP_HICONSM = -34

        assert user32.SendMessageW(hwnd, WM_GETICON, 0, 0) != 0
        assert user32.SendMessageW(hwnd, WM_GETICON, 1, 0) != 0
        assert (get_class_long(hwnd, GCLP_HICON) or 0) != 0
        assert (get_class_long(hwnd, GCLP_HICONSM) or 0) != 0
        assert run_module._set_windows_taskbar_properties(hwnd, icon_path) is True
    finally:
        form.Dispose()


def test_fts5_persistent_index_and_source_enable_toggle(tmp_path: Path) -> None:
    test_db = Database(
        db_path=tmp_path / "fts.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
    )
    test_db.init_db()
    session = test_db.create_session("Arquitectura de Computadoras")
    sid = session["id"]

    rec = test_db.create_recording(sid, "clase_intro.m4a", "intro.m4a", "intro.m4a")
    test_db.update_recording_transcript(
        rec["id"],
        "Hoy hablamos sobre registros del procesador y ALU.",
        [{"start": 12.0, "end": 20.0, "text": "Hoy hablamos sobre registros del procesador y ALU."}],
        "es",
        25.0,
        "small",
    )
    test_db.update_recording_summary(rec["id"], "Resumen de registros.")

    # Legacy ready recording without chunks should be lazily indexed
    test_db.ensure_recording_chunks_indexed(sid)
    assert test_db.count_session_chunks(sid, enabled_only=True) == 1

    src = test_db.create_source(sid, "apuntes_cache.pdf", "pdf", "apuntes.pdf", "application/pdf", 2048)
    pages = [
        {"locator": "p. 1", "label": "Página 1", "text": "La arquitectura Von Neumann utiliza memoria compartida."},
        {"locator": "p. 2", "label": "Página 2", "text": "La memoria caché L1 aprovecha la localidad temporal para reducir latencia."},
    ]
    test_db.update_source_content(src["id"], "\n\n".join(p["text"] for p in pages), pages, page_count=2)
    test_db.replace_item_chunks(
        sid,
        "source",
        src["id"],
        "apuntes_cache.pdf",
        "pdf",
        [{"locator": p["locator"], "content": p["text"]} for p in pages],
    )
    test_db.update_source_summary(src["id"], "Resumen sobre caché L1.")

    hits = test_db.search_session_chunks(sid, "¿Cómo reduce latencia la caché L1?", limit=5, enabled_only=True)
    assert len(hits) >= 1
    assert hits[0]["filename"] == "apuntes_cache.pdf"
    assert hits[0]["locator"] == "p. 2"

    # Toggle source disabled -> excluded from enabled-only FTS search and chunk count
    test_db.set_source_enabled(src["id"], False)
    assert test_db.count_session_chunks(sid, enabled_only=True) == 1
    hits_disabled = test_db.search_session_chunks(sid, "caché L1 localidad temporal", limit=5, enabled_only=True)
    assert all(h["item_id"] != src["id"] for h in hits_disabled)

    # Toggle back enabled -> returns cleanly
    test_db.set_source_enabled(src["id"], True)
    assert test_db.count_session_chunks(sid, enabled_only=True) == 3

    # Deleting session cascades source_chunks + triggers cleanly without errors
    assert test_db.delete_session(sid) is True
    assert test_db.count_session_chunks(sid, enabled_only=False) == 0
    test_db.close()


def test_document_and_image_extraction_and_chunking(tmp_path: Path) -> None:
    import base64
    from PIL import Image
    from docx import Document
    from pptx import Presentation
    from pypdf import PdfWriter

    from backend.source_service import (
        chunk_pages,
        detect_source_type,
        extract_document_pages,
        prepare_image_base64,
    )

    assert detect_source_type("tema1.pdf") == "pdf"
    assert detect_source_type("slides.PPTX") == "pptx"
    assert detect_source_type("notas.docx") == "docx"
    assert detect_source_type("apuntes.md") == "text"
    assert detect_source_type("pizarron.PNG") == "image"

    # TXT / MD
    txt_file = tmp_path / "notas.txt"
    txt_file.write_text("Primer bloque de apuntes.\n\nSegundo bloque sobre recursividad.", encoding="utf-8")
    txt_pages = extract_document_pages(txt_file, "text")
    assert txt_pages and txt_pages[0]["locator"] == "sec. 1"
    assert "recursividad" in txt_pages[0]["text"]
    assert chunk_pages(txt_pages)

    # DOCX
    docx_file = tmp_path / "guia.docx"
    doc = Document()
    doc.add_paragraph("Guía de estudio para programación orientada a objetos.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Polimorfismo"
    table.rows[0].cells[1].text = "Múltiples formas de un método"
    doc.save(str(docx_file))
    docx_pages = extract_document_pages(docx_file, "docx")
    assert "programación orientada a objetos" in docx_pages[0]["text"]
    assert "Polimorfismo" in docx_pages[0]["text"]

    # PPTX
    pptx_file = tmp_path / "clase.pptx"
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Complejidad Algorítmica"
    slide.placeholders[1].text = "Búsqueda binaria O(log n)"
    slide.notes_slide.notes_text_frame.text = "Insistir en precondición de arreglo ordenado."
    prs.save(str(pptx_file))
    pptx_pages = extract_document_pages(pptx_file, "pptx")
    assert len(pptx_pages) == 1
    assert pptx_pages[0]["locator"] == "diap. 1"
    assert "Complejidad Algorítmica" in pptx_pages[0]["text"]
    assert "arreglo ordenado" in pptx_pages[0]["text"]

    # PDF structure
    pdf_file = tmp_path / "vacio.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with pdf_file.open("wb") as fh:
        writer.write(fh)
    pdf_pages = extract_document_pages(pdf_file, "pdf")
    assert len(pdf_pages) == 1
    assert pdf_pages[0]["locator"] == "p. 1"

    # Image base64 resize + JPEG encoding
    img_file = tmp_path / "diagrama.png"
    Image.new("RGBA", (2200, 1200), color=(20, 40, 60, 255)).save(img_file)
    b64 = prepare_image_base64(img_file, max_dim=1024)
    decoded = base64.b64decode(b64)
    assert decoded[:2] == b"\xff\xd8"  # JPEG header


def test_sources_api_endpoints_and_document_only_chat_quiz(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_db = Database(
        db_path=tmp_path / "sources_api.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
    )
    test_db.init_db()

    monkeypatch.setattr(app_module, "db", test_db)
    monkeypatch.setattr("backend.resource_manager.db", test_db)
    monkeypatch.setattr(app_module, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(app_module, "SOURCES_DIR", tmp_path / "sources")
    monkeypatch.setattr(app_module.pipeline, "enqueue", lambda _id: None)
    monkeypatch.setattr(app_module.pipeline, "enqueue_source", lambda _id: None)
    monkeypatch.setattr(app_module.pipeline, "start", lambda: None)
    monkeypatch.setattr(app_module.resource_manager, "start_idle_watchdog", lambda: None)
    monkeypatch.setattr(app_module.ollama_service, "shutdown", AsyncMock())
    monkeypatch.setattr(
        app_module.ollama_service,
        "resolve_default_model",
        AsyncMock(return_value="gemma4:e4b"),
    )

    quiz_payload = (
        '[{"id": 1, "type": "mcq", "question": "¿Qué protocolo usa el servidor?", '
        '"options": ["FTP", "HTTP local", "SMTP", "IRC"], "correct_index": 1, '
        '"explanation": "Funciona sobre HTTP local.", "timestamp": "apuntes.md · sec. 1"}]'
    )
    monkeypatch.setattr(
        app_module.resource_manager,
        "summarize_exclusive",
        AsyncMock(return_value=quiz_payload),
    )

    async def fake_chat_events(*_args, **_kwargs):
        yield ("token", "El documento indica HTTP local [apuntes.md · sec. 1].")

    monkeypatch.setattr(app_module.resource_manager, "chat_events_exclusive", fake_chat_events)

    with TestClient(app_module.app) as client:
        sid = client.post("/api/sessions", json={"title": "Cuaderno PDF"}).json()["id"]

        up = client.post(
            f"/api/sessions/{sid}/sources",
            files=[("files", ("apuntes.md", b"# Redes\nEl servidor opera sobre HTTP local en 127.0.0.1.", "text/markdown"))],
        )
        assert up.status_code == 200
        src_id = up.json()[0]["id"]

        # Simulate completed extraction & FTS5 indexing
        test_db.update_source_content(
            src_id,
            "El servidor opera sobre HTTP local en 127.0.0.1.",
            [{"locator": "sec. 1", "label": "Sección 1", "text": "El servidor opera sobre HTTP local en 127.0.0.1."}],
            page_count=1,
        )
        test_db.replace_item_chunks(
            sid,
            "source",
            src_id,
            "apuntes.md",
            "text",
            [{"locator": "sec. 1", "content": "El servidor opera sobre HTTP local en 127.0.0.1."}],
        )
        test_db.update_source_summary(src_id, "### Resumen\nOpera en HTTP local.")

        # Serve stored source file
        file_resp = client.get(f"/api/sources/{src_id}/file")
        assert file_resp.status_code == 200
        assert b"HTTP local" in file_resp.content

        # Search persistent FTS5 index
        srch = client.get(f"/api/sessions/{sid}/search", params={"q": "HTTP local"})
        assert srch.status_code == 200
        assert len(srch.json()["results"]) == 1

        # Chat works with ONLY document sources (0 audios)
        chat_resp = client.post(
            f"/api/sessions/{sid}/chat",
            json={"question": "¿Sobre qué protocolo opera?"},
        )
        assert chat_resp.status_code == 200
        assert "HTTP local" in chat_resp.text

        # Quiz works with ONLY document sources (0 audios)
        quiz_resp = client.post(
            f"/api/sessions/{sid}/quiz/generate",
            json={"num_questions": 2, "mode": "mcq"},
        )
        assert quiz_resp.status_code == 200
        assert quiz_resp.json()["questions"][0]["timestamp"] == "apuntes.md · sec. 1"

        # Disable source -> chat rejects cleanly since 0 active sources remain
        toggled = client.patch(f"/api/sessions/{sid}/sources/{src_id}", json={"enabled": False})
        assert toggled.status_code == 200
        assert toggled.json()["sources"][0]["enabled"] is False

        chat_disabled = client.post(
            f"/api/sessions/{sid}/chat",
            json={"question": "¿Hola?"},
        )
        assert chat_disabled.status_code == 400

        # Re-enable source -> export includes Fuentes documentales e imágenes
        client.patch(f"/api/sessions/{sid}/sources/{src_id}", json={"enabled": True})
        exp = client.get(f"/api/sessions/{sid}/export")
        assert exp.status_code == 200
        assert "Fuentes documentales e imágenes" in exp.text
        assert "apuntes.md" in exp.text

        # Delete source cleans DB and file
        del_resp = client.delete(f"/api/sessions/{sid}/sources/{src_id}")
        assert del_resp.status_code == 200
        assert client.get(f"/api/sessions/{sid}").json()["sources"] == []

    test_db.close()


def test_podcast_script_parser_and_tts_sanitizer() -> None:
    from backend.podcast_service import (
        build_podcast_script_messages,
        parse_podcast_script_json,
        sanitize_text_for_tts,
    )

    cleaned = sanitize_text_for_tts(
        "### **Concepto clave** [04:12] revisar `fair test` en [tema.pdf · p. 3]!"
    )
    assert "[04:12]" not in cleaned
    assert "tema.pdf" not in cleaned
    assert "**" not in cleaned and "#" not in cleaned
    assert "Concepto clave" in cleaned
    assert "fair test" in cleaned

    raw_json = """```json
    [
      {"speaker": "A", "name": "Sofía", "text": "Hoy repasamos **la caché L1** [02:10]."},
      {"speaker": "B", "name": "Mateo", "text": "Exacto, aprovecha la localidad temporal [tema.pdf · p. 2]."}
    ]
    ```"""
    turns = parse_podcast_script_json(raw_json, language="es")
    assert len(turns) == 2
    assert turns[0]["speaker"] == "A" and turns[0]["name"] == "Sofía"
    assert "**" not in turns[0]["text"] and "[02:10]" not in turns[0]["text"]
    assert turns[1]["speaker"] == "B" and "[tema.pdf" not in turns[1]["text"]

    # Wrapper object JSON + line format + markdown bold + broken JSON + prose fallback
    wrapped_turns = parse_podcast_script_json(
        '{"dialogue": [{"host": "Sofía", "line": "Bienvenidos."}, {"host": "Mateo", "line": "Empezamos."}]}',
        language="es",
    )
    assert [t["speaker"] for t in wrapped_turns] == ["A", "B"]

    line_turns = parse_podcast_script_json(
        "SOFIA: Hola, hoy analizamos redes neuronales.\nMATEO: Y veremos cómo funcionan en la práctica.",
        language="es",
    )
    assert len(line_turns) == 2
    assert line_turns[0]["name"] == "Sofía" and line_turns[1]["name"] == "Mateo"

    md_turns = parse_podcast_script_json(
        "1. **Claire:** Welcome back to the study brief!\n2. **Ryan**: Today we connect lecture slides and notes.",
        language="en",
    )
    assert len(md_turns) == 2
    assert md_turns[0]["name"] == "Claire" and md_turns[1]["name"] == "Ryan"

    broken_json_turns = parse_podcast_script_json(
        '[{"speaker": "A", "text": "Primer punto clave."}, {"speaker": "B", "text": "Segundo punto sin cierre',
        language="es",
    )
    assert len(broken_json_turns) == 2

    prose_turns = parse_podcast_script_json(
        "Primero explicamos la memoria caché de primer nivel y su latencia.\n\nDespués comparamos cómo afecta al rendimiento real.",
        language="es",
    )
    assert [t["speaker"] for t in prose_turns] == ["A", "B"]

    msgs = build_podcast_script_messages(
        recordings=[],
        sources=[{"filename": "tema.pdf", "extracted_text": "La localidad temporal acelera la caché."}],
        focus_prompt="Enfóquense en la caché L1",
        style="exam_review",
        length="short",
        language="es",
    )
    assert "Enfóquense en la caché L1" in msgs[1]["content"]
    assert "localidad temporal" in msgs[1]["content"]
    assert "SOFIA:" in msgs[1]["content"] and "MATEO:" in msgs[1]["content"]


def test_podcast_db_crud_and_exclusive_409_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_db = Database(
        db_path=tmp_path / "pod.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
        podcasts_dir=tmp_path / "podcasts",
    )
    test_db.init_db()
    s = test_db.create_session("Curso Sistemas")
    src = test_db.create_source(s["id"], "guia.txt", "text", str(tmp_path / "guia.txt"))
    test_db.update_source_content(src["id"], "Contenido de prueba.", [], page_count=1)
    test_db.update_source_summary(src["id"], "Resumen breve.")

    pod_wav = tmp_path / "podcasts" / s["id"] / "ep1.wav"
    pod_wav.parent.mkdir(parents=True, exist_ok=True)
    pod_wav.write_bytes(b"RIFFdemo")

    pod = test_db.create_podcast(
        session_id=s["id"],
        title="Episodio Repaso",
        focus_prompt="Conceptos clave",
        style="conversation",
        length="short",
        language="es",
        source_refs=["guia.txt"],
        podcast_id="ep1",
    )
    assert pod["status"] == "queued"
    test_db.update_podcast_script("ep1", [{"speaker": "A", "name": "Sofía", "text": "Hola."}])
    test_db.update_podcast_ready("ep1", str(pod_wav), 12.5, "piper-onnx")

    detail = test_db.get_session_detail(s["id"])
    assert detail is not None and len(detail["podcasts"]) == 1
    assert detail["podcasts"][0]["status"] == "ready"
    assert detail["podcasts"][0]["tts_engine"] == "piper-onnx"

    monkeypatch.setattr(app_module, "db", test_db)
    monkeypatch.setattr("backend.resource_manager.db", test_db)
    monkeypatch.setattr(app_module.pipeline, "start", lambda: None)
    monkeypatch.setattr(app_module.resource_manager, "start_idle_watchdog", lambda: None)
    monkeypatch.setattr(app_module.ollama_service, "shutdown", AsyncMock())
    monkeypatch.setattr(app_module.resource_manager, "podcast_active", True)

    with TestClient(app_module.app) as client:
        # While podcast is active, other AI endpoints return 409 Conflict
        assert client.post(f"/api/sessions/{s['id']}/chat", json={"question": "Hola"}).status_code == 409
        assert client.post(f"/api/sessions/{s['id']}/quiz/generate", json={}).status_code == 409
        assert client.post(f"/api/sessions/{s['id']}/podcasts", json={}).status_code == 409

        monkeypatch.setattr(app_module.resource_manager, "podcast_active", False)
        audio_resp = client.get("/api/podcasts/ep1/audio")
        assert audio_resp.status_code == 200
        assert audio_resp.content == b"RIFFdemo"

        del_resp = client.delete(f"/api/sessions/{s['id']}/podcasts/ep1")
        assert del_resp.status_code == 200
        assert not pod_wav.exists()

        # Cancellation endpoint triggers cancel_event + task.cancel() cleanly
        import threading

        pod2 = test_db.create_podcast(
            session_id=s["id"],
            title="Cancelable Podcast",
            focus_prompt="",
            style="conversation",
            length="short",
            language="es",
            source_refs=["guia.txt"],
            podcast_id="ep_cancel",
        )
        ev = threading.Event()

        class DummyTask:
            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

        dummy_task = DummyTask()
        app_module._podcast_jobs[pod2["id"]] = (dummy_task, ev)  # type: ignore[assignment]
        cancel_resp = client.post(f"/api/sessions/{s['id']}/podcasts/{pod2['id']}/cancel")
        assert cancel_resp.status_code == 200
        assert cancel_resp.json()["cancelled"] is True
        assert ev.is_set() and dummy_task.cancelled
        app_module._podcast_jobs.pop(pod2["id"], None)

    test_db.close()


def test_kokoro_presets_and_podcast_cancellation_signal(tmp_path: Path) -> None:
    import threading
    from backend.config import KOKORO_VOICE_PRESETS
    from backend.podcast_service import PodcastCancelledError, synthesize_podcast_audio

    assert KOKORO_VOICE_PRESETS["es"]["A"]["voice"] == "ef_dora"
    assert KOKORO_VOICE_PRESETS["es"]["B"]["voice"] == "em_alex"
    assert KOKORO_VOICE_PRESETS["en"]["A"]["voice"] == "af_heart"

    ev = threading.Event()
    ev.set()
    with pytest.raises(PodcastCancelledError):
        synthesize_podcast_audio(
            turns=[{"speaker": "A", "name": "Sofía", "text": "Hola"}],
            output_path=tmp_path / "out.wav",
            language="es",
            models_dir=tmp_path / "tts",
            cancel_event=ev,
        )


def test_ollama_generate_complete_falls_back_to_thinking() -> None:
    import asyncio
    from backend.ollama_service import OllamaService

    svc = OllamaService()

    async def fake_stream_events(**_kwargs):
        yield ("thinking", "SOFIA: Hola.\nMATEO: Empezamos.")

    svc.stream_chat_events = fake_stream_events  # type: ignore[method-assign]
    out = asyncio.run(svc.generate_complete("demo-model", [{"role": "user", "content": "hi"}]))
    assert "SOFIA: Hola." in out
    assert "MATEO: Empezamos." in out


def test_eco_mode_skips_auto_summarization(tmp_path: Path, monkeypatch) -> None:
    import asyncio
    from unittest.mock import AsyncMock
    import backend.app as app_module

    test_db = Database(
        db_path=tmp_path / "eco.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
        podcasts_dir=tmp_path / "podcasts",
    )
    test_db.init_db()
    test_db.update_settings({"eco_mode": True})
    assert test_db.get_settings()["eco_mode"] is True

    monkeypatch.setattr(app_module, "db", test_db)
    summarize_mock = AsyncMock(return_value="should not run")
    monkeypatch.setattr(app_module.resource_manager, "summarize_exclusive", summarize_mock)

    sess = test_db.create_session("Sesión Eco")
    doc_path = tmp_path / "nota.md"
    doc_path.write_text("# Tema\nContenido de prueba para FTS5.", encoding="utf-8")
    src = test_db.create_source(
        session_id=sess["id"],
        filename="nota.md",
        source_type="text",
        stored_path=str(doc_path),
        mime_type="text/markdown",
        size_bytes=doc_path.stat().st_size,
    )

    asyncio.run(app_module.pipeline._process_source(src["id"]))

    updated = test_db.get_source(src["id"])
    assert updated is not None
    assert updated["status"] == "ready"
    assert updated["summary_text"] == ""
    assert summarize_mock.await_count == 0
    assert test_db.search_session_chunks(sess["id"], "FTS5")
    test_db.close()


def test_auto_summaries_off_keeps_indexing_and_manual_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_db = Database(
        db_path=tmp_path / "manual.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
        podcasts_dir=tmp_path / "podcasts",
    )
    test_db.init_db()
    test_db.update_settings({"ollama_model": "demo"})
    monkeypatch.setattr(app_module, "db", test_db)
    resolve_model = AsyncMock(return_value="demo")
    summarize = AsyncMock(return_value="Resumen solicitado")
    monkeypatch.setattr(app_module.ollama_service, "resolve_default_model", resolve_model)
    monkeypatch.setattr(app_module.resource_manager, "summarize_exclusive", summarize)

    sid = test_db.create_session("Resúmenes bajo demanda")["id"]
    audio = tmp_path / "clase.wav"
    audio.touch()
    rec = test_db.create_recording(sid, audio.name, str(audio), str(audio))
    monkeypatch.setattr(app_module, "prepare_audio_file", lambda *_args: (4.0, audio))
    transcript = "La grabación explica la fotosíntesis y cómo crecen las plantas."
    monkeypatch.setattr(app_module.resource_manager, "transcribe_exclusive", AsyncMock(
        return_value={
            "text": transcript,
            "segments": [{"start": 0.0, "end": 4.0, "text": transcript}],
            "language": "es",
            "duration_sec": 4.0,
        }
    ))
    doc = tmp_path / "apuntes.md"
    doc.write_text("El documento explica las raíces de las plantas.", encoding="utf-8")
    src = test_db.create_source(sid, doc.name, "text", str(doc))

    client = TestClient(app_module.app)
    try:
        assert test_db.get_settings()["auto_summarize"] is True
        response = client.put("/api/settings", json={"auto_summarize": False})
        assert response.status_code == 200
        assert response.json()["auto_summarize"] is False
        assert response.json()["eco_mode"] is False
        assert client.get("/api/settings").json()["auto_summarize"] is False

        asyncio.run(app_module.pipeline._process_recording(rec["id"]))
        asyncio.run(app_module.pipeline._process_source(src["id"]))
        for item in (test_db.get_recording(rec["id"]), test_db.get_source(src["id"])):
            assert item["status"] == "ready"
            assert item["summary_text"] == ""
        assert test_db.search_session_chunks(sid, "fotosíntesis")
        assert test_db.search_session_chunks(sid, "raíces")
        assert client.patch(
            f"/api/sessions/{sid}/sources/{src['id']}", json={"enabled": True}
        ).json()["combined_summary"] == ""
        assert not app_module._combined_summary_tasks
        assert client.post(f"/api/sessions/{sid}/suggestions").json()["suggested_prompts"] == []
        resolve_model.assert_not_awaited()
        summarize.assert_not_awaited()

        manual = client.post(f"/api/sessions/{sid}/resummarize").json()
        assert manual["combined_summary"] == "Resumen solicitado"
        assert summarize.await_count == 3
    finally:
        client.close()
        test_db.close()


def test_desktop_default_settings_save_and_no_eco_throttle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify desktop defaults keep eco_mode off, PUT /api/settings succeeds, and Ollama runs unthrottled."""
    import asyncio
    from backend.ollama_service import OllamaService

    test_db = Database(
        db_path=tmp_path / "desktop.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
        podcasts_dir=tmp_path / "podcasts",
    )
    test_db.init_db()
    defaults = test_db.get_settings()
    assert defaults["eco_mode"] is False
    if MLX_ASR_SUPPORTED:
        assert defaults["whisper_model"] == NEMOTRON_MODEL_ID
        assert defaults["whisper_language"] == "es"
    assert test_db.update_settings({"whisper_model": NEMOTRON_MODEL_ID})["whisper_language"] == "es"
    assert test_db.update_settings({"whisper_language": "en"})["whisper_language"] == "en"
    assert test_db.update_settings({"whisper_model": "large-v3-turbo", "whisper_language": "auto"})["whisper_language"] == "auto"

    monkeypatch.setattr(app_module, "db", test_db)
    monkeypatch.setattr("backend.resource_manager.db", test_db)
    monkeypatch.setattr(app_module.pipeline, "start", lambda: None)
    monkeypatch.setattr(app_module.resource_manager, "start_idle_watchdog", lambda: None)
    monkeypatch.setattr(app_module.ollama_service, "shutdown", AsyncMock())

    with TestClient(app_module.app) as client:
        resp = client.put(
            "/api/settings",
            json={
                "ollama_model": "Gemma-4-E4B-it:latest",
                "whisper_model": "large-v3-turbo",
                "whisper_device": "auto",
                "whisper_language": "es",
                "num_ctx": 8192,
                "idle_unload_sec": 180,
                "classroom_audio_enhance": True,
                "enable_thinking": False,
                "eco_mode": False,
                "eco_cpu_threads": 2,
            },
        )
        assert resp.status_code == 200
        saved = resp.json()
        assert saved["eco_mode"] is False
        assert saved["num_ctx"] == 8192

    # Confirm OllamaService.stream_chat_events does not inject num_thread/num_predict when eco_mode=False
    captured_payloads: list[dict] = []

    class FakeStreamResp:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def aiter_lines(self):
            yield '{"message": {"content": "ok"}}'

    class FakeAsyncClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        def stream(self, _method, _url, json=None):
            captured_payloads.append(json or {})
            return FakeStreamResp()

    svc = OllamaService()
    monkeypatch.setattr(svc, "ensure_running", AsyncMock())
    monkeypatch.setattr("backend.ollama_service.httpx.AsyncClient", FakeAsyncClient)

    async def _collect():
        return [evt async for evt in svc.stream_chat_events("demo", [{"role": "user", "content": "hi"}], eco_mode=False)]

    evts = asyncio.run(_collect())
    assert evts == [("token", "ok")]
    opts = captured_payloads[0]["options"]
    assert opts == {"num_ctx": 8192, "temperature": 0.3}
    assert "num_thread" not in opts
    assert "num_predict" not in opts

    messages = [{"role": "user", "content": "hi"}]
    monkeypatch.setattr("backend.resource_manager.ollama_service", svc)
    manager = ResourceManager()
    assert asyncio.run(manager.summarize_exclusive(messages, "demo", 8192, 180, num_predict=512)) == "ok"
    assert captured_payloads[1]["options"] == {**opts, "num_predict": 512}
    test_db.update_settings({"eco_mode": True})
    asyncio.run(manager.summarize_exclusive(messages, "demo", 8192, 180, num_predict=1024))
    assert captured_payloads[2]["options"]["num_predict"] == 768
    test_db.close()


def test_filename_matching_and_source_catalog_in_chat(tmp_path: Path) -> None:
    test_db = Database(
        db_path=tmp_path / "catalog.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
        podcasts_dir=tmp_path / "podcasts",
    )
    test_db.init_db()
    sess = test_db.create_session("Comunicación")
    sid = sess["id"]

    files_and_bodies = [
        ("Chapter_10_Intro_Messages.pdf", "Principios generales de redacción rutinaria."),
        (
            "Chapter_11_Persuasive_Messages_Summary.pdf",
            "El modelo AIDA organiza la persuasión en Atención, Interés, Deseo y Acción.",
        ),
        ("Chapter_12_Reports.pdf", "Estructuras formales para informes ejecutivos."),
    ]
    created_srcs = []
    for idx, (fname, body) in enumerate(files_and_bodies):
        s = test_db.create_source(sid, fname, "pdf", str(tmp_path / fname))
        pages = [{"locator": "p. 1", "label": "Página 1", "text": body}]
        test_db.update_source_content(s["id"], body, pages, page_count=1)
        test_db.replace_item_chunks(sid, "source", s["id"], fname, "pdf", [{"locator": "p. 1", "content": body}])
        test_db.update_source_summary(s["id"], f"Resumen de archivo {idx + 10}.")
        created_srcs.append(test_db.get_source(s["id"]))

    # "chapter 11" must match Chapter_11_Persuasive_Messages_Summary.pdf even though "chapter 11" isn't in body text
    hits_en = test_db.search_session_chunks(
        sid,
        "de que trata el chapter 11?",
        limit=6,
        enabled_only=True,
        primary_query="de que trata el chapter 11?",
    )
    assert hits_en
    assert hits_en[0]["filename"] == "Chapter_11_Persuasive_Messages_Summary.pdf"
    assert "AIDA" in hits_en[0]["content"]

    # Spanish translated reference ("capítulo 11") must also match Chapter_11_Persuasive_Messages_Summary.pdf
    hits_es = test_db.search_session_chunks(
        sid,
        "¿qué explica el capítulo 11?",
        limit=6,
        enabled_only=True,
        primary_query="¿qué explica el capítulo 11?",
    )
    assert hits_es
    assert hits_es[0]["filename"] == "Chapter_11_Persuasive_Messages_Summary.pdf"

    # Source catalog must appear verbatim in chat system prompt even when total content is large (retrieval mode)
    big_srcs = []
    for s in created_srcs:
        assert s is not None
        copy = dict(s)
        copy["extracted_text"] = ("Contenido extenso de prueba. " * 220)
        copy["pages"] = [{"locator": "p. 1", "text": copy["extracted_text"]}]
        big_srcs.append(copy)

    msgs, mode = build_chat_messages(
        recordings=[],
        combined_summary="Resumen combinado.",
        chat_history=[],
        user_question="¿de qué trata el chapter 11?",
        num_ctx=4096,
        sources=big_srcs,
        indexed_chunks=hits_en,
    )
    assert mode == "retrieval"
    assert "CATÁLOGO COMPLETO DE FUENTES ACTIVAS" in msgs[0]["content"]
    assert "Chapter_11_Persuasive_Messages_Summary.pdf" in msgs[0]["content"]
    assert "Chapter 11 Persuasive Messages Summary" in msgs[0]["content"]
    assert "AIDA" in msgs[0]["content"]
    test_db.close()


def test_empty_scanned_pdf_rejected_without_junk_chunks_and_legacy_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    from pypdf import PdfWriter
    import backend.app as app_module
    from backend.source_service import chunk_pages, pages_to_full_text

    # 1. Placeholder strings never produce chunks or full text
    fake_pages = [
        {"locator": "p. 1", "text": "[Página sin texto seleccionable o escaneada]"},
        {"locator": "p. 2", "text": "   "},
    ]
    assert chunk_pages(fake_pages) == []
    assert pages_to_full_text(fake_pages) == ""

    # 2. Blank scanned PDF submitted to _process_source ends in error with 0 chunks & 0 LLM calls
    test_db = Database(
        db_path=tmp_path / "junk.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
        podcasts_dir=tmp_path / "podcasts",
    )
    test_db.init_db()
    test_db.update_settings({"eco_mode": False})
    monkeypatch.setattr(app_module, "db", test_db)
    summarize_mock = AsyncMock(return_value="should never run")
    monkeypatch.setattr(app_module.resource_manager, "summarize_exclusive", summarize_mock)
    monkeypatch.setattr(app_module.ollama_service, "resolve_default_model", AsyncMock(return_value="demo"))

    sess = test_db.create_session("PDF Escaneado")
    pdf_path = tmp_path / "escaneado.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with pdf_path.open("wb") as fh:
        writer.write(fh)

    src = test_db.create_source(sess["id"], "escaneado.pdf", "pdf", str(pdf_path))
    asyncio.run(app_module.pipeline._process_source(src["id"]))

    updated = test_db.get_source(src["id"])
    assert updated is not None
    assert updated["status"] == "error"
    assert test_db.count_session_chunks(sess["id"], enabled_only=False) == 0
    assert summarize_mock.await_count == 0

    # 3. Legacy DB cleanup on init_db purges pre-existing placeholder chunks & placeholder-only sources
    legacy = test_db.create_source(sess["id"], "viejo_escaneo.pdf", "pdf", str(pdf_path))
    legacy_text = "[p. 1]\n[Página sin texto seleccionable o escaneada]\n\n[p. 2]\n[Página sin texto seleccionable o escaneada]"
    test_db.update_source_content(legacy["id"], legacy_text, fake_pages, page_count=2)
    test_db.update_source_summary(legacy["id"], "Resumen inventado")
    test_db.replace_item_chunks(
        sess["id"],
        "source",
        legacy["id"],
        "viejo_escaneo.pdf",
        "pdf",
        [{"locator": "p. 1", "content": "[Página sin texto seleccionable o escaneada]"}],
    )
    assert test_db.count_session_chunks(sess["id"], enabled_only=True) == 1

    test_db.init_db()
    cleaned_legacy = test_db.get_source(legacy["id"])
    assert cleaned_legacy is not None
    assert cleaned_legacy["status"] == "error"
    assert cleaned_legacy["extracted_text"] == ""
    assert test_db.count_session_chunks(sess["id"], enabled_only=False) == 0
    test_db.close()


def test_delete_source_returns_immediately_without_blocking_on_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import time

    test_db = Database(
        db_path=tmp_path / "fastdel.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
        podcasts_dir=tmp_path / "podcasts",
    )
    test_db.init_db()
    test_db.update_settings({"eco_mode": False})
    monkeypatch.setattr(app_module, "db", test_db)
    monkeypatch.setattr("backend.resource_manager.db", test_db)
    monkeypatch.setattr(app_module.pipeline, "start", lambda: None)
    monkeypatch.setattr(app_module.resource_manager, "start_idle_watchdog", lambda: None)
    monkeypatch.setattr(app_module.ollama_service, "shutdown", AsyncMock())

    async def slow_summarize(*_args, **_kwargs):
        await asyncio.sleep(5.0)
        return "delayed combined summary"

    monkeypatch.setattr(app_module.resource_manager, "summarize_exclusive", slow_summarize)
    monkeypatch.setattr(app_module.ollama_service, "resolve_default_model", AsyncMock(return_value="demo"))

    sess = test_db.create_session("Sesión Rápida")
    sid = sess["id"]
    src_ids = []
    for name in ("a.txt", "b.txt", "c.txt"):
        s = test_db.create_source(sid, name, "text", str(tmp_path / name))
        test_db.update_source_content(s["id"], f"Contenido {name}", [{"locator": "sec. 1", "text": f"Contenido {name}"}], 1)
        test_db.update_source_summary(s["id"], f"Resumen {name}")
        test_db.replace_item_chunks(sid, "source", s["id"], name, "text", [{"locator": "sec. 1", "content": f"Contenido {name}"}])
        src_ids.append(s["id"])

    with TestClient(app_module.app) as client:
        t0 = time.monotonic()
        resp = client.delete(f"/api/sessions/{sid}/sources/{src_ids[0]}")
        elapsed = time.monotonic() - t0
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        assert elapsed < 1.5
        assert len(client.get(f"/api/sessions/{sid}").json()["sources"]) == 2

    assert test_db.count_session_chunks(sid, enabled_only=False) == 2
    test_db.close()


def test_chat_prompt_requires_filename_in_multi_audio_citations() -> None:
    recs = [
        {
            "id": "r1",
            "filename": "Voz 028 W 20261124 220426.m4a",
            "status": "ready",
            "enabled": True,
            "duration_sec": 300.0,
            "transcript_text": "Explicación del tema A.",
            "segments": [{"start": 158.0, "end": 165.0, "text": "Explicación del tema A."}],
        },
        {
            "id": "r2",
            "filename": "Voz 029 W 20261124 220510.m4a",
            "status": "ready",
            "enabled": True,
            "duration_sec": 240.0,
            "transcript_text": "Explicación del tema B.",
            "segments": [{"start": 45.0, "end": 52.0, "text": "Explicación del tema B."}],
        },
    ]
    msgs, _mode = build_chat_messages(
        recordings=recs,
        combined_summary="",
        chat_history=[],
        user_question="¿Qué dice la grabación 1?",
        num_ctx=8192,
    )
    assert "[nombre_audio · MM:SS]" in msgs[0]["content"]
    assert "Voz 028 W 20261124 220426.m4a" in msgs[0]["content"]


def test_true_false_prompt_blueprint_and_inverted_options_parsing() -> None:
    from backend.context_manager import build_quiz_generation_messages, shuffle_quiz_mcq_options

    msgs = build_quiz_generation_messages(
        recordings=[
            {
                "id": "r1",
                "filename": "clase.m4a",
                "status": "ready",
                "transcript_text": "La fotosíntesis ocurre en los cloroplastos.",
                "segments": [{"start": 10.0, "end": 15.0, "text": "La fotosíntesis ocurre en los cloroplastos."}],
            }
        ],
        num_questions=6,
        mode="true_false",
    )
    prompt_txt = msgs[1]["content"]
    assert "Verdadero (correct_index: 0)" in prompt_txt
    assert "Falso (correct_index: 1)" in prompt_txt
    assert "NUNCA hagas que todas sean Falsas ni todas Verdaderas" in prompt_txt

    # Inverted ["Falso", "Verdadero"] options with correct_index=1 must resolve to canonical index 0 (Verdadero)
    inverted = parse_quiz_json(
        '[{"type": "true_false", "question": "La fotosíntesis ocurre en los cloroplastos.", "options": ["Falso", "Verdadero"], "correct_index": 1}]',
        mode="true_false",
    )
    assert len(inverted) == 1
    assert inverted[0]["options"] == ["Verdadero", "Falso"]
    assert inverted[0]["correct_index"] == 0

    # MCQ shuffle keeps positional options intact and updates correct_index cleanly
    mcq = [
        {
            "id": 1,
            "type": "mcq",
            "question": "¿Capital?",
            "options": ["Madrid", "París", "Roma", "Lisboa"],
            "correct_index": 0,
        },
        {
            "id": 2,
            "type": "mcq",
            "question": "¿Cuál aplica?",
            "options": ["A", "B", "C", "Todas las anteriores"],
            "correct_index": 3,
        },
    ]
    shuffled = shuffle_quiz_mcq_options(mcq)
    assert shuffled[0]["options"][shuffled[0]["correct_index"]] == "Madrid"
    assert shuffled[1]["options"] == ["A", "B", "C", "Todas las anteriores"]
    assert shuffled[1]["correct_index"] == 3


def test_quiz_generate_auto_rebalances_all_false_true_false_exam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_db = Database(db_path=tmp_path / "tf.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    monkeypatch.setattr(app_module, "db", test_db)
    monkeypatch.setattr("backend.resource_manager.db", test_db)
    monkeypatch.setattr(app_module.pipeline, "start", lambda: None)
    monkeypatch.setattr(app_module.resource_manager, "start_idle_watchdog", lambda: None)
    monkeypatch.setattr(app_module.ollama_service, "shutdown", AsyncMock())
    monkeypatch.setattr(app_module.ollama_service, "resolve_default_model", AsyncMock(return_value="demo"))

    sess = test_db.create_session("Biología")
    rec = test_db.create_recording(sess["id"], "bio.m4a", "bio.m4a", "bio.m4a")
    test_db.update_recording_transcript(
        rec["id"],
        "Las mitocondrias producen ATP y tienen su propio ADN.",
        [{"start": 5.0, "end": 12.0, "text": "Las mitocondrias producen ATP y tienen su propio ADN."}],
        "es",
        60.0,
        "small",
    )
    test_db.update_recording_summary(rec["id"], "Resumen mitocondrias.")

    all_false_raw = (
        '['
        '{"id": 1, "type": "true_false", "question": "Las mitocondrias no producen ATP.", "options": ["Verdadero", "Falso"], "correct_index": 1, "explanation": "Sí producen."},'
        '{"id": 2, "type": "true_false", "question": "Las mitocondrias carecen de ADN.", "options": ["Verdadero", "Falso"], "correct_index": 1, "explanation": "Sí tienen."},'
        '{"id": 3, "type": "true_false", "question": "El ATP se produce solo en el núcleo.", "options": ["Verdadero", "Falso"], "correct_index": 1, "explanation": "En mitocondrias."},'
        '{"id": 4, "type": "true_false", "question": "Las mitocondrias no están en eucariotas.", "options": ["Verdadero", "Falso"], "correct_index": 1, "explanation": "Sí están."}'
        ']'
    )
    balanced_raw = (
        '['
        '{"id": 1, "type": "true_false", "question": "Las mitocondrias producen ATP.", "options": ["Verdadero", "Falso"], "correct_index": 0, "explanation": "Correcto."},'
        '{"id": 2, "type": "true_false", "question": "Las mitocondrias tienen su propio ADN.", "options": ["Verdadero", "Falso"], "correct_index": 0, "explanation": "Correcto."},'
        '{"id": 3, "type": "true_false", "question": "El ATP se produce solo en el núcleo.", "options": ["Verdadero", "Falso"], "correct_index": 1, "explanation": "En mitocondrias."},'
        '{"id": 4, "type": "true_false", "question": "Las mitocondrias no están en eucariotas.", "options": ["Verdadero", "Falso"], "correct_index": 1, "explanation": "Sí están."}'
        ']'
    )
    call_count = 0

    async def fake_summarize(messages, **_kwargs):
        nonlocal call_count
        call_count += 1
        return all_false_raw if call_count == 1 else balanced_raw

    monkeypatch.setattr(app_module.resource_manager, "summarize_exclusive", fake_summarize)

    with TestClient(app_module.app) as client:
        resp = client.post(
            f"/api/sessions/{sess['id']}/quiz/generate",
            json={"num_questions": 4, "mode": "true_false", "difficulty": "hard", "question_style": "tricky"},
        )
        assert resp.status_code == 200
        qs = resp.json()["questions"]
        assert call_count == 2
        assert sum(1 for q in qs if q["correct_index"] == 0) == 2
        assert sum(1 for q in qs if q["correct_index"] == 1) == 2

    test_db.close()


def test_whisper_transcribe_cancel_event_stops_immediately(tmp_path: Path) -> None:
    import threading
    from types import SimpleNamespace

    from backend.whisper_service import WhisperCancelledError, WhisperService

    audio_file = tmp_path / "clase.wav"
    audio_file.write_bytes(b"RIFFdemo")

    cancel_ev = threading.Event()
    called_segments: list[str] = []

    class FakeWhisperModel:
        def transcribe(self, *_args, **_kwargs):
            def _gen():
                yield SimpleNamespace(start=0.0, end=1.0, text="Primer segmento")
                cancel_ev.set()
                yield SimpleNamespace(start=1.0, end=2.0, text="Segundo segmento")

            return _gen(), SimpleNamespace(duration=4.0, language="es")

    svc = WhisperService()
    svc._model = FakeWhisperModel()

    def _on_progress(_ratio: float, _lang: str) -> None:
        called_segments.append("progress")

    with pytest.raises(WhisperCancelledError):
        svc.transcribe(
            audio_file,
            language="es",
            progress_callback=_on_progress,
            classroom_enhance=False,
            cancel_event=cancel_ev,
        )
    assert len(called_segments) == 1


def test_ollama_abort_active_closes_streams() -> None:
    from backend.ollama_service import OllamaService

    class FakeClient:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    svc = OllamaService()
    c1 = FakeClient()
    c2 = FakeClient()
    svc._active_clients.add(c1)  # type: ignore[arg-type]
    svc._active_clients.add(c2)  # type: ignore[arg-type]

    closed = asyncio.run(svc.abort_active())
    assert closed == 2
    assert c1.closed and c2.closed
    assert c1 in svc._aborted_clients and c2 in svc._aborted_clients


def test_system_stop_endpoint_cancels_active_jobs_and_resets_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_db = Database(
        db_path=tmp_path / "stop.db",
        audio_dir=tmp_path / "audio",
        sources_dir=tmp_path / "sources",
        podcasts_dir=tmp_path / "podcasts",
    )
    test_db.init_db()
    monkeypatch.setattr(app_module, "db", test_db)
    monkeypatch.setattr("backend.resource_manager.db", test_db)
    monkeypatch.setattr(app_module.pipeline, "start", lambda: None)
    monkeypatch.setattr(app_module.resource_manager, "start_idle_watchdog", lambda: None)
    monkeypatch.setattr(app_module.ollama_service, "shutdown", AsyncMock())
    monkeypatch.setattr(app_module.ollama_service, "abort_active", AsyncMock(return_value=1))

    async def _sleep_forever():
        await asyncio.sleep(30)

    with TestClient(app_module.app) as client:
        # Simulate an active background task inside resource_manager & _combined_summary_tasks
        async def _trigger_and_stop():
            dummy_task = asyncio.create_task(_sleep_forever())
            combined_task = asyncio.create_task(_sleep_forever())
            app_module.resource_manager._active_tasks.add(dummy_task)
            app_module._combined_summary_tasks["sess1"] = combined_task
            await asyncio.sleep(0)
            res = await app_module.system_stop()
            assert res["stopped"] is True
            assert res["aborted_streams"] == 1
            assert res["cancelled_tasks"] >= 1
            assert res["cancelled_combined"] >= 1
            assert dummy_task.cancelled()
            assert combined_task.cancelled()
            assert res["resources"]["phase"] == "idle"
            app_module.resource_manager._active_tasks.discard(dummy_task)
            app_module._combined_summary_tasks.pop("sess1", None)

        asyncio.run(_trigger_and_stop())

        # Also exercise the HTTP route directly via TestClient
        resp = client.post("/api/system/stop")
        assert resp.status_code == 200
        body = resp.json()
        assert body["stopped"] is True
        assert body["resources"]["phase"] == "idle"

    test_db.close()


def test_windows_asr_presets_discovery_and_canary_language_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys
    from backend import config as cfg

    if sys.platform == "darwin":
        pytest.skip("Windows/Linux ONNX & Torch preset check")

    models = cfg.list_available_whisper_models(models_dir=tmp_path)
    ids = {m["id"]: m for m in models}
    if cfg.PARAKEET_ONNX_SUPPORTED:
        assert cfg.PARAKEET_MODEL_ID in ids
        assert ids[cfg.PARAKEET_MODEL_ID]["engine"] == "Parakeet"
    if cfg.CANARY_ONNX_SUPPORTED:
        assert cfg.CANARY_MODEL_ID in ids
        assert ids[cfg.CANARY_MODEL_ID].get("requires_language") is True
    if cfg.QWEN_TORCH_SUPPORTED:
        assert "qwen3-asr-0.6b" in ids
        assert "qwen3-asr-1.7b" in ids

    test_db = Database(db_path=tmp_path / "canary.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    test_db.update_settings({"whisper_model": cfg.CANARY_MODEL_ID, "whisper_language": "auto"})
    assert test_db.get_settings()["whisper_language"] == "es"
    test_db.close()


def test_qwen_chunk_builder_splits_long_audio_cleanly() -> None:
    import numpy as np
    from backend.whisper_service import WhisperService

    sr = 16000
    # 60s of speech with a quiet gap around 24s and 48s
    audio = np.full(60 * sr, 0.25, dtype=np.float32)
    audio[int(23.5 * sr) : int(24.5 * sr)] = 0.0
    audio[int(47.5 * sr) : int(48.5 * sr)] = 0.0

    chunks = WhisperService._build_qwen_chunks(audio, sr)
    assert len(chunks) == 3
    assert chunks[0][0] == 0.0
    assert 23.4 <= chunks[0][1] <= 24.6
    assert 47.4 <= chunks[1][1] <= 48.6
    assert chunks[-1][1] == pytest.approx(60.0, abs=0.05)


def test_audio_capture_mix_and_resample() -> None:
    import numpy as np
    from backend.audio_capture import mix_pcm_frames, resample_mono

    # Stereo 48kHz PCM (left=1000, right=3000 -> mono=2000) resampled to 44.1kHz
    frames_48k = np.tile(np.array([1000, 3000], dtype=np.int16), 480)
    mono_44k = resample_mono(frames_48k.tobytes(), src_rate=48000, channels=2, dst_rate=44100)
    assert mono_44k.dtype == np.int16
    assert len(mono_44k) == 441
    assert int(np.mean(mono_44k)) == pytest.approx(2000, abs=5)

    # Mixing with clipping protection & asymmetric length padding
    a = np.array([20000, -25000, 1000], dtype=np.int16)
    b = np.array([20000, -20000], dtype=np.int16)
    mixed = mix_pcm_frames(a, b)
    assert mixed.tolist() == [32767, -32768, 1000]


def test_system_capture_endpoints_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wave
    from fastapi.testclient import TestClient
    import backend.app as app_mod

    test_db = Database(db_path=tmp_path / "syscap.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    test_audio_dir = tmp_path / "audio"
    test_audio_dir.mkdir(parents=True, exist_ok=True)
    sess = test_db.create_session("Nueva conversación")
    enqueued: list[str] = []

    class FakeRecorder:
        def __init__(self) -> None:
            self.active = False
            self.session_id: str | None = None
            self.rec_id: str | None = None
            self.wav_path: Path | None = None
            self.mode = "system"

        def get_status(self):
            return {
                "recording": self.active,
                "available": True,
                "mode": self.mode if self.active else None,
                "session_id": self.session_id if self.active else None,
                "rec_id": self.rec_id if self.active else None,
                "elapsed_sec": 1.25 if self.active else 0.0,
                "level": 0.42 if self.active else 0.0,
                "device_name": "Speakers (WASAPI)" if self.active else "",
            }

        def start(self, session_id: str, rec_id: str, output_path: Path, mode: str = "system"):
            self.active = True
            self.session_id = session_id
            self.rec_id = rec_id
            self.wav_path = output_path
            self.mode = mode
            return self.get_status()

        def stop(self):
            if not self.active or not self.wav_path:
                return {"recording": False, "session_id": None, "rec_id": None, "wav_path": None, "duration_sec": 0.0}
            self.wav_path.parent.mkdir(parents=True, exist_ok=True)
            with wave.open(str(self.wav_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(44100)
                wf.writeframes(b"\x01\x00" * 4410)
            out = {
                "recording": False,
                "session_id": self.session_id,
                "rec_id": self.rec_id,
                "mode": self.mode,
                "wav_path": self.wav_path,
                "duration_sec": 0.1,
            }
            self.active = False
            return out

    fake = FakeRecorder()
    monkeypatch.setattr(app_mod, "db", test_db)
    monkeypatch.setattr(app_mod, "AUDIO_DIR", test_audio_dir)
    monkeypatch.setattr(app_mod, "system_audio_recorder", fake)
    monkeypatch.setattr(app_mod, "is_system_audio_available", lambda: True)
    monkeypatch.setattr(app_mod.pipeline, "enqueue", lambda rid: enqueued.append(rid))

    client = TestClient(app_mod.app)

    res_status_sys = client.get("/api/system/status")
    assert res_status_sys.status_code == 200
    assert res_status_sys.json()["system_audio_available"] is True

    res_bad = client.post(f"/api/sessions/{sess['id']}/recordings/system-capture/start", json={"mode": "invalid"})
    assert res_bad.status_code == 400

    res_start = client.post(f"/api/sessions/{sess['id']}/recordings/system-capture/start", json={"mode": "both"})
    assert res_start.status_code == 200
    assert res_start.json()["recording"] is True
    assert res_start.json()["mode"] == "both"

    res_poll = client.get("/api/recordings/system-capture/status")
    assert res_poll.status_code == 200
    assert res_poll.json()["level"] == pytest.approx(0.42)

    res_stop = client.post("/api/recordings/system-capture/stop")
    assert res_stop.status_code == 200
    payload = res_stop.json()
    assert payload["recording"]["id"] in enqueued
    assert payload["recording"]["filename"].startswith("Equipo + Mic ")
    assert test_db.get_session(sess["id"])["title"].startswith("Equipo + Mic ")
    test_db.close()


def test_live_preview_omits_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(app_module.db, "get_settings", lambda: {})
    transcribe = AsyncMock(return_value={
        "text": "[No se detectó voz comprensible en la grabación]",
        "segments": [],
    })
    monkeypatch.setattr(app_module.resource_manager, "transcribe_exclusive", transcribe)
    response = TestClient(app_module.app).post(
        "/api/recordings/live-transcript", files={"file": ("silencio.webm", b"audio", "audio/webm")},
    )
    assert response.status_code == 200
    assert response.json() == {"text": ""}
    transcribe.assert_awaited_once()


@pytest.mark.parametrize("cancel", [False, True])
def test_recording_text_available_during_speaker_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel: bool,
) -> None:
    test_db = Database(db_path=tmp_path / "test.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    test_db.update_settings({"auto_summarize": False, "enable_diarization": True})
    session = test_db.create_session("Grabación")
    audio = tmp_path / "grabacion.wav"
    audio.touch()
    rec = test_db.create_recording(
        session_id=session["id"], filename=audio.name, stored_path=str(audio), playable_path=str(audio),
    )
    rec_id = rec["id"]
    text = "La transcripción está lista."
    segments = [{"start": 0.0, "end": 4.0, "text": text}]
    monkeypatch.setattr(app_module, "db", test_db)
    monkeypatch.setattr(app_module, "prepare_audio_file", lambda *_: (4.0, audio))
    monkeypatch.setattr(app_module, "is_diarization_runtime_available", lambda: True)
    monkeypatch.setattr(app_module.resource_manager, "transcribe_exclusive", AsyncMock(return_value={
        "text": text, "segments": segments, "language": "es", "duration_sec": 4.0,
    }))

    snapshots = []

    async def identify(*_):
        snapshots.append(test_db.get_recording(rec_id))
        if cancel:
            raise asyncio.CancelledError
        return [{"start": 0.0, "end": 4.0, "speaker": "Hablante 1"}]

    identify_speakers = AsyncMock(side_effect=identify)
    monkeypatch.setattr(app_module, "diarize_audio_async", identify_speakers)
    try:
        asyncio.run(app_module.ProcessingPipeline()._process_recording(rec_id))
        identify_speakers.assert_awaited_once()
        assert snapshots[0]["status"] == "transcribing"
        assert snapshots[0]["transcript_text"] == text
        assert snapshots[0]["segments"] == segments
        saved = test_db.get_recording(rec_id)
        assert saved["status"] == "ready"
        assert saved["transcript_text"] == (text if cancel else f"Hablante 1: {text}")
        assert test_db.search_session_chunks(session["id"], "transcripción", limit=1)
    finally:
        test_db.close()


def test_diarization_assign_merge_bleed_and_format() -> None:
    from backend.context_manager import format_transcript_for_llm
    from backend.diarization_service import (
        assign_speakers_to_segments,
        dedupe_call_bleed,
        format_segments_plain_text,
        merge_call_segments,
    )
    from backend.source_service import chunk_recording_for_index

    # Word-level split when a single ASR block spans a clean speaker switch (>= 3 words, >= 1.0s each)
    asr_segs = [
        {
            "start": 0.0,
            "end": 5.0,
            "text": "Hola buenos días amigos Perfecto muchas gracias por venir",
            "words": [
                {"start": 0.0, "end": 0.4},
                {"start": 0.5, "end": 0.9},
                {"start": 1.0, "end": 1.4},
                {"start": 1.5, "end": 1.9},
                {"start": 2.0, "end": 2.4},
                {"start": 2.6, "end": 3.1},
                {"start": 3.2, "end": 3.7},
                {"start": 3.8, "end": 4.3},
                {"start": 4.4, "end": 4.9},
            ],
        },
        {"start": 5.5, "end": 8.0, "text": "Revisemos la agenda completa."},
    ]
    turns = [
        {"start": 0.0, "end": 2.5, "speaker": "Hablante 2", "speaker_key": "spk_2"},
        {"start": 2.5, "end": 8.2, "speaker": "Hablante 5", "speaker_key": "spk_5"},
    ]
    assigned = assign_speakers_to_segments(asr_segs, turns)
    assert len(assigned) == 3
    # First-appearance renormalization produces Hablante 1, Hablante 2 (no gaps)
    assert assigned[0]["speaker"] == "Hablante 1"
    assert assigned[1]["speaker"] == "Hablante 2"
    assert assigned[2]["speaker"] == "Hablante 2"

    # Acoustic bleed suppression when mic picks up speaker output simultaneously
    mic_segs = [
        {"start": 2.6, "end": 4.9, "text": "Perfecto muchas gracias por venir."},
        {"start": 8.5, "end": 11.0, "text": "Sí, aquí tengo los datos listos."},
    ]
    clean_mic = dedupe_call_bleed(mic_segs, assigned)
    assert len(clean_mic) == 1
    assert "tengo los datos" in clean_mic[0]["text"]

    merged = merge_call_segments(mic_segs, assigned)
    assert merged[-1]["speaker"] == "Tú"
    assert merged[0]["speaker"] == "Hablante 1"

    plain = format_segments_plain_text(merged)
    assert "Hablante 1:" in plain
    assert "Hablante 2:" in plain
    assert "Tú:" in plain

    rec_mock = {"filename": "Llamada.wav", "segments": merged, "transcript_text": plain}
    llm_fmt = format_transcript_for_llm(rec_mock)
    assert "[00:00] Hablante 1:" in llm_fmt
    assert "[00:08] Tú:" in llm_fmt or "[00:09] Tú:" in llm_fmt
    chunks = chunk_recording_for_index(rec_mock)
    assert any("Tú:" in c["content"] for c in chunks)


def test_dual_track_split_and_speaker_rename_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import wave
    import numpy as np
    from fastapi.testclient import TestClient
    import backend.app as app_mod
    from backend.audio_capture import split_dual_track_wav

    tracks_wav = tmp_path / "call.tracks.wav"
    sr = 16000
    ch0_mic = np.full(sr // 2, 1200, dtype=np.int16)
    ch1_sys = np.full(sr // 2, -2400, dtype=np.int16)
    stereo = np.column_stack([ch0_mic, ch1_sys])
    with wave.open(str(tracks_wav), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(stereo.tobytes())

    mic_path, sys_path = split_dual_track_wav(tracks_wav, tmp_path / "split")
    with wave.open(str(mic_path), "rb") as wf_m, wave.open(str(sys_path), "rb") as wf_s:
        m_arr = np.frombuffer(wf_m.readframes(wf_m.getnframes()), dtype=np.int16)
        s_arr = np.frombuffer(wf_s.readframes(wf_s.getnframes()), dtype=np.int16)
    assert int(np.mean(m_arr)) == 1200
    assert int(np.mean(s_arr)) == -2400

    # Test speaker rename endpoint updates segments, transcript_text & FTS5 chunks
    test_db = Database(db_path=tmp_path / "rename.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    sess = test_db.create_session("Reunión de prueba")
    rec = test_db.create_recording(
        session_id=sess["id"],
        filename="Equipo + Mic.wav",
        stored_path=str(tmp_path / "x.wav"),
        playable_path=str(tmp_path / "x.wav"),
        capture_mode="both",
        tracks_path=str(tracks_wav),
    )
    segs = [
        {"start": 0.0, "end": 2.0, "text": "Hola equipo", "speaker": "Tú", "speaker_key": "me"},
        {"start": 2.2, "end": 5.0, "text": "Revisemos el presupuesto", "speaker": "Hablante 1", "speaker_key": "spk_1"},
    ]
    test_db.update_recording_transcript(rec["id"], "Tú: Hola equipo\nHablante 1: Revisemos el presupuesto", segs, "es", 5.0, "parakeet-tdt-0.6b-v3")
    test_db.update_recording_summary(rec["id"], "Resumen", status="ready", progress=1.0, stage="Listo")

    monkeypatch.setattr(app_mod, "db", test_db)
    client = TestClient(app_mod.app)

    res = client.patch(
        f"/api/sessions/{sess['id']}/recordings/{rec['id']}/speakers",
        json={"mapping": {"Hablante 1": "Ana"}},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["segments"][1]["speaker"] == "Ana"
    assert "Ana: Revisemos el presupuesto" in data["transcript_text"]
    hits = test_db.search_session_chunks(sess["id"], "presupuesto", limit=5)
    assert hits and "Ana:" in hits[0]["content"]
    test_db.close()


def test_appearance_settings_validation_and_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    import backend.app as app_mod

    test_db = Database(db_path=tmp_path / "appearance.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    defaults = test_db.get_settings()
    assert defaults["accent_color"] == "#c8ee26"
    assert defaults["font_scale"] == pytest.approx(1.0)

    monkeypatch.setattr(app_mod, "db", test_db)
    client = TestClient(app_mod.app)

    res = client.put(
        "/api/settings",
        json={"accent_color": "#3B82F6", "font_scale": 1.15},
    )
    assert res.status_code == 200
    assert res.json()["accent_color"] == "#3b82f6"
    assert res.json()["font_scale"] == pytest.approx(1.15)

    res_clamp = client.put(
        "/api/settings",
        json={"accent_color": "not-a-hex", "font_scale": 3.0},
    )
    assert res_clamp.status_code == 200
    assert res_clamp.json()["accent_color"] == "#c8ee26"
    assert res_clamp.json()["font_scale"] == pytest.approx(1.25)

    res_low = client.put("/api/settings", json={"font_scale": 0.2})
    assert res_low.status_code == 200
    assert res_low.json()["font_scale"] == pytest.approx(0.85)
    test_db.close()


def test_asr_model_status_and_modular_endpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    import backend.app as app_mod
    from backend import config as cfg

    models_dir = tmp_path / "whisper"
    asr_dir = tmp_path / "asr"
    hf_cache = tmp_path / "hf_hub"

    # Simulate ONLY small Whisper, Parakeet ONNX, and Qwen3 0.6B installed
    turbo_snap = models_dir / "models--mobiuslabsgmbh--faster-whisper-large-v3-turbo" / "snapshots" / "rev1"
    turbo_snap.mkdir(parents=True)
    (turbo_snap / "model.bin").write_bytes(b"ok")

    parakeet_snap = hf_cache / "models--istupakov--parakeet-tdt-0.6b-v3-onnx" / "snapshots" / "rev1"
    parakeet_snap.mkdir(parents=True)
    (parakeet_snap / "encoder.onnx").write_bytes(b"onnx")

    qwen_dir = asr_dir / "qwen3-asr-0.6b"
    qwen_dir.mkdir(parents=True)
    (qwen_dir / "config.json").write_text("{}", encoding="utf-8")
    (qwen_dir / "model.safetensors").write_bytes(b"weights")

    # Precise suffix check: large-v3-turbo must NOT mark large-v3 as installed
    assert cfg.is_asr_model_installed("large-v3-turbo", models_dir=models_dir, asr_dir=asr_dir, hf_cache_dir=hf_cache) is True
    assert cfg.is_asr_model_installed("large-v3", models_dir=models_dir, asr_dir=asr_dir, hf_cache_dir=hf_cache) is False
    assert cfg.is_asr_model_installed(cfg.PARAKEET_MODEL_ID, models_dir=models_dir, asr_dir=asr_dir, hf_cache_dir=hf_cache) is True
    assert cfg.is_asr_model_installed(cfg.CANARY_MODEL_ID, models_dir=models_dir, asr_dir=asr_dir, hf_cache_dir=hf_cache) is False
    assert cfg.is_asr_model_installed("qwen3-asr-0.6b", models_dir=models_dir, asr_dir=asr_dir, hf_cache_dir=hf_cache) is True
    assert cfg.is_asr_model_installed("qwen3-asr-1.7b", models_dir=models_dir, asr_dir=asr_dir, hf_cache_dir=hf_cache) is False

    listed = {
        m["id"]: m
        for m in cfg.list_available_whisper_models(models_dir=models_dir, asr_dir=asr_dir, hf_cache_dir=hf_cache)
    }
    assert listed["large-v3-turbo"]["installed"] is True
    assert listed["large-v3-turbo"]["can_delete"] is True
    assert listed["large-v3"]["installed"] is False
    assert listed["large-v3"]["can_download"] is True

    # Endpoint wiring check with mocked download/delete functions
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(app_mod, "download_asr_model", lambda mid: (calls.append(("dl", mid)), {"id": mid, "installed": True})[1])
    monkeypatch.setattr(app_mod, "delete_asr_model", lambda mid: (calls.append(("rm", mid)), True)[1])

    client = TestClient(app_mod.app)
    res_dl = client.post("/api/whisper/models/small/download")
    assert res_dl.status_code == 200
    assert res_dl.json()["status"] == "downloaded"
    assert ("dl", "small") in calls

    res_rm = client.delete("/api/whisper/models/small")
    assert res_rm.status_code == 200
    assert res_rm.json()["deleted"] is True
    assert ("rm", "small") in calls


def test_quiz_json_schema_and_hardened_parser(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi.testclient import TestClient
    import backend.app as app_mod
    from backend.context_manager import build_quiz_json_schema, parse_quiz_json

    for mode, expected_enum in [
        ("mixed", ["mcq", "true_false", "open"]),
        ("mcq", ["mcq"]),
        ("true_false", ["true_false"]),
        ("open", ["open"]),
    ]:
        schema = build_quiz_json_schema(mode)
        assert schema["properties"]["questions"]["items"]["properties"]["type"]["enum"] == expected_enum

    fenced_with_prose = (
        "Aquí tienes el examen [00:12]:\n"
        "```json\n"
        '{"preguntas": [{"pregunta": "¿Qué es Molang?", "type": "open", "expected_answer": "Un lenguaje de expresiones.", "pista": "Motor matemático", "timestamp": "apuntes.pdf · p. 2"}]}\n'
        "```\n"
        "Espero que sea útil [01:30]."
    )
    parsed = parse_quiz_json(fenced_with_prose, mode="mixed")
    assert len(parsed) == 1
    assert parsed[0]["type"] == "open"
    assert parsed[0]["hint"] == "Motor matemático"
    assert parsed[0]["expected_answer"] == "Un lenguaje de expresiones."

    test_db = Database(db_path=tmp_path / "quiz_schema.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    sess = test_db.create_session("Animación")
    rec = test_db.create_recording(sess["id"], "clase.wav", "clase.wav", "clase.wav")
    test_db.update_recording_transcript(rec["id"], "Blockbench es un IDE 3D.", [{"start": 1.0, "end": 4.0, "text": "Blockbench es un IDE 3D."}], "es", 4.0, "large-v3-turbo")
    test_db.update_recording_summary(rec["id"], "Resumen")

    seen_formats = []

    async def fake_summarize(messages, model_name, num_ctx, idle_unload_sec, **kwargs):
        seen_formats.append(kwargs.get("response_format"))
        return '{"questions": [{"id": 1, "type": "mcq", "question": "¿Qué es Blockbench?", "options": ["Un IDE", "Un codec", "Una GPU", "Un navegador"], "correct_index": 0, "explanation": "IDE 3D.", "hint": "Herramienta", "timestamp": "00:01"}]}'

    monkeypatch.setattr(app_mod, "db", test_db)
    monkeypatch.setattr(app_mod.resource_manager, "summarize_exclusive", fake_summarize)
    client = TestClient(app_mod.app)
    res = client.post(f"/api/sessions/{sess['id']}/quiz/generate", json={"num_questions": 3, "mode": "mcq"})
    assert res.status_code == 200
    assert seen_formats and isinstance(seen_formats[0], dict)
    assert seen_formats[0]["properties"]["questions"]["items"]["properties"]["type"]["enum"] == ["mcq"]
    test_db.close()


def test_quiz_pdf_whitespace_compaction_budget_and_truncated_salvage() -> None:
    from backend.context_manager import _compact_llm_whitespace, build_quiz_generation_messages, parse_quiz_json

    fragmented = "Hola\n \nmundo\n \nde\n \npruebas\n \ncon\n \nmuchas\n \nlineas\n \nrotas"
    compacted = _compact_llm_whitespace(fragmented)
    assert "\n \n" not in compacted
    assert compacted == "Hola mundo de pruebas con muchas lineas rotas"

    huge_chunks = [
        {"filename": "guia.pdf", "locator": f"p. {i}", "content": ("palabra\n \n" * 110)}
        for i in range(1, 15)
    ]
    srcs = [{"id": "s1", "filename": "guia.pdf", "source_type": "pdf", "status": "ready", "extracted_text": "palabra " * 5000}]
    msgs = build_quiz_generation_messages(
        recordings=[],
        combined_summary="Resumen " * 600,
        num_questions=10,
        mode="mixed",
        num_ctx=8192,
        sources=srcs,
        indexed_chunks=huge_chunks,
    )
    # Prompt must stay compact enough to leave >= 3500 tokens of output headroom in num_ctx=8192
    assert len(msgs[-1]["content"]) < 11000

    truncated_json = (
        '{"questions": ['
        '{"id": 1, "type": "mcq", "question": "¿Qué es IK?", "options": ["Cinemática Inversa", "Keyframes", "Shader", "Vértice"], "correct_index": 0, "explanation": "Cinemática Inversa.", "hint": "IK", "timestamp": "p. 1"},'
        '{"id": 2, "type": "true_false", "question": "Molang solo corre en CPU.", "options": ["Verdadero", "Falso"], "correct_index": 1, "explanation": "Tiene aceleración.", "hint": "Molang", "timestamp": "p. 2"},'
        '{"id": 3, "type": "open", "question": "Pregunta cortada'
    )
    salvaged = parse_quiz_json(truncated_json, mode="mixed")
    assert len(salvaged) == 2
    assert salvaged[0]["question"] == "¿Qué es IK?"
    assert salvaged[1]["correct_index"] == 1


def test_individual_recording_summary_endpoint_and_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient
    import backend.app as app_mod
    from backend.context_manager import build_recording_summary_messages, format_transcript_for_llm

    rec_sample = {
        "filename": "reunion.wav",
        "duration_sec": 75.0,
        "transcript_text": "Revisamos el despliegue [01:05] con el equipo.",
        "segments": [
            {"start": 5.0, "end": 10.0, "speaker": "Ana", "text": "Revisamos el despliegue."},
            {"start": 65.0, "end": 72.0, "speaker": "Tú", "text": "Confirmamos la fecha del viernes."},
        ],
    }
    formatted = format_transcript_for_llm(rec_sample)
    assert "[00:05] Ana:" in formatted
    assert "[01:05] Tú:" in formatted
    prompt_msgs = build_recording_summary_messages("reunion.wav", formatted, 75.0)
    assert "[MM:SS]" in prompt_msgs[0]["content"]
    assert "[01:05] Tú:" in prompt_msgs[1]["content"]

    test_db = Database(db_path=tmp_path / "rec_summary.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    test_db.update_settings({"eco_mode": True, "auto_summarize": False, "ollama_model": "gemma4:e4b"})
    sess = test_db.create_session("Sesión individual")
    rec1 = test_db.create_recording(sess["id"], "audio1.wav", "audio1.wav", "audio1.wav")
    rec2 = test_db.create_recording(sess["id"], "audio2.wav", "audio2.wav", "audio2.wav")
    test_db.update_recording_transcript(
        rec1["id"],
        "Primer bloque de prueba.",
        [{"start": 12.0, "end": 18.0, "speaker": "Hablante 1", "text": "Primer bloque de prueba."}],
        "es",
        18.0,
        "parakeet",
    )
    test_db.update_recording_transcript(
        rec2["id"],
        "Segundo audio sin cambios.",
        [{"start": 4.0, "end": 8.0, "speaker": "Tú", "text": "Segundo audio sin cambios."}],
        "es",
        8.0,
        "parakeet",
    )
    # Keep rec1 disabled to verify explicit single-recording resummarize still works on demand
    test_db.set_recording_enabled(rec1["id"], False)

    captured_prompts: list[str] = []

    async def fake_summarize(messages, model_name, num_ctx, idle_unload_sec, **kwargs):
        captured_prompts.append(messages[-1]["content"])
        return "### Resumen general\nResumen individual con cita [00:12]."

    combined_called = {"count": 0}

    async def fake_combined(session_id: str, extra_rec_id: str | None = None):
        combined_called["count"] += 1

    monkeypatch.setattr(app_mod, "db", test_db)
    monkeypatch.setattr(app_mod.resource_manager, "summarize_exclusive", fake_summarize)
    monkeypatch.setattr(app_mod, "regenerate_combined_summary_if_needed", fake_combined)

    client = TestClient(app_mod.app)
    res = client.post(
        f"/api/sessions/{sess['id']}/resummarize",
        json={"recording_id": rec1["id"], "update_combined": False},
    )
    assert res.status_code == 200
    body = res.json()
    recs_by_id = {r["id"]: r for r in body["recordings"]}
    assert recs_by_id[rec1["id"]]["summary_text"] == "### Resumen general\nResumen individual con cita [00:12]."
    assert recs_by_id[rec2["id"]]["summary_text"] == ""
    assert combined_called["count"] == 0
    assert len(captured_prompts) == 1
    assert "[00:12] Hablante 1: Primer bloque de prueba." in captured_prompts[0]
    test_db.close()


def test_rename_recording_and_source_updates_db_and_fts5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient
    import backend.app as app_mod

    test_db = Database(db_path=tmp_path / "rename_sources.db", audio_dir=tmp_path / "audio")
    test_db.init_db()
    sess = test_db.create_session("Redes")
    rec = test_db.create_recording(sess["id"], "grabacion_raw.wav", "grabacion_raw.wav", "grabacion_raw.wav")
    test_db.update_recording_transcript(
        rec["id"],
        "Conceptos iniciales de enrutamiento.",
        [{"start": 0.0, "end": 4.0, "text": "Conceptos iniciales de enrutamiento."}],
        "es",
        4.0,
        "parakeet",
    )
    test_db.update_recording_summary(rec["id"], "")
    test_db.replace_item_chunks(
        session_id=sess["id"],
        item_kind="recording",
        item_id=rec["id"],
        filename="grabacion_raw.wav",
        source_type="audio",
        chunks=[{"locator": "00:00", "content": "Conceptos iniciales de enrutamiento."}],
    )

    src = test_db.create_source(sess["id"], "doc_sin_nombre.pdf", "pdf", "doc_sin_nombre.pdf")
    test_db.update_source_content(
        src["id"],
        "Tabla de subredes.",
        [{"locator": "p. 1", "text": "Tabla de subredes."}],
        1,
    )
    test_db.update_source_summary(src["id"], "")
    test_db.replace_item_chunks(
        session_id=sess["id"],
        item_kind="source",
        item_id=src["id"],
        filename="doc_sin_nombre.pdf",
        source_type="pdf",
        chunks=[{"locator": "p. 1", "content": "Tabla de subredes."}],
    )

    monkeypatch.setattr(app_mod, "db", test_db)
    client = TestClient(app_mod.app)

    res_rec = client.patch(
        f"/api/sessions/{sess['id']}/recordings/{rec['id']}",
        json={"filename": "Clase de Redes 01.wav"},
    )
    assert res_rec.status_code == 200
    recs = {r["id"]: r for r in res_rec.json()["recordings"]}
    assert recs[rec["id"]]["filename"] == "Clase de Redes 01.wav"
    assert recs[rec["id"]]["enabled"] is True

    res_src = client.patch(
        f"/api/sessions/{sess['id']}/sources/{src['id']}",
        json={"filename": "Guia TCP-IP.pdf"},
    )
    assert res_src.status_code == 200
    srcs = {s["id"]: s for s in res_src.json()["sources"]}
    assert srcs[src["id"]]["filename"] == "Guia TCP-IP.pdf"
    assert srcs[src["id"]]["enabled"] is True

    # Empty filename must be rejected cleanly
    res_empty = client.patch(
        f"/api/sessions/{sess['id']}/sources/{src['id']}",
        json={"filename": "   "},
    )
    assert res_empty.status_code == 400

    # FTS5 index must reflect the updated filenames
    rec_hits = test_db.search_session_chunks(sess["id"], "Redes", limit=5)
    assert any(h["item_id"] == rec["id"] and h["filename"] == "Clase de Redes 01.wav" for h in rec_hits)
    src_hits = test_db.search_session_chunks(sess["id"], "TCP", limit=5)
    assert any(h["item_id"] == src["id"] and h["filename"] == "Guia TCP-IP.pdf" for h in src_hits)

    test_db.close()
