"""Automated core tests for TransCrypting database, context manager, resource lock, and API."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.config import list_available_whisper_models
from backend.context_manager import (
    build_chat_messages,
    estimate_session_context_usage,
    format_transcript_for_llm,
    parse_grade_json,
    parse_quiz_json,
    parse_suggestions_json,
)
from backend.database import Database
from backend.resource_manager import ResourceManager


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
        events.append(f"whisper_loaded:{model_name}:{device_pref}")
        return "cuda"

    def fake_transcribe(
        audio_path: Path,
        language: str = "auto",
        progress_callback=None,
        classroom_enhance: bool = True,
    ):
        events.append("whisper_transcribed")
        return {"text": "hola", "segments": [], "language": "es", "duration_sec": 1.0}

    def fake_whisper_unload() -> None:
        events.append("whisper_unloaded")

    with (
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

    async def fake_summarize(messages, model_name, num_ctx, idle_unload_sec):
        content = messages[-1]["content"]
        if "Respuesta del estudiante" in content:
            return '{"verdict": "correct", "score": 90, "feedback": "Exacto.", "missing_points": "", "timestamp": "05:01"}'
        return raw_quiz

    monkeypatch.setattr(app_module.resource_manager, "summarize_exclusive", fake_summarize)

    with TestClient(app_module.app) as client:
        g_resp = client.post(
            f"/api/sessions/{s['id']}/quiz/generate",
            json={"num_questions": 2, "mode": "mixed"},
        )
        assert g_resp.status_code == 200
        assert len(g_resp.json()["questions"]) == 2

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

    import clr

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
