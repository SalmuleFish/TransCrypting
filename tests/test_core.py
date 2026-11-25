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
