"""SQLite persistence layer for sessions, recordings, chat messages, and settings."""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import threading
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.config import (
    AUDIO_DIR,
    DB_PATH,
    DEFAULT_IDLE_UNLOAD_SEC,
    DEFAULT_NUM_CTX,
    DEFAULT_WHISPER_DEVICE,
    DEFAULT_WHISPER_LANGUAGE,
    DEFAULT_WHISPER_MODEL,
    PODCASTS_DIR,
    SOURCES_DIR,
)

_FTS_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "a", "en", "con", "por", "para", "y", "o", "que", "es", "son", "fue", "se",
    "su", "sus", "lo", "le", "les", "me", "te", "nos", "como", "mas", "pero",
    "si", "no", "ya", "muy", "este", "esta", "estos", "estas", "ese", "esa",
    "eso", "sobre", "entre", "hasta", "desde", "hay", "ser", "estar", "cual",
    "cuales", "como", "donde", "cuando", "quien", "the", "and", "for", "with",
    "that", "this", "from", "are", "was", "were", "have", "has", "had",
}


def _normalize_fts_tokens(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFD", (text or "").lower())
    ascii_text = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    tokens = re.findall(r"[a-z0-9]{2,}", ascii_text)
    filtered = [t for t in tokens if t not in _FTS_STOPWORDS]
    chosen = filtered or tokens
    seen: list[str] = []
    for tok in chosen:
        if tok not in seen:
            seen.append(tok)
        if len(seen) >= 10:
            break
    return seen


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


DEFAULT_SETTINGS: dict[str, str] = {
    "ollama_model": "",
    "whisper_model": DEFAULT_WHISPER_MODEL,
    "whisper_device": DEFAULT_WHISPER_DEVICE,
    "whisper_language": DEFAULT_WHISPER_LANGUAGE,
    "num_ctx": str(DEFAULT_NUM_CTX),
    "idle_unload_sec": str(DEFAULT_IDLE_UNLOAD_SEC),
    "classroom_audio_enhance": "true",
    "enable_thinking": "false",
    "theme": "light",
}


class Database:
    """Thread-safe SQLite database manager with FTS5 persistent source indexing."""

    def __init__(
        self,
        db_path: Path = DB_PATH,
        audio_dir: Path = AUDIO_DIR,
        sources_dir: Path = SOURCES_DIR,
        podcasts_dir: Path = PODCASTS_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.audio_dir = Path(audio_dir)
        self.sources_dir = Path(sources_dir)
        self.podcasts_dir = Path(podcasts_dir)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
        return self._conn

    def close(self) -> None:
        """Close the SQLite connection cleanly."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def init_db(self) -> None:
        """Create tables and default settings if missing."""
        with self._lock:
            conn = self._get_conn()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    combined_summary TEXT NOT NULL DEFAULT '',
                    suggested_prompts_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS recordings (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    playable_path TEXT NOT NULL,
                    duration_sec REAL NOT NULL DEFAULT 0.0,
                    language TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0.0,
                    stage TEXT NOT NULL DEFAULT 'En cola',
                    error_message TEXT NOT NULL DEFAULT '',
                    transcript_text TEXT NOT NULL DEFAULT '',
                    segments_json TEXT NOT NULL DEFAULT '[]',
                    summary_text TEXT NOT NULL DEFAULT '',
                    whisper_model TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    mime_type TEXT NOT NULL DEFAULT '',
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0.0,
                    stage TEXT NOT NULL DEFAULT 'En cola',
                    error_message TEXT NOT NULL DEFAULT '',
                    extracted_text TEXT NOT NULL DEFAULT '',
                    pages_json TEXT NOT NULL DEFAULT '[]',
                    summary_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    item_kind TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    locator TEXT NOT NULL DEFAULT '',
                    chunk_index INTEGER NOT NULL DEFAULT 0,
                    content TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS source_chunks_fts USING fts5(
                    content,
                    filename,
                    locator,
                    content='source_chunks',
                    content_rowid='id',
                    tokenize='unicode61 remove_diacritics 2'
                );

                CREATE TRIGGER IF NOT EXISTS source_chunks_ai AFTER INSERT ON source_chunks BEGIN
                    INSERT INTO source_chunks_fts(rowid, content, filename, locator)
                    VALUES (new.id, new.content, new.filename, new.locator);
                END;

                CREATE TRIGGER IF NOT EXISTS source_chunks_ad AFTER DELETE ON source_chunks BEGIN
                    INSERT INTO source_chunks_fts(source_chunks_fts, rowid, content, filename, locator)
                    VALUES('delete', old.id, old.content, old.filename, old.locator);
                END;

                CREATE TRIGGER IF NOT EXISTS source_chunks_au AFTER UPDATE ON source_chunks BEGIN
                    INSERT INTO source_chunks_fts(source_chunks_fts, rowid, content, filename, locator)
                    VALUES('delete', old.id, old.content, old.filename, old.locator);
                    INSERT INTO source_chunks_fts(rowid, content, filename, locator)
                    VALUES (new.id, new.content, new.filename, new.locator);
                END;

                CREATE INDEX IF NOT EXISTS idx_recordings_session
                    ON recordings(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_sources_session
                    ON sources(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_chunks_session_item
                    ON source_chunks(session_id, item_kind, item_id);

                CREATE TABLE IF NOT EXISTS podcasts (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    focus_prompt TEXT NOT NULL DEFAULT '',
                    style TEXT NOT NULL DEFAULT 'conversation',
                    length TEXT NOT NULL DEFAULT 'medium',
                    language TEXT NOT NULL DEFAULT 'es',
                    source_refs_json TEXT NOT NULL DEFAULT '[]',
                    script_json TEXT NOT NULL DEFAULT '[]',
                    audio_path TEXT NOT NULL DEFAULT '',
                    duration_sec REAL NOT NULL DEFAULT 0.0,
                    tts_engine TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'queued',
                    progress REAL NOT NULL DEFAULT 0.0,
                    stage TEXT NOT NULL DEFAULT 'En cola',
                    error_message TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_podcasts_session
                    ON podcasts(session_id, created_at DESC);
                """
            )
            for key, val in DEFAULT_SETTINGS.items():
                conn.execute(
                    "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                    (key, val),
                )
            # Safe additive migrations for existing databases
            session_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "suggested_prompts_json" not in session_cols:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN suggested_prompts_json TEXT NOT NULL DEFAULT '[]'"
                )
            rec_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(recordings)").fetchall()
            }
            if "enabled" not in rec_cols:
                conn.execute(
                    "ALTER TABLE recordings ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1"
                )
            # Recover recordings & sources interrupted by an unexpected shutdown
            conn.execute(
                """
                UPDATE recordings
                SET status = 'error',
                    stage = 'Interrumpido',
                    error_message = 'El procesamiento se interrumpió al cerrar el programa.'
                WHERE status IN ('queued', 'preparing', 'transcribing', 'summarizing')
                """
            )
            conn.execute(
                """
                UPDATE sources
                SET status = 'error',
                    stage = 'Interrumpido',
                    error_message = 'El procesamiento de esta fuente se interrumpió al cerrar el programa.'
                WHERE status IN ('queued', 'extracting', 'indexing', 'summarizing')
                """
            )
            conn.execute(
                """
                UPDATE podcasts
                SET status = 'error',
                    stage = 'Interrumpido',
                    error_message = 'La generación del podcast se interrumpió al cerrar el programa.'
                WHERE status IN ('queued', 'scripting', 'synthesizing', 'mixing')
                """
            )
            conn.commit()

    def _row_to_recording(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["segments"] = json.loads(item.get("segments_json") or "[]")
        except json.JSONDecodeError:
            item["segments"] = []
        item.pop("segments_json", None)
        item["enabled"] = bool(item.get("enabled", 1))
        return item

    def _row_to_source(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        try:
            item["pages"] = json.loads(item.get("pages_json") or "[]")
        except json.JSONDecodeError:
            item["pages"] = []
        item.pop("pages_json", None)
        item["enabled"] = bool(item.get("enabled", 1))
        return item

    def _row_to_podcast(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for json_col, key in (("source_refs_json", "source_refs"), ("script_json", "script")):
            try:
                item[key] = json.loads(item.get(json_col) or "[]")
            except json.JSONDecodeError:
                item[key] = []
            item.pop(json_col, None)
        return item

    # --- Sessions ---

    def create_session(self, title: str = "Nueva conversación") -> dict[str, Any]:
        now = _utc_now()
        session_id = uuid.uuid4().hex[:12]
        clean_title = (title or "Nueva conversación").strip()[:120] or "Nueva conversación"
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO sessions (id, title, combined_summary, created_at, updated_at)
                VALUES (?, ?, '', ?, ?)
                """,
                (session_id, clean_title, now, now),
            )
            conn.commit()
        return self.get_session(session_id)  # type: ignore[return-value]

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT
                    s.*,
                    (SELECT COUNT(*) FROM recordings r WHERE r.session_id = s.id) AS recording_count,
                    (SELECT COUNT(*) FROM recordings r WHERE r.session_id = s.id AND r.status = 'ready') AS ready_count,
                    (SELECT COUNT(*) FROM sources src WHERE src.session_id = s.id) AS source_count,
                    (SELECT COUNT(*) FROM sources src WHERE src.session_id = s.id AND src.status = 'ready') AS source_ready_count,
                    (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
            return [dict(r) for r in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_session_detail(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            session = self.get_session(session_id)
            if not session:
                return None
            try:
                session["suggested_prompts"] = json.loads(
                    session.get("suggested_prompts_json") or "[]"
                )
            except json.JSONDecodeError:
                session["suggested_prompts"] = []
            session.pop("suggested_prompts_json", None)
            session["recordings"] = self.list_recordings(session_id)
            session["sources"] = self.list_sources(session_id)
            session["podcasts"] = self.list_podcasts(session_id)
            session["messages"] = self.get_messages(session_id)
            session["indexed_chunks_count"] = self.count_session_chunks(
                session_id, enabled_only=True
            )
            return session

    def update_session_suggestions(
        self, session_id: str, prompts: list[dict[str, str]]
    ) -> bool:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "UPDATE sessions SET suggested_prompts_json = ? WHERE id = ?",
                (json.dumps(prompts, ensure_ascii=False), session_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def update_session_title(self, session_id: str, title: str) -> bool:
        clean_title = title.strip()[:120]
        if not clean_title:
            return False
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (clean_title, _utc_now(), session_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def update_session_summary(self, session_id: str, combined_summary: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                "UPDATE sessions SET combined_summary = ?, updated_at = ? WHERE id = ?",
                (combined_summary, _utc_now(), session_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            deleted = cur.rowcount > 0
        if deleted:
            for base_dir in (self.audio_dir, self.sources_dir, self.podcasts_dir):
                session_dir = base_dir / session_id
                if session_dir.exists():
                    shutil.rmtree(session_dir, ignore_errors=True)
        return deleted

    # --- Recordings ---

    def create_recording(
        self,
        session_id: str,
        filename: str,
        stored_path: str,
        playable_path: str,
        whisper_model: str = "",
        rec_id: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        recording_id = rec_id or uuid.uuid4().hex[:12]
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO recordings (
                    id, session_id, filename, stored_path, playable_path,
                    status, progress, stage, whisper_model, created_at
                )
                VALUES (?, ?, ?, ?, ?, 'queued', 0.0, 'En cola', ?, ?)
                """,
                (recording_id, session_id, filename, stored_path, playable_path, whisper_model, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
        return self.get_recording(recording_id)  # type: ignore[return-value]

    def get_recording(self, rec_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM recordings WHERE id = ?", (rec_id,)).fetchone()
            return self._row_to_recording(row) if row else None

    def list_recordings(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM recordings WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            return [self._row_to_recording(r) for r in rows]

    def update_recording_progress(
        self,
        rec_id: str,
        status: str,
        progress: float,
        stage: str,
        duration_sec: float | None = None,
        playable_path: str | None = None,
    ) -> None:
        clamped = max(0.0, min(1.0, float(progress)))
        with self._lock:
            conn = self._get_conn()
            fields = ["status = ?", "progress = ?", "stage = ?"]
            params: list[Any] = [status, clamped, stage]
            if duration_sec is not None:
                fields.append("duration_sec = ?")
                params.append(float(duration_sec))
            if playable_path is not None:
                fields.append("playable_path = ?")
                params.append(playable_path)
            params.append(rec_id)
            conn.execute(f"UPDATE recordings SET {', '.join(fields)} WHERE id = ?", params)
            conn.commit()

    def update_recording_transcript(
        self,
        rec_id: str,
        transcript_text: str,
        segments: list[dict[str, Any]],
        language: str,
        duration_sec: float,
        whisper_model: str,
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                UPDATE recordings
                SET transcript_text = ?,
                    segments_json = ?,
                    language = ?,
                    duration_sec = ?,
                    whisper_model = ?,
                    error_message = ''
                WHERE id = ?
                """,
                (
                    transcript_text,
                    json.dumps(segments, ensure_ascii=False),
                    language,
                    float(duration_sec),
                    whisper_model,
                    rec_id,
                ),
            )
            conn.commit()

    def update_recording_summary(
        self,
        rec_id: str,
        summary_text: str,
        status: str = "ready",
        progress: float = 1.0,
        stage: str = "Completado",
    ) -> None:
        now = _utc_now()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                UPDATE recordings
                SET summary_text = ?,
                    status = ?,
                    progress = ?,
                    stage = ?,
                    error_message = ''
                WHERE id = ?
                """,
                (summary_text, status, progress, stage, rec_id),
            )
            rec = self.get_recording(rec_id)
            if rec:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now, rec["session_id"]),
                )
            conn.commit()

    def update_recording_error(self, rec_id: str, error_message: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                UPDATE recordings
                SET status = 'error',
                    stage = 'Error',
                    error_message = ?
                WHERE id = ?
                """,
                (error_message, rec_id),
            )
            conn.commit()

    def set_recording_enabled(self, rec_id: str, enabled: bool) -> dict[str, Any] | None:
        val = 1 if enabled else 0
        now = _utc_now()
        with self._lock:
            conn = self._get_conn()
            rec = self.get_recording(rec_id)
            if not rec:
                return None
            conn.execute("UPDATE recordings SET enabled = ? WHERE id = ?", (val, rec_id))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, rec["session_id"]),
            )
            conn.commit()
        return self.get_recording(rec_id)

    def delete_recording(self, rec_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self.get_recording(rec_id)
            if not rec:
                return None
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM source_chunks WHERE item_kind = 'recording' AND item_id = ?",
                (rec_id,),
            )
            conn.execute("DELETE FROM recordings WHERE id = ?", (rec_id,))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_utc_now(), rec["session_id"]),
            )
            conn.commit()
        for path_key in ("stored_path", "playable_path"):
            p = rec.get(path_key)
            if p:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError:
                    pass
        return rec

    # --- Sources (PDF / PPTX / DOCX / TXT / Image) ---

    def create_source(
        self,
        session_id: str,
        filename: str,
        source_type: str,
        stored_path: str,
        mime_type: str = "",
        size_bytes: int = 0,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        sid = source_id or uuid.uuid4().hex[:12]
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO sources (
                    id, session_id, filename, source_type, stored_path, mime_type,
                    size_bytes, page_count, enabled, status, progress, stage, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, 'queued', 0.0, 'En cola', ?)
                """,
                (sid, session_id, filename, source_type, stored_path, mime_type, int(size_bytes), now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
        return self.get_source(sid)  # type: ignore[return-value]

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
            return self._row_to_source(row) if row else None

    def list_sources(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM sources WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
            return [self._row_to_source(r) for r in rows]

    def update_source_progress(
        self,
        source_id: str,
        status: str,
        progress: float,
        stage: str,
        page_count: int | None = None,
    ) -> None:
        clamped = max(0.0, min(1.0, float(progress)))
        with self._lock:
            conn = self._get_conn()
            fields = ["status = ?", "progress = ?", "stage = ?"]
            params: list[Any] = [status, clamped, stage]
            if page_count is not None:
                fields.append("page_count = ?")
                params.append(int(page_count))
            params.append(source_id)
            conn.execute(f"UPDATE sources SET {', '.join(fields)} WHERE id = ?", params)
            conn.commit()

    def update_source_content(
        self,
        source_id: str,
        extracted_text: str,
        pages: list[dict[str, Any]],
        page_count: int | None = None,
    ) -> None:
        count = page_count if page_count is not None else len(pages)
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                UPDATE sources
                SET extracted_text = ?,
                    pages_json = ?,
                    page_count = ?,
                    error_message = ''
                WHERE id = ?
                """,
                (
                    extracted_text,
                    json.dumps(pages, ensure_ascii=False),
                    int(count),
                    source_id,
                ),
            )
            conn.commit()

    def update_source_summary(
        self,
        source_id: str,
        summary_text: str,
        status: str = "ready",
        progress: float = 1.0,
        stage: str = "Indexado y listo",
    ) -> None:
        now = _utc_now()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                UPDATE sources
                SET summary_text = ?,
                    status = ?,
                    progress = ?,
                    stage = ?,
                    error_message = ''
                WHERE id = ?
                """,
                (summary_text, status, float(progress), stage, source_id),
            )
            src = self.get_source(source_id)
            if src:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now, src["session_id"]),
                )
            conn.commit()

    def update_source_error(self, source_id: str, error_message: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                UPDATE sources
                SET status = 'error',
                    stage = 'Error',
                    error_message = ?
                WHERE id = ?
                """,
                (error_message, source_id),
            )
            conn.commit()

    def set_source_enabled(self, source_id: str, enabled: bool) -> dict[str, Any] | None:
        val = 1 if enabled else 0
        now = _utc_now()
        with self._lock:
            conn = self._get_conn()
            src = self.get_source(source_id)
            if not src:
                return None
            conn.execute("UPDATE sources SET enabled = ? WHERE id = ?", (val, source_id))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, src["session_id"]),
            )
            conn.commit()
        return self.get_source(source_id)

    def delete_source(self, source_id: str) -> dict[str, Any] | None:
        with self._lock:
            src = self.get_source(source_id)
            if not src:
                return None
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM source_chunks WHERE item_kind = 'source' AND item_id = ?",
                (source_id,),
            )
            conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_utc_now(), src["session_id"]),
            )
            conn.commit()
        stored = src.get("stored_path")
        if stored:
            try:
                Path(stored).unlink(missing_ok=True)
            except OSError:
                pass
        return src

    # --- Persistent FTS5 Source Index ---

    def replace_item_chunks(
        self,
        session_id: str,
        item_kind: str,
        item_id: str,
        filename: str,
        source_type: str,
        chunks: list[dict[str, str]],
    ) -> int:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM source_chunks WHERE item_kind = ? AND item_id = ?",
                (item_kind, item_id),
            )
            rows = []
            for idx, chunk in enumerate(chunks):
                text = (chunk.get("content") or "").strip()
                if not text:
                    continue
                locator = (chunk.get("locator") or "").strip()
                rows.append(
                    (session_id, item_kind, item_id, filename, source_type, locator, idx, text)
                )
            if rows:
                conn.executemany(
                    """
                    INSERT INTO source_chunks (
                        session_id, item_kind, item_id, filename, source_type,
                        locator, chunk_index, content
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            conn.commit()
            return len(rows)

    def delete_item_chunks(self, item_kind: str, item_id: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "DELETE FROM source_chunks WHERE item_kind = ? AND item_id = ?",
                (item_kind, item_id),
            )
            conn.commit()

    def count_session_chunks(self, session_id: str, enabled_only: bool = True) -> int:
        with self._lock:
            conn = self._get_conn()
            if not enabled_only:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM source_chunks WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                return int(row["cnt"] if row else 0)
            row = conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM source_chunks c
                LEFT JOIN recordings r ON c.item_kind = 'recording' AND c.item_id = r.id
                LEFT JOIN sources s ON c.item_kind = 'source' AND c.item_id = s.id
                WHERE c.session_id = ?
                  AND (
                      (c.item_kind = 'recording' AND r.status = 'ready' AND COALESCE(r.enabled, 1) = 1)
                      OR (c.item_kind = 'source' AND s.status = 'ready' AND COALESCE(s.enabled, 1) = 1)
                  )
                """,
                (session_id,),
            ).fetchone()
            return int(row["cnt"] if row else 0)

    def ensure_recording_chunks_indexed(self, session_id: str) -> None:
        """Lazily index pre-existing ready recordings that lack FTS5 chunks."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                """
                SELECT r.*
                FROM recordings r
                WHERE r.session_id = ?
                  AND r.status = 'ready'
                  AND r.transcript_text != ''
                  AND NOT EXISTS (
                      SELECT 1 FROM source_chunks c
                      WHERE c.item_kind = 'recording' AND c.item_id = r.id
                  )
                """,
                (session_id,),
            ).fetchall()
        if not rows:
            return
        from backend.source_service import chunk_recording_for_index

        for row in rows:
            rec = self._row_to_recording(row)
            chunks = chunk_recording_for_index(rec)
            if chunks:
                self.replace_item_chunks(
                    session_id=session_id,
                    item_kind="recording",
                    item_id=rec["id"],
                    filename=rec["filename"],
                    source_type="audio",
                    chunks=chunks,
                )

    def search_session_chunks(
        self,
        session_id: str,
        query: str,
        limit: int = 12,
        enabled_only: bool = True,
    ) -> list[dict[str, Any]]:
        """Query the persistent FTS5 index scoped to active sources & recordings."""
        safe_limit = max(1, min(50, int(limit)))
        tokens = _normalize_fts_tokens(query)
        enabled_clause = (
            """
            AND (
                (c.item_kind = 'recording' AND r.status = 'ready' AND COALESCE(r.enabled, 1) = 1)
                OR (c.item_kind = 'source' AND s.status = 'ready' AND COALESCE(s.enabled, 1) = 1)
            )
            """
            if enabled_only
            else ""
        )

        with self._lock:
            conn = self._get_conn()
            results: list[dict[str, Any]] = []
            if tokens:
                fts_expr = " OR ".join(f"{tok}*" for tok in tokens)
                try:
                    rows = conn.execute(
                        f"""
                        SELECT c.*, bm25(source_chunks_fts) AS rank
                        FROM source_chunks_fts
                        JOIN source_chunks c ON c.id = source_chunks_fts.rowid
                        LEFT JOIN recordings r ON c.item_kind = 'recording' AND c.item_id = r.id
                        LEFT JOIN sources s ON c.item_kind = 'source' AND c.item_id = s.id
                        WHERE source_chunks_fts MATCH ?
                          AND c.session_id = ?
                          {enabled_clause}
                        ORDER BY rank ASC, c.id ASC
                        LIMIT ?
                        """,
                        (fts_expr, session_id, safe_limit),
                    ).fetchall()
                    results = [dict(r) for r in rows]
                except sqlite3.OperationalError:
                    results = []

            if results:
                return results

            # Fallback: return evenly distributed chunks across active items
            all_rows = conn.execute(
                f"""
                SELECT c.*
                FROM source_chunks c
                LEFT JOIN recordings r ON c.item_kind = 'recording' AND c.item_id = r.id
                LEFT JOIN sources s ON c.item_kind = 'source' AND c.item_id = s.id
                WHERE c.session_id = ?
                  {enabled_clause}
                ORDER BY c.item_kind ASC, c.item_id ASC, c.chunk_index ASC
                """,
                (session_id,),
            ).fetchall()
            if not all_rows:
                return []
            if len(all_rows) <= safe_limit:
                return [dict(r) for r in all_rows]
            step = max(1, len(all_rows) // safe_limit)
            picked = [dict(all_rows[i]) for i in range(0, len(all_rows), step)][:safe_limit]
            return picked

    # --- Chat Messages ---

    def add_message(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        now = _utc_now()
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
            msg_id = cur.lastrowid
            row = conn.execute("SELECT * FROM messages WHERE id = ?", (msg_id,)).fetchone()
            return dict(row)

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def clear_session_messages(self, session_id: str) -> int:
        """Delete ONLY chat messages for the session, keeping recordings, transcripts and summaries."""
        with self._lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_utc_now(), session_id),
            )
            conn.commit()
            return cur.rowcount

    # --- Settings ---

    @staticmethod
    def _parse_bool(val: str | bool, default: bool = False) -> bool:
        if isinstance(val, bool):
            return val
        low = str(val).strip().lower()
        if low in ("true", "1", "yes", "si", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        return default

    def get_settings(self) -> dict[str, Any]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute("SELECT key, value FROM settings").fetchall()
            raw = {**DEFAULT_SETTINGS, **{r["key"]: r["value"] for r in rows}}
        return {
            "ollama_model": raw["ollama_model"],
            "whisper_model": raw["whisper_model"],
            "whisper_device": raw["whisper_device"],
            "whisper_language": raw["whisper_language"],
            "num_ctx": int(raw["num_ctx"]),
            "idle_unload_sec": int(raw["idle_unload_sec"]),
            "classroom_audio_enhance": self._parse_bool(raw.get("classroom_audio_enhance", "true"), True),
            "enable_thinking": self._parse_bool(raw.get("enable_thinking", "false"), False),
            "theme": "dark" if str(raw.get("theme", "light")).lower() == "dark" else "light",
        }

    def update_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        allowed = set(DEFAULT_SETTINGS.keys())
        bool_keys = {"classroom_audio_enhance", "enable_thinking"}
        with self._lock:
            conn = self._get_conn()
            for key, val in updates.items():
                if key in allowed and val is not None:
                    stored = ("true" if self._parse_bool(val) else "false") if key in bool_keys else str(val)
                    conn.execute(
                        """
                        INSERT INTO settings (key, value) VALUES (?, ?)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value
                        """,
                        (key, stored),
                    )
            conn.commit()
        return self.get_settings()

    # --- Podcasts ---

    def create_podcast(
        self,
        session_id: str,
        title: str,
        focus_prompt: str = "",
        style: str = "conversation",
        length: str = "medium",
        language: str = "es",
        source_refs: list[str] | None = None,
        podcast_id: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        pid = podcast_id or uuid.uuid4().hex[:12]
        clean_title = (title or "Podcast de estudio").strip()[:120] or "Podcast de estudio"
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                INSERT INTO podcasts (
                    id, session_id, title, focus_prompt, style, length, language,
                    source_refs_json, script_json, status, progress, stage, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, '[]', 'queued', 0.0, 'En cola', ?)
                """,
                (
                    pid,
                    session_id,
                    clean_title,
                    (focus_prompt or "").strip(),
                    style or "conversation",
                    length or "medium",
                    language or "es",
                    json.dumps(source_refs or [], ensure_ascii=False),
                    now,
                ),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            conn.commit()
        return self.get_podcast(pid)  # type: ignore[return-value]

    def get_podcast(self, podcast_id: str) -> dict[str, Any] | None:
        with self._lock:
            conn = self._get_conn()
            row = conn.execute("SELECT * FROM podcasts WHERE id = ?", (podcast_id,)).fetchone()
            return self._row_to_podcast(row) if row else None

    def list_podcasts(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT * FROM podcasts WHERE session_id = ? ORDER BY created_at DESC",
                (session_id,),
            ).fetchall()
            return [self._row_to_podcast(r) for r in rows]

    def update_podcast_progress(
        self,
        podcast_id: str,
        status: str,
        progress: float,
        stage: str,
    ) -> None:
        clamped = max(0.0, min(1.0, float(progress)))
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE podcasts SET status = ?, progress = ?, stage = ? WHERE id = ?",
                (status, clamped, stage, podcast_id),
            )
            conn.commit()

    def update_podcast_script(
        self,
        podcast_id: str,
        script: list[dict[str, str]],
    ) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                "UPDATE podcasts SET script_json = ? WHERE id = ?",
                (json.dumps(script, ensure_ascii=False), podcast_id),
            )
            conn.commit()

    def update_podcast_ready(
        self,
        podcast_id: str,
        audio_path: str,
        duration_sec: float,
        tts_engine: str,
    ) -> None:
        now = _utc_now()
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                UPDATE podcasts
                SET audio_path = ?,
                    duration_sec = ?,
                    tts_engine = ?,
                    status = 'ready',
                    progress = 1.0,
                    stage = 'Listo',
                    error_message = ''
                WHERE id = ?
                """,
                (audio_path, float(duration_sec), tts_engine, podcast_id),
            )
            pod = self.get_podcast(podcast_id)
            if pod:
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (now, pod["session_id"]),
                )
            conn.commit()

    def update_podcast_error(self, podcast_id: str, error_message: str) -> None:
        with self._lock:
            conn = self._get_conn()
            conn.execute(
                """
                UPDATE podcasts
                SET status = 'error',
                    stage = 'Error',
                    error_message = ?
                WHERE id = ?
                """,
                (error_message, podcast_id),
            )
            conn.commit()

    def delete_podcast(self, podcast_id: str) -> dict[str, Any] | None:
        with self._lock:
            pod = self.get_podcast(podcast_id)
            if not pod:
                return None
            conn = self._get_conn()
            conn.execute("DELETE FROM podcasts WHERE id = ?", (podcast_id,))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_utc_now(), pod["session_id"]),
            )
            conn.commit()
        audio_path = pod.get("audio_path")
        if audio_path:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError:
                pass
        return pod


db = Database()
