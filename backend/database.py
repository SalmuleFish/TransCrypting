"""SQLite persistence layer for sessions, recordings, chat messages, and settings."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
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
)


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
    """Thread-safe SQLite database manager."""

    def __init__(self, db_path: Path = DB_PATH, audio_dir: Path = AUDIO_DIR) -> None:
        self.db_path = Path(db_path)
        self.audio_dir = Path(audio_dir)
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

                CREATE INDEX IF NOT EXISTS idx_recordings_session
                    ON recordings(session_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);
                """
            )
            for key, val in DEFAULT_SETTINGS.items():
                conn.execute(
                    "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
                    (key, val),
                )
            # Safe additive migration for existing databases
            session_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "suggested_prompts_json" not in session_cols:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN suggested_prompts_json TEXT NOT NULL DEFAULT '[]'"
                )
            # Recover recordings interrupted by an unexpected shutdown
            conn.execute(
                """
                UPDATE recordings
                SET status = 'error',
                    stage = 'Interrumpido',
                    error_message = 'El procesamiento se interrumpió al cerrar el programa.'
                WHERE status IN ('queued', 'preparing', 'transcribing', 'summarizing')
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
            session["messages"] = self.get_messages(session_id)
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
            session_audio_dir = self.audio_dir / session_id
            if session_audio_dir.exists():
                shutil.rmtree(session_audio_dir, ignore_errors=True)
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

    def delete_recording(self, rec_id: str) -> dict[str, Any] | None:
        with self._lock:
            rec = self.get_recording(rec_id)
            if not rec:
                return None
            conn = self._get_conn()
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


db = Database()
