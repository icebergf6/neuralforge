"""
Fase 3: Feedback & Learning Log — NeuralForge
===============================================
Menyimpan feedback pengguna (thumbs up/down + koreksi) secara aman dan terstruktur.

Fitur:
  1. Log setiap event (pesan, konteks, jawaban, rating, koreksi, versi prompt, waktu)
  2. Redaksi otomatis data sensitif (API key, email, password, nomor kartu)
  3. Opsi opt-out per sesi
  4. Koreksi valid -> kandidat knowledge (belum aktif, butuh review)
  5. Query log untuk keperluan AI Learn (Fase 4)
"""

from __future__ import annotations

import os
import re
import json
import time
import uuid
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Optional

# --- Paths ---
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_DB    = os.path.join(_PROJECT_ROOT, "models", "feedback.db")
OPT_OUT_FILE  = os.path.join(_PROJECT_ROOT, "models", "optout_sessions.json")

# --- Sensitive-data patterns (redact before storing) ---
_SENSITIVE_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
     "[EMAIL]"),
    (re.compile(r"\b(?:password|passwd|pw|secret|token|api[_\-]?key)\s*[:=]\s*\S+",
                re.IGNORECASE),
     "[REDACTED_CREDENTIAL]"),
    (re.compile(r"\b\d{13,19}\b"),
     "[CARD_NUMBER]"),
    (re.compile(r"\b(?:sk|pk)-[A-Za-z0-9]{20,}\b"),
     "[API_KEY]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
     "[SSN]"),
]


def _redact(text: str) -> str:
    """Hapus data sensitif dari teks sebelum disimpan."""
    if not text:
        return text
    for pattern, replacement in _SENSITIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# --- Data model ---
@dataclass
class FeedbackEvent:
    event_id:          str   = field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id:        str   = ""
    conv_id:           str   = ""
    timestamp:         float = field(default_factory=time.time)
    user_message:      str   = ""
    assistant_answer:  str   = ""
    rating:            int   = 0    # +1 = thumb up, -1 = thumb down, 0 = no rating
    correction:        str   = ""
    rag_sources:       str   = ""   # JSON list of sources used
    prompt_version:    str   = ""
    knowledge_version: str   = ""
    intent:            str   = ""
    latency_ms:        float = 0.0
    correction_status: str   = "pending"


# --- SQLite storage ---
class FeedbackStore:
    """
    Thread-safe SQLite store untuk feedback event.
    Dua tabel: 'events' (semua feedback) dan 'candidates' (koreksi yang menunggu review).
    """

    _lock = threading.Lock()

    def __init__(self, db_path: str = DEFAULT_DB) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self._optout: set[str] = self._load_optout()

    def _init_db(self) -> None:
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id          TEXT PRIMARY KEY,
                    session_id        TEXT,
                    conv_id           TEXT,
                    timestamp         REAL,
                    user_message      TEXT,
                    assistant_answer  TEXT,
                    rating            INTEGER DEFAULT 0,
                    correction        TEXT DEFAULT '',
                    rag_sources       TEXT DEFAULT '[]',
                    prompt_version    TEXT DEFAULT '',
                    knowledge_version TEXT DEFAULT '',
                    intent            TEXT DEFAULT '',
                    latency_ms        REAL DEFAULT 0,
                    correction_status TEXT DEFAULT 'pending',
                    created_at        TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS candidates (
                    cand_id    TEXT PRIMARY KEY,
                    event_id   TEXT,
                    question   TEXT,
                    answer     TEXT,
                    source     TEXT DEFAULT 'user_correction',
                    score      REAL DEFAULT 0.0,
                    status     TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (event_id) REFERENCES events(event_id)
                );

                CREATE INDEX IF NOT EXISTS idx_session  ON events (session_id);
                CREATE INDEX IF NOT EXISTS idx_rating   ON events (rating);
                CREATE INDEX IF NOT EXISTS idx_cand_st  ON candidates (status);
            """)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path, timeout=10)
        con.row_factory = sqlite3.Row
        return con

    # --- Opt-out ---
    def _load_optout(self) -> set[str]:
        if os.path.exists(OPT_OUT_FILE):
            try:
                with open(OPT_OUT_FILE, "r", encoding="utf-8") as f:
                    return set(json.load(f))
            except Exception:
                return set()
        return set()

    def _save_optout(self) -> None:
        os.makedirs(os.path.dirname(OPT_OUT_FILE), exist_ok=True)
        with open(OPT_OUT_FILE, "w", encoding="utf-8") as f:
            json.dump(list(self._optout), f)

    def opt_out(self, session_id: str) -> None:
        self._optout.add(session_id)
        self._save_optout()
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM events WHERE session_id = ?", (session_id,))

    def opt_in(self, session_id: str) -> None:
        self._optout.discard(session_id)
        self._save_optout()

    def is_opted_out(self, session_id: str) -> bool:
        return session_id in self._optout

    # --- Write ---
    def record(self, event: FeedbackEvent) -> str:
        """Simpan feedback. Auto-redact konten sensitif. Returns event_id."""
        if self.is_opted_out(event.session_id):
            return ""

        ev = FeedbackEvent(
            event_id          = event.event_id,
            session_id        = event.session_id,
            conv_id           = event.conv_id,
            timestamp         = event.timestamp,
            user_message      = _redact(event.user_message),
            assistant_answer  = _redact(event.assistant_answer),
            rating            = event.rating,
            correction        = _redact(event.correction),
            rag_sources       = event.rag_sources,
            prompt_version    = event.prompt_version,
            knowledge_version = event.knowledge_version,
            intent            = event.intent,
            latency_ms        = event.latency_ms,
            correction_status = event.correction_status,
        )

        with self._lock, self._connect() as con:
            con.execute("""
                INSERT OR REPLACE INTO events
                (event_id, session_id, conv_id, timestamp,
                 user_message, assistant_answer, rating, correction,
                 rag_sources, prompt_version, knowledge_version,
                 intent, latency_ms, correction_status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                ev.event_id, ev.session_id, ev.conv_id, ev.timestamp,
                ev.user_message, ev.assistant_answer, ev.rating, ev.correction,
                ev.rag_sources, ev.prompt_version, ev.knowledge_version,
                ev.intent, ev.latency_ms, ev.correction_status,
            ))

            if ev.correction.strip():
                cand_id = str(uuid.uuid4())[:12]
                score   = self._score_correction(ev)
                con.execute("""
                    INSERT INTO candidates
                    (cand_id, event_id, question, answer, source, score, status)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                    cand_id, ev.event_id,
                    ev.user_message, ev.correction,
                    "user_correction", score, "pending",
                ))
        return ev.event_id

    def update_rating(self, event_id: str, rating: int) -> bool:
        with self._lock, self._connect() as con:
            cur = con.execute(
                "UPDATE events SET rating=? WHERE event_id=?", (rating, event_id)
            )
        return cur.rowcount > 0

    def add_correction(self, event_id: str, correction: str) -> str:
        correction = _redact(correction)
        with self._lock, self._connect() as con:
            con.execute(
                "UPDATE events SET correction=? WHERE event_id=?",
                (correction, event_id)
            )
            row = con.execute(
                "SELECT user_message FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
            if not row:
                return ""
            cand_id = str(uuid.uuid4())[:12]
            con.execute("""
                INSERT INTO candidates (cand_id, event_id, question, answer, source, score, status)
                VALUES (?,?,?,?,?,?,?)
            """, (cand_id, event_id, row["user_message"], correction,
                  "user_correction", 0.7, "pending"))
        return cand_id

    # --- Read (for AI Learn) ---
    def get_events_since(
        self,
        since_ts: float = 0.0,
        min_rating: int = -2,
        limit: int = 500,
    ) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT * FROM events
                WHERE timestamp >= ? AND rating >= ?
                ORDER BY timestamp ASC LIMIT ?
            """, (since_ts, min_rating, limit)).fetchall()
        return [dict(r) for r in rows]

    def get_pending_candidates(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT c.*, e.user_message as question_orig
                FROM candidates c
                LEFT JOIN events e ON c.event_id = e.event_id
                WHERE c.status = 'pending'
                ORDER BY c.score DESC, c.created_at ASC
            """).fetchall()
        return [dict(r) for r in rows]

    def review_candidate(self, cand_id: str, action: str) -> bool:
        status = "approved" if action == "approve" else "rejected"
        with self._lock, self._connect() as con:
            cur = con.execute(
                "UPDATE candidates SET status=? WHERE cand_id=?", (status, cand_id)
            )
        return cur.rowcount > 0

    def get_approved_candidates(self) -> list[dict]:
        with self._connect() as con:
            rows = con.execute("""
                SELECT * FROM candidates
                WHERE status = 'approved'
            """).fetchall()
        return [dict(r) for r in rows]

    def mark_candidates_used(self, cand_ids: list[str]) -> None:
        with self._lock, self._connect() as con:
            for cid in cand_ids:
                con.execute(
                    "UPDATE candidates SET status='used' WHERE cand_id=?", (cid,)
                )

    def get_stats(self) -> dict:
        with self._connect() as con:
            total       = con.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            thumbs_up   = con.execute("SELECT COUNT(*) FROM events WHERE rating > 0").fetchone()[0]
            thumbs_down = con.execute("SELECT COUNT(*) FROM events WHERE rating < 0").fetchone()[0]
            corrections = con.execute("SELECT COUNT(*) FROM events WHERE correction != ''").fetchone()[0]
            pending     = con.execute("SELECT COUNT(*) FROM candidates WHERE status='pending'").fetchone()[0]
            approved    = con.execute("SELECT COUNT(*) FROM candidates WHERE status='approved'").fetchone()[0]
        return {
            "total_events":       total,
            "thumbs_up":          thumbs_up,
            "thumbs_down":        thumbs_down,
            "corrections":        corrections,
            "pending_candidates": pending,
            "approved_candidates": approved,
        }

    # --- Scoring ---
    @staticmethod
    def _score_correction(ev: FeedbackEvent) -> float:
        score = 0.5
        if ev.rating < 0:
            score += 0.2
        if len(ev.correction) > 30:
            score += 0.15
        if ev.correction.lower() != ev.assistant_answer.lower():
            score += 0.15
        return min(1.0, score)
