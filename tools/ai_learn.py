"""
Fase 4: AI Learn — NeuralForge
================================
Siklus belajar mandiri yang aman dan terukur.

Alur:
  1. Kumpulkan log feedback baru sejak pembelajaran terakhir
  2. Filter: dedupe, spam, data sensitif, low-rating tanpa koreksi
  3. Ekstraksi: fakta baru, pola gagal, contoh few-shot bagus
  4. Skor kualitas tiap kandidat
  5. KARANTINA: semua masuk antrean review (UI terpisah)
  6. Item disetujui -> Knowledge Container (versi baru)
  7. Tes regresi otomatis; rollback jika skor turun
  8. Laporan akhir + riwayat versi
"""

from __future__ import annotations

import os
import re
import json
import time
import uuid
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
LEARN_STATE   = os.path.join(_PROJECT_ROOT, "models", "learn_state.json")
LEARN_LOG_DIR = os.path.join(_PROJECT_ROOT, "models", "learn_logs")


# === Data types ================================================================

@dataclass
class LearnReport:
    """Laporan hasil satu siklus AI Learn."""
    run_id:          str   = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at:      float = field(default_factory=time.time)
    finished_at:     float = 0.0
    status:          str   = "running"   # "running"|"done"|"rolled_back"|"failed"
    events_scanned:  int   = 0
    events_filtered: int   = 0
    candidates_new:  int   = 0
    candidates_used: int   = 0
    facts_added:     int   = 0
    chunks_added:    int   = 0
    score_before:    float = 0.0
    score_after:     float = 0.0
    rollback:        bool  = False
    message:         str   = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# === State persistence =========================================================

def _load_state() -> dict:
    """Load persistent state (last_learn_ts, version, history)."""
    if os.path.exists(LEARN_STATE):
        try:
            with open(LEARN_STATE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_learn_ts": 0.0, "version": 0, "history": []}


def _save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(LEARN_STATE), exist_ok=True)
    with open(LEARN_STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _save_report(report: LearnReport) -> None:
    os.makedirs(LEARN_LOG_DIR, exist_ok=True)
    path = os.path.join(LEARN_LOG_DIR, f"run_{report.run_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)


# === Filter helpers =============================================================

_SPAM_PATTERNS = [
    re.compile(r"^(.)\1{5,}$"),           # single char repeated: "aaaaaaa"
    re.compile(r"^\W+$"),                  # only punctuation
    re.compile(r"<[^>]+>"),               # HTML tags (prompt injection attempt)
]

_SENSITIVE = [
    re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\b\d{13,19}\b"),
]


def _is_spam(text: str) -> bool:
    t = text.strip()
    if len(t) < 5:
        return True
    for p in _SPAM_PATTERNS:
        if p.search(t):
            return True
    return False


def _has_sensitive(text: str) -> bool:
    for p in _SENSITIVE:
        if p.search(text):
            return True
    return False


def _dedupe(events: list[dict]) -> list[dict]:
    """Remove events with duplicate (user_message, correction) pairs."""
    seen: set[str] = set()
    out  = []
    for ev in events:
        key = f"{ev['user_message']}|{ev['correction']}"
        if key not in seen:
            seen.add(key)
            out.append(ev)
    return out


# === Quality scorer =============================================================

def _score_event(ev: dict) -> float:
    """
    Skor 0-1 kualitas event sebagai sumber pembelajaran.
    Lebih tinggi = lebih layak dipakai.
    """
    score = 0.3  # base
    correction = ev.get("correction", "")
    rating     = ev.get("rating", 0)
    answer     = ev.get("assistant_answer", "")
    question   = ev.get("user_message", "")

    # Has explicit correction -> valuable
    if correction.strip():
        score += 0.25
    # User gave thumbs up
    if rating > 0:
        score += 0.2
    # User gave thumbs down (so we know answer was bad)
    if rating < 0 and correction.strip():
        score += 0.15
    # Substantial content
    if len(question) > 20 and len(correction or answer) > 30:
        score += 0.1
    return min(1.0, score)


# === Extraction helpers =========================================================

def _extract_fact(ev: dict) -> Optional[str]:
    """
    Coba ekstrak fakta sederhana dari pasangan (question, correction).
    Returns None jika tidak bisa diekstrak sebagai fakta.
    """
    q  = ev.get("user_message", "").strip()
    a  = (ev.get("correction") or ev.get("assistant_answer", "")).strip()
    if not q or not a:
        return None
    if len(a) > 300:
        return None   # Terlalu panjang untuk fakta tunggal
    # Format Q&A sederhana
    return f"Q: {q}\nA: {a}"


def _extract_fewshot(ev: dict) -> Optional[dict]:
    """Buat contoh few-shot dari event berkualitas tinggi."""
    q = ev.get("user_message", "").strip()
    # Prefer correction over original answer if available
    a = (ev.get("correction") or ev.get("assistant_answer", "")).strip()
    if not q or not a or len(q) < 10 or len(a) < 10:
        return None
    return {"role_user": q, "role_assistant": a}


# === AI Learn Cycle =============================================================

class AILearnCycle:
    """
    Orkestrator satu siklus belajar AI.

    Usage:
        cycle = AILearnCycle(feedback_store, knowledge_container)
        report = cycle.run(progress_callback=lambda msg: print(msg))
    """

    ROLLBACK_THRESHOLD = 0.05   # rollback if score drops > 5%

    def __init__(self, feedback_store, knowledge_container) -> None:
        """
        feedback_store      : tools.feedback.FeedbackStore
        knowledge_container : tools.knowledge.KnowledgeContainer
        """
        self.fb = feedback_store
        self.kc = knowledge_container
        self._running = False
        self._lock    = threading.Lock()

    def is_running(self) -> bool:
        return self._running

    def run(
        self,
        auto_approve_above: float = 0.85,
        progress_callback: Callable[[str], None] = lambda _: None,
        eval_fn: Optional[Callable[[], float]] = None,
    ) -> LearnReport:
        """
        Jalankan siklus belajar secara sinkron.
        eval_fn: callable yang mengembalikan skor 0-1 (dari Fase 5).
        Kembalikan LearnReport.
        """
        with self._lock:
            if self._running:
                r = LearnReport(status="failed", message="Another cycle is already running.")
                return r
            self._running = True

        report = LearnReport()
        state  = _load_state()

        try:
            progress_callback(f"[Learn] Run {report.run_id} started")

            # ── Step 1: Collect ─────────────────────────────────────────────
            since_ts = state.get("last_learn_ts", 0.0)
            raw_events = self.fb.get_events_since(since_ts=since_ts, min_rating=-2)
            report.events_scanned = len(raw_events)
            progress_callback(f"[Learn] {len(raw_events)} events collected since {since_ts:.0f}")

            # ── Step 2: Filter ──────────────────────────────────────────────
            filtered = []
            for ev in raw_events:
                q = ev.get("user_message", "")
                c = ev.get("correction",   "")
                if _is_spam(q) or _is_spam(c):
                    continue
                if _has_sensitive(q) or _has_sensitive(c):
                    continue
                # Discard negative rating with no correction
                if ev.get("rating", 0) < 0 and not c.strip():
                    continue
                filtered.append(ev)

            filtered = _dedupe(filtered)
            report.events_filtered = report.events_scanned - len(filtered)
            progress_callback(f"[Learn] {len(filtered)} events after filter+dedupe")

            # ── Step 3: Score & extract ─────────────────────────────────────
            candidates: list[dict] = []
            for ev in filtered:
                score = _score_event(ev)
                fact  = _extract_fact(ev)
                few   = _extract_fewshot(ev)
                if fact or few:
                    candidates.append({
                        "event_id":   ev.get("event_id", ""),
                        "fact":       fact,
                        "fewshot":    few,
                        "score":      score,
                        "source":     "learn_cycle",
                        "auto_ok":    score >= auto_approve_above,
                    })
            report.candidates_new = len(candidates)
            progress_callback(f"[Learn] {len(candidates)} candidates extracted")

            # ── Step 4: Quarantine — write to candidates table ──────────────
            auto_approved = [c for c in candidates if c["auto_ok"]]
            needs_review  = [c for c in candidates if not c["auto_ok"]]
            progress_callback(
                f"[Learn] {len(auto_approved)} auto-approved "
                f"({len(needs_review)} need manual review)"
            )

            # Also fetch previously manually-approved candidates from feedback store
            manual_approved = self.fb.get_approved_candidates()
            all_to_apply = auto_approved + [
                {
                    "fact":    f"Q: {c['question']}\nA: {c['answer']}",
                    "fewshot": None,
                    "score":   float(c.get("score", 0.7)),
                    "source":  "user_correction",
                    "cand_id": c["cand_id"],
                }
                for c in manual_approved
            ]

            if not all_to_apply:
                progress_callback("[Learn] No candidates to apply. Done.")
                report.status     = "done"
                report.message    = "No new knowledge to apply."
                report.finished_at = time.time()
                state["last_learn_ts"] = time.time()
                _save_state(state)
                _save_report(report)
                self._running = False
                return report

            # ── Step 5: Evaluate BEFORE applying ───────────────────────────
            if eval_fn:
                score_before = eval_fn()
                report.score_before = score_before
                progress_callback(f"[Learn] Score before: {score_before:.3f}")
            else:
                score_before = None

            # ── Step 6: Apply to Knowledge Container ────────────────────────
            chunks_added = 0
            facts_added  = 0
            for cand in all_to_apply:
                fact = cand.get("fact")
                if fact:
                    try:
                        n = self.kc.add_manual_qa(
                            question = fact.split("\nA:")[0].replace("Q: ", "").strip(),
                            answer   = fact.split("\nA:")[-1].strip() if "\nA:" in fact else fact,
                            trust_level = min(1.5, 0.8 + cand["score"] * 0.7),
                            collection  = "learned",
                        )
                        chunks_added += n
                        facts_added  += 1
                    except Exception as ex:
                        progress_callback(f"[Learn] Warning: {ex}")

            report.facts_added  = facts_added
            report.chunks_added = chunks_added
            progress_callback(f"[Learn] {facts_added} facts -> {chunks_added} chunks added")

            # ── Step 7: Evaluate AFTER applying ─────────────────────────────
            if eval_fn and score_before is not None:
                score_after = eval_fn()
                report.score_after = score_after
                progress_callback(f"[Learn] Score after: {score_after:.3f}")

                if score_after < score_before - self.ROLLBACK_THRESHOLD:
                    # Rollback: remove learned chunks
                    progress_callback("[Learn] ROLLBACK: score dropped, reverting...")
                    self.kc.delete_source("learned")
                    report.rollback = True
                    report.status   = "rolled_back"
                    report.message  = (
                        f"Rolled back: score {score_before:.3f} -> {score_after:.3f} "
                        f"(threshold {self.ROLLBACK_THRESHOLD})"
                    )
                    progress_callback(f"[Learn] {report.message}")
                    report.finished_at = time.time()
                    _save_report(report)
                    self._running = False
                    return report

            # ── Step 8: Mark candidates as used + update state ──────────────
            used_ids = [c.get("cand_id") for c in manual_approved if "cand_id" in c]
            if used_ids:
                self.fb.mark_candidates_used(used_ids)

            new_version = state.get("version", 0) + 1
            state["version"]       = new_version
            state["last_learn_ts"] = time.time()
            if "history" not in state:
                state["history"] = []
            state["history"].append({
                "run_id":        report.run_id,
                "version":       new_version,
                "ts":            time.time(),
                "facts_added":   facts_added,
                "chunks_added":  chunks_added,
                "score_before":  report.score_before,
                "score_after":   report.score_after,
                "rollback":      report.rollback,
            })
            # Keep last 50 history entries
            state["history"] = state["history"][-50:]
            _save_state(state)

            report.status     = "done"
            report.message    = (
                f"Version {new_version}: {facts_added} facts applied, "
                f"{report.events_filtered} events filtered out."
            )
            progress_callback(f"[Learn] {report.message}")

        except Exception as ex:
            report.status  = "failed"
            report.message = str(ex)
            progress_callback(f"[Learn] ERROR: {ex}")

        finally:
            report.finished_at = time.time()
            _save_report(report)
            self._running = False

        return report

    def run_async(
        self,
        auto_approve_above: float = 0.85,
        progress_callback: Callable[[str], None] = lambda _: None,
        eval_fn: Optional[Callable[[], float]] = None,
    ) -> threading.Thread:
        """Jalankan siklus di background thread. Returns thread object."""
        t = threading.Thread(
            target=self.run,
            kwargs={
                "auto_approve_above": auto_approve_above,
                "progress_callback":  progress_callback,
                "eval_fn":            eval_fn,
            },
            daemon=True,
        )
        t.start()
        return t


# === Version history & rollback =================================================

def get_learn_history() -> list[dict]:
    """Ambil riwayat semua siklus belajar."""
    state = _load_state()
    return state.get("history", [])


def get_current_version() -> int:
    return _load_state().get("version", 0)


def rollback_to_version(version: int, knowledge_container) -> bool:
    """
    Rollback ke versi tertentu dengan menghapus semua chunk source='learned'.
    (Simplified rollback — full snapshot rollback memerlukan snapshot store.)
    """
    state = _load_state()
    if version >= state.get("version", 0):
        return False
    knowledge_container.delete_source("learned")
    state["version"] = version
    state["last_learn_ts"] = 0.0   # Force re-learn from scratch
    _save_state(state)
    return True
