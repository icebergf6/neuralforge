"""
Fase 1: Conversation Pipeline — NeuralForge
============================================
Pipeline per-pesan yang modular dan terukur untuk membuat percakapan benar-benar nyambung.

Modul:
  1. InputNormalizer    — bersihkan teks, deteksi bahasa, tangani slang/typo Indonesia
  2. ShortTermMemory    — N giliran terakhir + rolling summary otomatis
  3. UserFactMemory     — simpan preferensi/nama/konteks pengguna per sesi
  4. IntentClassifier   — klasifikasi maksud dengan confidence score
  5. ReferenceResolver  — ubah "itu/yang tadi" menjadi entitas eksplisit
  6. AnswerPlanner      — putuskan: jawab langsung / tanya balik / akui tidak tahu
  7. PromptBuilder      — bangun prompt terpusat, terversi, konsisten
  8. AnswerVerifier     — cek relevance + consistency + source support
  9. ConversationLogger — structured logging per langkah dengan conversation ID
"""

from __future__ import annotations

import re
import time
import uuid
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional

# ─── Structured Logger ────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, "conversation.jsonl"), encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter("%(message)s"))

conv_logger = logging.getLogger("neuralforge.conversation")
conv_logger.setLevel(logging.DEBUG)
conv_logger.addHandler(_file_handler)
conv_logger.propagate = False  # don't bubble up to root logger


def _log_step(conv_id: str, step: str, data: dict) -> None:
    """Emit one structured JSON log line."""
    record = {"ts": time.time(), "conv_id": conv_id, "step": step, **data}
    conv_logger.info(json.dumps(record, ensure_ascii=False))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class Intent(str, Enum):
    QUESTION        = "question"        # pertanyaan umum
    COMMAND         = "command"         # perintah / instruksi
    CHITCHAT        = "chitchat"        # obrolan santai
    CLARIFICATION   = "clarification"  # meminta klarifikasi
    CONTINUATION    = "continuation"   # melanjutkan topik sebelumnya
    OUT_OF_SCOPE    = "out_of_scope"    # di luar kemampuan sistem
    FEEDBACK        = "feedback"        # pujian/kritik terhadap AI
    UNKNOWN         = "unknown"


@dataclass
class Turn:
    role: str       # "user" | "assistant"
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class PipelineContext:
    """Hasil lengkap dari satu siklus pipeline untuk satu pesan user."""
    conv_id: str
    raw_input: str
    normalized_input: str       = ""
    detected_lang: str          = "id"          # "id" | "en" | "mixed"
    resolved_input: str         = ""            # setelah reference resolution
    intent: Intent              = Intent.UNKNOWN
    intent_confidence: float    = 0.0
    should_clarify: bool        = False
    clarification_question: str = ""
    rag_context: str            = ""
    rag_sources: list[str]      = field(default_factory=list)
    episodic_context: str       = ""
    user_facts: dict            = field(default_factory=dict)
    prompt: str                 = ""
    prompt_version: str         = "v1.0"
    answer: str                 = ""
    answer_verified: bool       = False
    verification_score: float   = 0.0
    latency_ms: float           = 0.0
    steps_log: list[dict]       = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. INPUT NORMALIZER
# ═══════════════════════════════════════════════════════════════════════════════

# Kamus slang Indonesia → bentuk baku
_ID_SLANG: dict[str, str] = {
    # Kata ganti / sapaan
    "gw": "saya", "gue": "saya", "aku": "saya", "w": "saya",
    "lo": "kamu", "lu": "kamu", "elo": "kamu", "u": "kamu",
    # Kata umum
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena",
    "karna": "karena", "krna": "karena",
    "udah": "sudah", "udh": "sudah", "dah": "sudah",
    "blm": "belum", "blum": "belum",
    "jg": "juga", "juga": "juga",
    "aja": "saja", "doang": "saja",
    "bgt": "banget", "bngt": "banget",
    "ga": "tidak", "gak": "tidak", "ngga": "tidak", "nggak": "tidak",
    "gk": "tidak", "g": "tidak",
    "sy": "saya", "sdh": "sudah", "spt": "seperti",
    "dll": "dan lain-lain", "dsb": "dan sebagainya",
    "tp": "tapi", "tapi": "tapi",
    "org": "orang", "bisa": "bisa",
    "gmn": "bagaimana", "gmana": "bagaimana", "gimana": "bagaimana",
    "knp": "kenapa", "knapa": "kenapa",
    "kpn": "kapan", "kpan": "kapan",
    "dmn": "di mana", "dimana": "di mana",
    "sm": "sama", "bareng": "bersama",
    "trs": "terus", "lanjut": "lanjutkan",
    "lagi": "lagi", "lg": "lagi",
    "makasih": "terima kasih", "mks": "terima kasih", "thx": "terima kasih",
    "ok": "baik", "oke": "baik", "okey": "baik",
    "sip": "baik", "siap": "baik",
    "btw": "omong-omong", "fyi": "sebagai informasi",
    "asap": "sesegera mungkin",
    "idk": "saya tidak tahu", "imo": "menurut saya",
    "ntar": "nanti", "ntaar": "nanti",
    "emg": "memang", "emang": "memang",
    "kek": "seperti", "kayak": "seperti",
    "bener": "benar", "bnar": "benar",
    "mau": "ingin", "mo": "ingin",
}

# Deteksi kata Indonesia yang kuat
_ID_KEYWORDS = {
    "apa", "siapa", "dimana", "kenapa", "kapan", "bagaimana", "berapa",
    "yang", "dengan", "untuk", "karena", "sudah", "akan", "bisa", "tidak",
    "ini", "itu", "dan", "atau", "di", "ke", "dari", "juga", "saya", "kamu",
    "adalah", "merupakan", "memiliki", "dapat", "harus",
}


class InputNormalizer:
    """
    Modul 1: Normalkan input pengguna.
    - Bersihkan whitespace & karakter berulang
    - Normalisasi slang Indonesia
    - Deteksi bahasa (id / en / mixed)
    """

    def normalize(self, text: str) -> tuple[str, str]:
        """
        Returns (normalized_text, detected_language).
        """
        # 1. Strip & collapse whitespace
        text = text.strip()
        text = re.sub(r"\s+", " ", text)

        # 2. Remove repeated characters (e.g. "heyyy" → "hey", "bangeeet" → "banget")
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)

        # 3. Normalize slang word by word (case-insensitive, preserve punctuation)
        words = text.split()
        normalized = []
        for word in words:
            # Separate trailing punctuation
            punct = ""
            while word and word[-1] in ".,!?;:":
                punct = word[-1] + punct
                word = word[:-1]
            lower = word.lower()
            replacement = _ID_SLANG.get(lower, word)
            normalized.append(replacement + punct)
        text = " ".join(normalized)

        # 4. Ensure sentence ends with punctuation
        if text and text[-1] not in ".!?":
            text += "."

        # 5. Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]

        # 6. Detect language
        lang = self._detect_lang(text)

        return text, lang

    def _detect_lang(self, text: str) -> str:
        words_lower = set(re.findall(r"\b\w+\b", text.lower()))
        id_hits = len(words_lower & _ID_KEYWORDS)
        # Simple heuristic: if > 2 Indonesian keyword hits → "id"
        if id_hits >= 2:
            return "id"
        # Mixed detection
        en_patterns = r"\b(the|is|are|was|were|have|has|do|does|can|will|would)\b"
        if re.search(en_patterns, text, re.IGNORECASE) and id_hits > 0:
            return "mixed"
        if id_hits > 0:
            return "id"
        return "en"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SHORT-TERM MEMORY (Rolling Window + Auto-Summary)
# ═══════════════════════════════════════════════════════════════════════════════

class ShortTermMemory:
    """
    Modul 2: Memori jangka pendek per sesi.
    - Menyimpan N giliran terakhir secara penuh
    - Jika riwayat > max_turns, otomatis ringkas menjadi satu baris summary
    - Format output: string siap pakai untuk prompt
    """

    def __init__(self, max_full_turns: int = 8, summary_turns: int = 4):
        self.max_full_turns  = max_full_turns   # giliran yang dibawa penuh
        self.summary_turns   = summary_turns    # giliran sebelumnya yang diringkas
        # store: {session_id → [Turn, ...]}
        self._store: dict[str, list[Turn]] = {}

    def add_turn(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(Turn(role=role, content=content))

    def get_recent(self, session_id: str) -> list[Turn]:
        """Return the last max_full_turns turns."""
        turns = self._store.get(session_id, [])
        return turns[-self.max_full_turns:]

    def get_summary(self, session_id: str) -> str:
        """
        Return a brief summary of turns older than max_full_turns.
        Uses extractive summarization (first sentence from each assistant turn).
        """
        turns = self._store.get(session_id, [])
        older = turns[: max(0, len(turns) - self.max_full_turns)]
        if not older:
            return ""
        # Extractive: pick first sentence of each assistant turn
        summary_parts = []
        for t in older[-self.summary_turns:]:
            if t.role == "assistant":
                first_sent = re.split(r"[.!?]", t.content)[0].strip()
                if first_sent:
                    summary_parts.append(first_sent)
        if summary_parts:
            return "[Earlier context: " + "; ".join(summary_parts) + "]"
        return ""

    def format_history(self, session_id: str) -> str:
        """Format recent turns as a prompt-ready string."""
        summary = self.get_summary(session_id)
        recent  = self.get_recent(session_id)

        lines = []
        if summary:
            lines.append(summary)
        for t in recent:
            role_label = "User" if t.role == "user" else "Assistant"
            lines.append(f"{role_label}: {t.content}")
        return "\n".join(lines)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    def load_from_messages(self, session_id: str, messages: list[dict]) -> None:
        """Sync from messages array received from client."""
        self._store[session_id] = [
            Turn(role=m["role"], content=m.get("content", ""))
            for m in messages
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. USER FACT MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class UserFactMemory:
    """
    Modul 3: Memori fakta pengguna per sesi (nama, preferensi, konteks proyek).
    Pengguna dapat menghapus atau mengedit via API.
    """

    # Pola untuk ekstraksi fakta otomatis dari kalimat pengguna
    _FACT_PATTERNS: list[tuple[str, str]] = [
        (r"\bsaya (?:bernama|nama saya|dipanggil|namaku)\s+([A-Za-z]+)", "name"),
        (r"\baku (?:bernama|dipanggil)\s+([A-Za-z]+)", "name"),
        (r"\bpanggil (?:aku|saya)\s+([A-Za-z]+)", "name"),
        (r"\bsaya (?:suka|prefer|lebih suka)\s+(.+?)(?:\.|$)", "preference"),
        (r"\bproyek saya\s+(?:adalah|itu|ini)?\s*(.+?)(?:\.|$)", "project"),
        (r"\bsaya (?:bekerja|kerja) (?:di|sebagai|pada)\s+(.+?)(?:\.|$)", "work"),
    ]

    def __init__(self):
        self._facts: dict[str, dict[str, str]] = {}  # {session_id → {key → value}}

    def extract_and_store(self, session_id: str, text: str) -> dict[str, str]:
        """Auto-extract facts from user message. Returns newly found facts."""
        found: dict[str, str] = {}
        for pattern, key in self._FACT_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                value = m.group(1).strip().rstrip(".")
                found[key] = value
        if found:
            if session_id not in self._facts:
                self._facts[session_id] = {}
            self._facts[session_id].update(found)
        return found

    def get(self, session_id: str) -> dict[str, str]:
        return self._facts.get(session_id, {})

    def set_fact(self, session_id: str, key: str, value: str) -> None:
        if session_id not in self._facts:
            self._facts[session_id] = {}
        self._facts[session_id][key] = value

    def delete_fact(self, session_id: str, key: str) -> bool:
        if session_id in self._facts and key in self._facts[session_id]:
            del self._facts[session_id][key]
            return True
        return False

    def format_for_prompt(self, session_id: str) -> str:
        facts = self.get(session_id)
        if not facts:
            return ""
        parts = []
        if "name" in facts:
            parts.append(f"User's name: {facts['name']}")
        if "preference" in facts:
            parts.append(f"User prefers: {facts['preference']}")
        if "project" in facts:
            parts.append(f"User's project: {facts['project']}")
        if "work" in facts:
            parts.append(f"User works at/as: {facts['work']}")
        return "[User facts: " + " | ".join(parts) + "]"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. INTENT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class IntentClassifier:
    """
    Modul 4: Klasifikasi maksud pengguna (rule-based + keyword matching).
    Untuk produksi: ganti dengan model klasifikasi ringan (e.g., fasttext).
    """

    # (pattern, intent, base_confidence)
    _RULES: list[tuple[str, Intent, float]] = [
        # Question patterns (Indonesia + English)
        (r"^(apa|siapa|kapan|dimana|kenapa|mengapa|bagaimana|berapa|apakah|bisakah|bolehkah|dapatkah)",
         Intent.QUESTION, 0.90),
        (r"^(what|who|when|where|why|how|which|is|are|can|could|would|should)\b",
         Intent.QUESTION, 0.88),
        (r"\?$", Intent.QUESTION, 0.75),

        # Command patterns
        (r"^(tolong|mohon|coba|buatkan|buat|tuliskan|tulis|jelaskan|rangkum|ringkas|hitung|cari|carikan|terjemahkan|translate|generate|make|create|write|explain|summarize|calculate|find)",
         Intent.COMMAND, 0.90),

        # Continuation patterns
        (r"^(lanjutkan|lanjut|teruskan|dan kemudian|lalu|selanjutnya|next|continue|then|also|besides)",
         Intent.CONTINUATION, 0.85),
        (r"\b(tadi|sebelumnya|barusan|yang itu|yang tadi|itu tadi|earlier|before|previously|that one)\b",
         Intent.CONTINUATION, 0.75),

        # Clarification
        (r"^(maksudnya|artinya|maksud saya|yang saya maksud|i mean|meaning|clarify|what do you mean)",
         Intent.CLARIFICATION, 0.85),

        # Chitchat
        (r"^(halo|hai|hi|hello|selamat|good morning|good evening|apa kabar|how are you|hey)\b",
         Intent.CHITCHAT, 0.90),
        (r"^(terima kasih|thanks|makasih|thx|ok|oke|baik|sip|mantap|bagus|keren|cool|great)",
         Intent.CHITCHAT, 0.75),

        # Feedback
        (r"\b(jawaban kamu|jawabanmu|respons kamu|you were wrong|kamu salah|kurang tepat|tidak benar|not right|incorrect)\b",
         Intent.FEEDBACK, 0.80),

        # Out of scope
        (r"\b(hack|hacking|exploit|crack|password|illegal|ilegal|narkoba|senjata|bomb|bom)\b",
         Intent.OUT_OF_SCOPE, 0.95),
    ]

    def classify(self, text: str) -> tuple[Intent, float]:
        """Returns (intent, confidence). Confidence in [0.0, 1.0]."""
        text_lower = text.lower().strip()
        best_intent     = Intent.UNKNOWN
        best_confidence = 0.0

        for pattern, intent, base_conf in self._RULES:
            m = re.search(pattern, text_lower, re.IGNORECASE)
            if m:
                # Boost confidence if pattern matches at start of sentence
                at_start = m.start() == 0
                confidence = base_conf if at_start else base_conf * 0.85
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_intent = intent

        # Default: if nothing matched with confidence, guess question or chitchat
        if best_intent == Intent.UNKNOWN:
            if len(text.split()) <= 3:
                best_intent, best_confidence = Intent.CHITCHAT, 0.50
            else:
                best_intent, best_confidence = Intent.QUESTION, 0.45

        return best_intent, round(best_confidence, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REFERENCE RESOLVER
# ═══════════════════════════════════════════════════════════════════════════════

class ReferenceResolver:
    """
    Modul 5: Ubah kata ganti demonstratif dan rujukan ambigu menjadi entitas eksplisit.

    Contoh:
      History: "User: jelaskan Flash Attention"
               "Assistant: Flash Attention adalah..."
      Input:   "bagaimana cara kerjanya?"
      Output:  "bagaimana cara kerja Flash Attention?"
    """

    # Pola kata rujukan yang perlu di-resolve
    _REF_PATTERNS = [
        r"\b(itu|ini|tersebut|yang dimaksud|yang tadi|yang barusan|hal itu|hal ini)\b",
        r"\b(it|that|this|the above|the mentioned|the one|those|these)\b",
        r"\b(kerjanya|caranya|fungsinya|bedanya|contohnya|maksudnya)\b",
    ]

    def resolve(self, current_input: str, recent_turns: list[Turn]) -> str:
        """
        Jika kalimat mengandung rujukan, coba ganti dengan entitas dari giliran terakhir.
        Strategi sederhana: ekstrak noun/topik dari N kalimat terakhir asisten.
        """
        if not recent_turns:
            return current_input

        # Check if current input contains any reference patterns
        has_ref = any(
            re.search(pat, current_input, re.IGNORECASE)
            for pat in self._REF_PATTERNS
        )

        # Also check for short continuation queries (likely referring to prev topic)
        is_short_query = len(current_input.split()) <= 5

        if not (has_ref or is_short_query):
            return current_input

        # Extract the most recent topic from last assistant turn
        last_topic = self._extract_last_topic(recent_turns)
        if not last_topic:
            return current_input

        # If the input is very short and lacks a subject, prepend topic
        if is_short_query and not has_ref:
            # e.g. "bagaimana cara kerjanya?" → "bagaimana cara kerja [Flash Attention]?"
            return current_input  # Don't auto-inject for short queries without explicit ref

        # Replace demonstrative reference with explicit topic
        resolved = current_input
        for pat in self._REF_PATTERNS:
            resolved = re.sub(pat, last_topic, resolved, flags=re.IGNORECASE)

        # If resolved == original (pattern didn't match directly), append topic as context
        if resolved == current_input and has_ref:
            resolved = f"{current_input} (tentang {last_topic})"

        return resolved

    def _extract_last_topic(self, turns: list[Turn]) -> str:
        """
        Extract the main topic/noun phrase from recent turns.
        Looks at recent user queries for capitalized nouns or technical terms.
        """
        # Look backward through turns
        for turn in reversed(turns):
            if turn.role == "user":
                text = turn.content
                # Extract capitalized words (likely proper nouns / technical terms)
                caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
                if caps:
                    return caps[-1]  # last capitalized term
                # Extract words after common query starters
                m = re.search(
                    r"(?:jelaskan|apa itu|tentang|soal|mengenai|about|explain|what is)\s+(.+?)(?:\?|$)",
                    text, re.IGNORECASE
                )
                if m:
                    return m.group(1).strip().rstrip(".")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ANSWER PLANNER
# ═══════════════════════════════════════════════════════════════════════════════

class AnswerPlanner:
    """
    Modul 6: Putuskan strategi jawaban.

    Keputusan:
      - ANSWER_DIRECT     : langsung jawab
      - CLARIFY           : tanya balik karena ambigu
      - ACKNOWLEDGE_LIMIT : akui tidak tahu / di luar cakupan
    """

    class Decision(str, Enum):
        ANSWER_DIRECT     = "answer_direct"
        CLARIFY           = "clarify"
        ACKNOWLEDGE_LIMIT = "acknowledge_limit"

    def plan(
        self,
        intent: Intent,
        confidence: float,
        resolved_input: str,
        has_knowledge: bool
    ) -> tuple["AnswerPlanner.Decision", str]:
        """
        Returns (decision, clarification_question_if_any).
        """
        # Out of scope → acknowledge limit
        if intent == Intent.OUT_OF_SCOPE:
            return self.Decision.ACKNOWLEDGE_LIMIT, ""

        # Very low confidence → ask for clarification
        if confidence < 0.45 and len(resolved_input.split()) < 4:
            return self.Decision.CLARIFY, self._generate_clarification(resolved_input)

        # Unknown intent + no knowledge hit + very short input
        if intent == Intent.UNKNOWN and not has_knowledge and len(resolved_input.split()) <= 3:
            return self.Decision.CLARIFY, "Bisa kamu jelaskan lebih detail apa yang kamu maksud?"

        # Default: answer directly
        return self.Decision.ANSWER_DIRECT, ""

    def _generate_clarification(self, text: str) -> str:
        """Generate a context-appropriate clarification question."""
        starters = [
            "Maaf, bisa lebih spesifik?",
            "Boleh saya tahu lebih detail tentang pertanyaanmu?",
            "Apakah maksudmu",
        ]
        # Pick based on text length
        if len(text.split()) <= 2:
            return f"Bisa kamu ceritakan lebih lengkap tentang '{text}'?"
        return starters[1]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. PROMPT BUILDER (Terpusat, Terversi)
# ═══════════════════════════════════════════════════════════════════════════════

# Persona dan aturan sistem — satu tempat, versi terkontrol
SYSTEM_PERSONA_V1 = """You are NeuralForge Assistant, a helpful, honest, and concise AI.
Rules:
- Answer in the same language the user uses (Indonesian or English).
- Be factual. If you are unsure, say so clearly — do not invent facts.
- If knowledge context is provided, base your answer on it and cite the source.
- Keep answers focused and relevant to the question asked.
- Do not repeat the question back verbatim.
- If the user makes a factual error, politely correct it.
"""

PROMPT_VERSIONS: dict[str, str] = {
    "v1.0": SYSTEM_PERSONA_V1,
}
ACTIVE_PROMPT_VERSION = "v1.0"


class PromptBuilder:
    """
    Modul 7: Bangun prompt final yang terstruktur dan konsisten.

    Struktur prompt:
      [SYSTEM PERSONA]
      [USER FACTS]
      [EPISODIC MEMORY]
      [KNOWLEDGE CONTEXT + SOURCES]
      [CONVERSATION HISTORY]
      [CURRENT USER QUERY]
      Assistant:
    """

    def __init__(self, version: str = ACTIVE_PROMPT_VERSION):
        self.version = version
        self.persona = PROMPT_VERSIONS.get(version, SYSTEM_PERSONA_V1)

    def build(
        self,
        resolved_input: str,
        history_str: str,
        user_facts_str: str = "",
        rag_context: str = "",
        episodic_ctx: str = "",
        decision: str = "answer_direct",
        clarification_q: str = "",
    ) -> str:
        parts: list[str] = []

        # 1. Persona (always present)
        parts.append(self.persona.strip())

        # 2. User facts (if any)
        if user_facts_str:
            parts.append(user_facts_str)

        # 3. Episodic memory (past session summaries)
        if episodic_ctx:
            parts.append(episodic_ctx)

        # 4. Knowledge context
        if rag_context:
            parts.append(rag_context)

        # 5. Conversation history
        if history_str:
            parts.append(history_str)

        # 6. Current turn
        if decision == AnswerPlanner.Decision.CLARIFY and clarification_q:
            # Override: prompt AI to ask the clarification question
            parts.append(f"User: {resolved_input}")
            parts.append(f"Assistant: {clarification_q}")
        elif decision == AnswerPlanner.Decision.ACKNOWLEDGE_LIMIT:
            parts.append(f"User: {resolved_input}")
            parts.append("Assistant: Maaf, pertanyaan itu di luar cakupan yang dapat saya bantu.")
        else:
            parts.append(f"User: {resolved_input}")
            parts.append("Assistant:")

        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ANSWER VERIFIER
# ═══════════════════════════════════════════════════════════════════════════════

class AnswerVerifier:
    """
    Modul 8: Verifikasi jawaban AI sebelum dikirim ke pengguna.

    Cek:
      (a) Relevance  — apakah menjawab pertanyaan?
      (b) Consistency — apakah konsisten dengan konteks?
      (c) Source support — jika ada RAG, apakah didukung sumber?
    """

    def verify(
        self,
        question: str,
        answer: str,
        rag_context: str = "",
        history_str: str = "",
    ) -> tuple[bool, float]:
        """
        Returns (is_valid, score_0_to_1).
        """
        score = 1.0

        if not answer or len(answer.strip()) < 5:
            return False, 0.0

        # (a) Relevance: check keyword overlap between question and answer
        q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
        a_words = set(re.findall(r"\b\w{3,}\b", answer.lower()))
        overlap = len(q_words & a_words)
        if overlap == 0 and len(q_words) > 2:
            score -= 0.3  # Penalize no keyword overlap

        # (b) Consistency: answer should not contradict a "no" from previous turns
        if history_str:
            if "tidak" in history_str.lower() and "tidak" not in answer.lower():
                # Weak check — could indicate flip-flop
                score -= 0.1

        # (c) Source support: if RAG context provided but answer doesn't use any of its words
        if rag_context:
            ctx_words = set(re.findall(r"\b\w{4,}\b", rag_context.lower()))
            a_words_long = set(re.findall(r"\b\w{4,}\b", answer.lower()))
            if ctx_words and not (ctx_words & a_words_long):
                score -= 0.2  # Possibly hallucinating

        # (d) Detect common hallucination markers
        hallucination_markers = [
            r"pada tahun \d{4}.*?ditemukan oleh",
            r"menurut penelitian.*?\d{4}",
        ]
        for marker in hallucination_markers:
            if re.search(marker, answer, re.IGNORECASE) and not rag_context:
                score -= 0.3
                break

        score = max(0.0, min(1.0, score))
        is_valid = score >= 0.5

        return is_valid, round(score, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CONVERSATION PIPELINE (Orchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

class ConversationPipeline:
    """
    Orkestrator utama: jalankan seluruh pipeline per pesan.

    Penggunaan:
        pipeline = ConversationPipeline()
        context  = pipeline.process(
            session_id="abc123",
            raw_input="bagaimana cara kerjanya?",
            messages=[...],
            rag_context="...",
            rag_sources=[...],
            episodic_context="..."
        )
        # context.prompt siap dipakai untuk generate
        # Setelah generate: pipeline.finalize(session_id, context, answer)
    """

    def __init__(self):
        self.normalizer  = InputNormalizer()
        self.memory      = ShortTermMemory(max_full_turns=8, summary_turns=4)
        self.user_facts  = UserFactMemory()
        self.classifier  = IntentClassifier()
        self.resolver    = ReferenceResolver()
        self.planner     = AnswerPlanner()
        self.builder     = PromptBuilder(version=ACTIVE_PROMPT_VERSION)
        self.verifier    = AnswerVerifier()

    def process(
        self,
        session_id: str,
        raw_input: str,
        messages: list[dict],
        rag_context: str = "",
        rag_sources: list[str] = None,
        episodic_context: str = "",
    ) -> PipelineContext:
        """
        Jalankan pipeline lengkap. Returns PipelineContext.
        Semua langkah dicatat dalam context.steps_log.
        """
        t0      = time.time()
        conv_id = str(uuid.uuid4())[:8]
        ctx     = PipelineContext(conv_id=conv_id, raw_input=raw_input)
        ctx.rag_context = rag_context
        ctx.rag_sources = rag_sources or []

        # ── Sync memory from client messages ─────────────────────────────────
        self.memory.load_from_messages(session_id, messages)

        # ── Step 1: Normalize ─────────────────────────────────────────────────
        norm, lang         = self.normalizer.normalize(raw_input)
        ctx.normalized_input = norm
        ctx.detected_lang    = lang
        _log_step(conv_id, "normalize", {"raw": raw_input, "norm": norm, "lang": lang})

        # ── Step 2: Extract user facts ────────────────────────────────────────
        new_facts = self.user_facts.extract_and_store(session_id, norm)
        ctx.user_facts = self.user_facts.get(session_id)
        if new_facts:
            _log_step(conv_id, "user_facts_extracted", {"new": new_facts})

        # ── Step 3: Classify intent ───────────────────────────────────────────
        intent, conf         = self.classifier.classify(norm)
        ctx.intent           = intent
        ctx.intent_confidence = conf
        _log_step(conv_id, "intent", {"intent": intent.value, "confidence": conf})

        # ── Step 4: Resolve references ────────────────────────────────────────
        recent = self.memory.get_recent(session_id)
        resolved          = self.resolver.resolve(norm, recent)
        ctx.resolved_input = resolved
        _log_step(conv_id, "reference_resolve",
                  {"original": norm, "resolved": resolved,
                   "changed": resolved != norm})

        # ── Step 5: Plan answer strategy ──────────────────────────────────────
        has_knowledge = bool(rag_context.strip())
        decision, clarify_q = self.planner.plan(intent, conf, resolved, has_knowledge)
        ctx.should_clarify       = (decision == AnswerPlanner.Decision.CLARIFY)
        ctx.clarification_question = clarify_q
        _log_step(conv_id, "plan", {"decision": decision.value, "clarify_q": clarify_q})

        # ── Step 6: Build prompt ──────────────────────────────────────────────
        history_str   = self.memory.format_history(session_id)
        user_facts_str = self.user_facts.format_for_prompt(session_id)

        prompt = self.builder.build(
            resolved_input=resolved,
            history_str=history_str,
            user_facts_str=user_facts_str,
            rag_context=rag_context,
            episodic_ctx=episodic_context,
            decision=decision,
            clarification_q=clarify_q,
        )
        ctx.prompt         = prompt
        ctx.prompt_version = self.builder.version
        _log_step(conv_id, "prompt_built",
                  {"version": ctx.prompt_version,
                   "prompt_len": len(prompt),
                   "has_rag": has_knowledge,
                   "decision": decision.value})

        ctx.latency_ms = round((time.time() - t0) * 1000, 1)
        return ctx

    def finalize(
        self,
        session_id: str,
        ctx: PipelineContext,
        answer: str,
    ) -> PipelineContext:
        """
        Dipanggil setelah generasi. Verifikasi jawaban dan simpan ke memori.
        """
        ctx.answer = answer

        # ── Step 7: Verify answer ─────────────────────────────────────────────
        is_valid, score = self.verifier.verify(
            question=ctx.resolved_input,
            answer=answer,
            rag_context=ctx.rag_context,
            history_str=self.memory.format_history(session_id),
        )
        ctx.answer_verified    = is_valid
        ctx.verification_score = score

        _log_step(ctx.conv_id, "verify",
                  {"valid": is_valid, "score": score, "answer_len": len(answer)})

        # ── Step 8: Store turns in memory ─────────────────────────────────────
        self.memory.add_turn(session_id, "user",      ctx.resolved_input)
        self.memory.add_turn(session_id, "assistant", answer)

        return ctx

    def clear_session(self, session_id: str) -> None:
        self.memory.clear(session_id)

    def get_session_facts(self, session_id: str) -> dict:
        return self.user_facts.get(session_id)

    def delete_user_fact(self, session_id: str, key: str) -> bool:
        return self.user_facts.delete_fact(session_id, key)
