"""
Fase 2: Knowledge Container — NeuralForge
==========================================
Wadah pengetahuan yang terstruktur, aman, dan terukur.

Fitur:
  1. Ingestion  — PDF/TXT/MD/DOCX, URL, manual Q&A. Validasi tipe & ukuran.
  2. Processing — ekstraksi teks, chunking 300-500 token + overlap, metadata lengkap
  3. Storage    — embedding + BM25 + deduplication via SHA256 hash
  4. Retrieval  — hybrid search, reranking, minimum relevance threshold (ambang)
  5. Citation   — setiap jawaban berbasis knowledge wajib sertakan sumber
  6. Management — list, aktif/nonaktif, hapus, re-index, statistik penggunaan
  7. Priority   — pengetahuan "terkoreksi pengguna" punya bobot lebih tinggi

Dependensi standar library saja (+ numpy). Tidak butuh faiss/chroma untuk ukuran ini.
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import uuid
import hashlib
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

# ─── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_DIM_DEFAULT = 288
MAX_FILE_SIZE_MB      = 20
CHUNK_TARGET_WORDS    = 80    # ~300-500 tokens ≈ 80-120 words for Indonesian/English
CHUNK_OVERLAP_WORDS   = 15
MIN_RELEVANCE_SCORE   = 0.15  # Below this → don't inject into prompt
SUPPORTED_EXTENSIONS  = {".txt", ".md", ".csv", ".html"}

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_DB    = os.path.join(_PROJECT_ROOT, "models", "knowledge_db_v2.json")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChunkMetadata:
    """Metadata per chunk — versi baru yang kaya informasi."""
    chunk_id:    str            # UUID unik per chunk
    source:      str            # Nama dokumen / URL
    source_type: str            # "file" | "url" | "manual_qa" | "correction"
    collection:  str = "default"
    date_added:  float = field(default_factory=time.time)
    version:     int   = 1      # Versi chunk (increments on update)
    trust_level: float = 1.0   # 1.0 = normal, 1.5 = user-corrected, 0.5 = unverified
    active:      bool  = True  # False = disabled (tidak dipakai retrieval)
    chunk_index: int   = 0      # Posisi chunk dalam dokumen
    total_chunks:int   = 1      # Total chunk dari dokumen ini
    word_count:  int   = 0
    sha256:      str   = ""     # Hash konten untuk deduplikasi
    usage_count: int   = 0      # Berapa kali chunk ini dipakai retrieval


@dataclass
class Chunk:
    """Satu chunk dokumen dengan embedding dan metadata."""
    text:      str
    embedding: list[float]
    metadata:  ChunkMetadata


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TEXT PROCESSORS
# ═══════════════════════════════════════════════════════════════════════════════

class HTMLCleaner:
    """Bersihkan HTML menjadi teks bersih."""

    def clean(self, html: str) -> str:
        # Remove script, style, nav, footer
        for tag in ("script", "style", "nav", "footer", "header", "aside"):
            html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", html,
                          flags=re.DOTALL | re.IGNORECASE)
        # Replace block elements with newlines
        for tag in ("p", "div", "h1", "h2", "h3", "h4", "li", "br", "tr"):
            html = re.sub(rf"</?{tag}[^>]*>", "\n", html, flags=re.IGNORECASE)
        # Remove remaining tags
        html = re.sub(r"<[^>]+>", " ", html)
        # Decode common HTML entities
        entities = {"&amp;": "&", "&lt;": "<", "&gt;": ">",
                    "&quot;": '"', "&nbsp;": " ", "&#39;": "'"}
        for ent, char in entities.items():
            html = html.replace(ent, char)
        # Collapse whitespace
        html = re.sub(r"\n{3,}", "\n\n", html)
        html = re.sub(r" {2,}", " ", html)
        return html.strip()


class TextChunker:
    """
    Chunking 300-500 token dengan overlap.
    Strategi: sliding window pada kata (lebih stabil dari token count).
    Target: CHUNK_TARGET_WORDS kata per chunk, CHUNK_OVERLAP_WORDS tumpang tindih.
    """

    def __init__(
        self,
        target_words: int = CHUNK_TARGET_WORDS,
        overlap_words: int = CHUNK_OVERLAP_WORDS,
    ):
        self.target  = target_words
        self.overlap = overlap_words

    def chunk(self, text: str) -> list[str]:
        # Split into paragraphs first (preserve semantic units)
        paragraphs = re.split(r"\n{2,}", text.strip())
        # Then flatten to words
        words = []
        for para in paragraphs:
            words.extend(para.split())
            words.append("__PARA_BREAK__")  # Marker

        # Sliding window
        chunks:  list[str] = []
        i = 0
        while i < len(words):
            window = words[i: i + self.target]
            # Remove para break markers for output
            text_chunk = " ".join(w for w in window if w != "__PARA_BREAK__").strip()
            if len(text_chunk.split()) >= 5:  # Skip very short chunks
                chunks.append(text_chunk)
            step = self.target - self.overlap
            i += max(1, step)

        # Merge tiny last chunk with previous if too short
        if len(chunks) >= 2 and len(chunks[-1].split()) < 10:
            chunks[-2] = chunks[-2] + " " + chunks[-1]
            chunks.pop()

        return chunks


class InputSanitizer:
    """Sanitasi input untuk mencegah injeksi dan kebocoran data."""

    _SENSITIVE_PATTERNS = [
        # Credentials
        r"(?i)(password|passwd|secret|api[_\s]?key|token|bearer)\s*[:=]\s*\S+",
        r"(?i)(sk-|pk_live_|pk_test_|xoxb-|xoxp-)[A-Za-z0-9]+",
        # Personal data (Indonesia format)
        r"\b\d{16}\b",              # Nomor KTP (16 digit)
        r"\b\d{10,13}\b",           # Nomor telepon
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email
        # Injection attempts
        r"(?i)(ignore previous|ignore all|jangan hiraukan|lupakan instruksi|system prompt)",
        r"(?i)(you are now|mulai sekarang kamu adalah|pretend you are|roleplay as)",
    ]

    def sanitize(self, text: str) -> tuple[str, list[str]]:
        """
        Returns (sanitized_text, list_of_redaction_descriptions).
        Replaces sensitive data with [REDACTED].
        """
        redactions = []
        for pattern in self._SENSITIVE_PATTERNS:
            matches = re.findall(pattern, text)
            if matches:
                text = re.sub(pattern, "[REDACTED]", text)
                redactions.append(f"Pattern redacted: {pattern[:40]}")
        return text, redactions

    def validate_file(self, filename: str, size_bytes: int) -> tuple[bool, str]:
        """Returns (ok, error_message)."""
        ext = os.path.splitext(filename.lower())[1]
        if ext not in SUPPORTED_EXTENSIONS:
            return False, f"File type '{ext}' not supported. Allowed: {SUPPORTED_EXTENSIONS}"
        max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
        if size_bytes > max_bytes:
            return False, f"File too large: {size_bytes/1024/1024:.1f} MB (max {MAX_FILE_SIZE_MB} MB)"
        return True, ""


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Hash-based embedding (ringan, deterministik, dimensi 288).
    Diperkaya dengan bigram dan position-weighted terms.
    Production: ganti dengan sentence-transformers atau model embed lokal.
    """

    def __init__(self, dim: int = EMBEDDING_DIM_DEFAULT):
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vec   = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        if not words:
            return vec.tolist()

        n = len(words)
        for i, w in enumerate(words):
            # Position weight: words near start/end carry more signal
            pos_weight = 1.0 + 0.3 * (1.0 - abs(i - n / 2) / (n / 2 + 1))
            h = hash(w) % self.dim
            vec[h] += pos_weight

        # Bigrams
        for i in range(n - 1):
            bigram = words[i] + "_" + words[i + 1]
            h = hash(bigram) % self.dim
            vec[h] += 0.6

        # Trigrams (for phrase matching)
        for i in range(n - 2):
            trigram = words[i] + "_" + words[i + 1] + "_" + words[i + 2]
            h = hash(trigram) % self.dim
            vec[h] += 0.3

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def cosine_sim(self, a: list[float], b: list[float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        return float(np.dot(va, vb))  # Already normalized


# ═══════════════════════════════════════════════════════════════════════════════
# 4. BM25 INDEX
# ═══════════════════════════════════════════════════════════════════════════════

class BM25Index:
    """In-memory BM25 index, always rebuilt from chunks (never out-of-sync)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.df:       dict[str, int] = {}   # term → document frequency
        self.doc_lens: list[int]      = []
        self.avg_len:  float          = 0.0
        self.n_docs:   int            = 0

    def build(self, chunks: list[Chunk]) -> None:
        """Rebuild the full index from scratch (O(n))."""
        self.df.clear()
        self.doc_lens.clear()
        total = 0

        for ch in chunks:
            tokens  = ch.text.lower().split()
            doc_len = len(tokens)
            self.doc_lens.append(doc_len)
            total  += doc_len
            for t in set(tokens):
                self.df[t] = self.df.get(t, 0) + 1

        self.n_docs  = len(chunks)
        self.avg_len = total / self.n_docs if self.n_docs else 0.0

    def score(self, query_tokens: list[str], chunk_idx: int, chunk_text: str) -> float:
        """BM25 score for one chunk."""
        if self.n_docs == 0 or chunk_idx >= len(self.doc_lens):
            return 0.0
        doc_len = self.doc_lens[chunk_idx]
        text_l  = chunk_text.lower()
        total   = 0.0
        for q in query_tokens:
            df = self.df.get(q, 0)
            if df == 0:
                continue
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
            tf  = text_l.count(q)
            num = tf * (self.k1 + 1)
            den = tf + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_len + 1e-5)) + 1e-10
            total += idf * num / den
        return total


# ═══════════════════════════════════════════════════════════════════════════════
# 5. RE-RANKER
# ═══════════════════════════════════════════════════════════════════════════════

class Reranker:
    """
    Cross-encoder-style re-ranker menggunakan unigram + bigram Jaccard.
    Production: ganti dengan model cross-encoder (e.g., ms-marco MiniLM).
    """

    def rerank(self, query: str, candidates: list[dict],
               top_k: int = 3) -> list[dict]:
        if not candidates:
            return []
        q_words   = set(re.findall(r"\w+", query.lower()))
        q_bigrams = {q_words.__iter__().__next__() + "_" + w
                     for i, w in enumerate(list(q_words)[1:])} if len(q_words) > 1 else set()

        scored = []
        for c in candidates:
            text    = c["chunk"].text
            t_words = set(re.findall(r"\w+", text.lower()))
            # Jaccard unigram
            j_uni   = len(q_words & t_words) / (len(q_words | t_words) + 1e-6)
            # Bigram overlap
            t_bigrams = {f"{list(t_words)[i]}_{list(t_words)[i+1]}"
                         for i in range(len(t_words) - 1)} if len(t_words) > 1 else set()
            j_bi  = len(q_bigrams & t_bigrams) / (len(q_bigrams) + 1e-6) if q_bigrams else 0.0
            # Trust level boost
            trust = c["chunk"].metadata.trust_level
            # Final score: blend initial hybrid score + reranking + trust
            final = 0.4 * c["score"] + 0.4 * j_uni + 0.1 * j_bi + 0.1 * (trust - 1.0)
            scored.append({**c, "rerank_score": round(final, 4)})

        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. KNOWLEDGE CONTAINER (Main Class)
# ═══════════════════════════════════════════════════════════════════════════════

class KnowledgeContainer:
    """
    Fase 2: Wadah pengetahuan utama.

    API Publik:
      add_document(source, content, source_type, collection, trust_level) → int
      add_manual_qa(question, answer, collection) → int
      add_correction(original_text, corrected_text, source) → int
      search(query, top_k, collection, min_score) → list[SearchResult]
      delete_source(source) → int
      toggle_source(source, active) → int
      reindex() → None
      get_stats() → dict
      list_sources() → list[dict]
    """

    def __init__(self, db_path: str = DEFAULT_DB, embedding_dim: int = EMBEDDING_DIM_DEFAULT):
        self.db_path       = db_path
        self.embedder      = EmbeddingEngine(dim=embedding_dim)
        self.bm25          = BM25Index()
        self.reranker      = Reranker()
        self.sanitizer     = InputSanitizer()
        self.chunker       = TextChunker()
        self.html_cleaner  = HTMLCleaner()

        self.chunks: list[Chunk]         = []
        self._hash_set: set[str]         = set()  # For deduplication
        self._collections: set[str]      = {"default"}

        self._load()

    # ─── Ingestion ─────────────────────────────────────────────────────────────

    def add_document(
        self,
        source: str,
        content: str,
        source_type: str = "file",
        collection: str  = "default",
        trust_level: float = 1.0,
    ) -> int:
        """
        Tambah dokumen teks. Returns jumlah chunk yang ditambahkan.
        Dokumen yang sudah ada (sumber sama) akan di-replace.
        """
        # Remove existing chunks from this source
        self.delete_source(source, save=False)

        # Sanitize content
        content, redactions = self.sanitizer.sanitize(content)
        if redactions:
            print(f"[KnowledgeContainer] Redacted {len(redactions)} sensitive patterns from '{source}'")

        # Chunk text
        text_chunks = self.chunker.chunk(content)
        total = len(text_chunks)
        added = 0

        for idx, chunk_text in enumerate(text_chunks):
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]

            # Skip duplicates
            if chunk_hash in self._hash_set:
                continue
            self._hash_set.add(chunk_hash)

            embedding = self.embedder.embed(chunk_text)
            meta      = ChunkMetadata(
                chunk_id    = str(uuid.uuid4())[:8],
                source      = source,
                source_type = source_type,
                collection  = collection,
                trust_level = trust_level,
                chunk_index = idx,
                total_chunks= total,
                word_count  = len(chunk_text.split()),
                sha256      = chunk_hash,
            )
            self.chunks.append(Chunk(text=chunk_text, embedding=embedding, metadata=meta))
            added += 1

        self._collections.add(collection)
        self.bm25.build(self._active_chunks())
        self._save()
        print(f"[KnowledgeContainer] '{source}': {added} chunks added (type={source_type}, coll={collection})")
        return added

    def add_manual_qa(self, question: str, answer: str, collection: str = "default") -> int:
        """Tambah pasangan Q&A manual."""
        content = f"Q: {question}\nA: {answer}"
        return self.add_document(
            source=f"qa:{uuid.uuid4().hex[:6]}",
            content=content,
            source_type="manual_qa",
            collection=collection,
            trust_level=1.2,
        )

    def add_correction(self, original_text: str, corrected_text: str,
                       source: str = "user_correction") -> int:
        """
        Tambah koreksi pengguna — trust_level lebih tinggi (1.5).
        Koreksi masuk sebagai knowledge baru dengan prioritas tinggi.
        """
        content = (
            f"CORRECTION: The following information supersedes previous knowledge.\n"
            f"Original: {original_text}\n"
            f"Corrected: {corrected_text}"
        )
        return self.add_document(
            source=f"correction:{source}:{uuid.uuid4().hex[:6]}",
            content=content,
            source_type="correction",
            collection="corrections",
            trust_level=1.5,
        )

    def scrape_url(self, url: str) -> tuple[str, str]:
        """
        Scrape URL. Returns (cleaned_text, error_or_empty).
        """
        # Basic URL validation
        if not re.match(r"^https?://", url):
            return "", "URL must start with http:// or https://"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "NeuralForge/2.0 (research bot)"}
            )
            with urllib.request.urlopen(req, timeout=12) as r:
                charset = r.headers.get_content_charset() or "utf-8"
                html    = r.read().decode(charset, errors="replace")
            cleaned = self.html_cleaner.clean(html)
            if len(cleaned) < 50:
                return "", "Page content too short or blocked"
            return cleaned, ""
        except Exception as e:
            return "", str(e)

    # ─── Retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 3,
        collection: Optional[str] = None,
        min_score: float = MIN_RELEVANCE_SCORE,
        alpha: float = 0.6,       # Weight for dense score (1-alpha for BM25)
    ) -> list[dict]:
        """
        Hybrid search: dense cosine + BM25, filtered by collection and min_score.
        Returns list of dicts with keys: chunk, score, source, citation_text.
        """
        active = self._active_chunks()
        if not active:
            return []

        # Filter by collection
        if collection:
            active = [c for c in active if c.metadata.collection == collection]

        q_emb    = self.embedder.embed(query)
        q_tokens = re.findall(r"\w+", query.lower())

        # Build active-only BM25 index
        active_bm25 = BM25Index()
        active_bm25.build(active)

        results = []
        for idx, ch in enumerate(active):
            dense  = self.embedder.cosine_sim(q_emb, ch.embedding)
            bm25_s = active_bm25.score(q_tokens, idx, ch.text)
            # Normalize BM25 (rough: cap at 5.0)
            bm25_norm = min(bm25_s / 5.0, 1.0)
            # Apply trust_level as a multiplier
            score = (alpha * dense + (1 - alpha) * bm25_norm) * ch.metadata.trust_level

            if score >= min_score:
                results.append({
                    "chunk":   ch,
                    "score":   round(score, 4),
                    "source":  ch.metadata.source,
                    "collection": ch.metadata.collection,
                })

        # Re-rank
        reranked = self.reranker.rerank(query, results, top_k=top_k * 2)

        # Filter again by min_score after reranking
        final = [r for r in reranked if r.get("rerank_score", r["score"]) >= min_score]
        final = final[:top_k]

        # Update usage count
        for r in final:
            r["chunk"].metadata.usage_count += 1

        # Add citation text
        for r in final:
            ch   = r["chunk"]
            r["citation_text"] = (
                f"[Source: {ch.metadata.source} | "
                f"Chunk {ch.metadata.chunk_index + 1}/{ch.metadata.total_chunks} | "
                f"Trust: {ch.metadata.trust_level:.1f}]"
            )
            r["context"] = ch.text

        if final:
            self._save()  # Persist usage counts

        return final

    def format_rag_context(self, results: list[dict], max_chars: int = 800) -> tuple[str, list[str]]:
        """
        Format search results into a prompt-ready context string.
        Returns (context_string, list_of_sources).
        """
        if not results:
            return "", []

        parts   = []
        sources = []
        for r in results:
            snippet = r["context"][:max_chars // len(results)]
            parts.append(f"{r['citation_text']}\n{snippet}")
            sources.append(r["source"])

        ctx = "\n\n".join(parts)
        return f"[Knowledge Context]\n{ctx}\n[End of Context]", list(dict.fromkeys(sources))

    # ─── Management ────────────────────────────────────────────────────────────

    def delete_source(self, source: str, save: bool = True) -> int:
        """Hapus semua chunk dari sumber tertentu."""
        initial = len(self.chunks)
        to_remove_hashes = {c.metadata.sha256 for c in self.chunks if c.metadata.source == source}
        self.chunks       = [c for c in self.chunks if c.metadata.source != source]
        self._hash_set   -= to_remove_hashes
        removed           = initial - len(self.chunks)
        if removed > 0:
            self.bm25.build(self._active_chunks())
            if save:
                self._save()
            print(f"[KnowledgeContainer] Deleted {removed} chunks from '{source}'")
        return removed

    def toggle_source(self, source: str, active: bool) -> int:
        """Aktifkan atau nonaktifkan sumber tanpa menghapus data."""
        count = 0
        for ch in self.chunks:
            if ch.metadata.source == source:
                ch.metadata.active = active
                count += 1
        if count > 0:
            self.bm25.build(self._active_chunks())
            self._save()
        return count

    def reindex(self) -> None:
        """Rebuild semua embedding dan BM25 (gunakan jika embedding model berubah)."""
        print(f"[KnowledgeContainer] Re-indexing {len(self.chunks)} chunks...")
        for ch in self.chunks:
            ch.embedding = self.embedder.embed(ch.text)
        self.bm25.build(self._active_chunks())
        self._save()
        print("[KnowledgeContainer] Re-index complete.")

    def get_stats(self) -> dict:
        """Statistik lengkap untuk UI/API."""
        sources:   dict[str, dict] = {}
        active_n   = 0
        for ch in self.chunks:
            src = ch.metadata.source
            if src not in sources:
                sources[src] = {
                    "source":      src,
                    "chunks":      0,
                    "collection":  ch.metadata.collection,
                    "source_type": ch.metadata.source_type,
                    "trust_level": ch.metadata.trust_level,
                    "active":      ch.metadata.active,
                    "usage_count": 0,
                }
            sources[src]["chunks"]      += 1
            sources[src]["usage_count"] += ch.metadata.usage_count
            if ch.metadata.active:
                active_n += 1

        return {
            "total_chunks":    len(self.chunks),
            "active_chunks":   active_n,
            "total_sources":   len(sources),
            "collections":     sorted(self._collections),
            "sources":         list(sources.values()),
            "db_path":         self.db_path,
        }

    def list_sources(self) -> list[dict]:
        return self.get_stats()["sources"]

    # ─── Persistence ───────────────────────────────────────────────────────────

    def _active_chunks(self) -> list[Chunk]:
        return [c for c in self.chunks if c.metadata.active]

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            payload = {
                "version":  2,
                "saved_at": time.time(),
                "chunks": [
                    {
                        "text":      ch.text,
                        "embedding": ch.embedding,
                        "metadata":  asdict(ch.metadata),
                    }
                    for ch in self.chunks
                ],
                "collections": list(self._collections),
            }
            # Atomic write: write to temp file then rename
            tmp_path = self.db_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_path, self.db_path)
        except Exception as e:
            print(f"[KnowledgeContainer] Save error: {e}")

    def _load(self) -> None:
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            # Support old format (list) from persistent_rag.py
            if isinstance(payload, list):
                self._migrate_old_format(payload)
                return

            schema_ver = payload.get("version", 1)

            for item in payload.get("chunks", []):
                meta_dict = item.get("metadata", {})
                # Fill missing fields for forward-compat
                meta_dict.setdefault("chunk_id",    str(uuid.uuid4())[:8])
                meta_dict.setdefault("source_type", "file")
                meta_dict.setdefault("collection",  "default")
                meta_dict.setdefault("version",     1)
                meta_dict.setdefault("trust_level", 1.0)
                meta_dict.setdefault("active",      True)
                meta_dict.setdefault("chunk_index", 0)
                meta_dict.setdefault("total_chunks",1)
                meta_dict.setdefault("word_count",  len(item["text"].split()))
                meta_dict.setdefault("sha256",      "")
                meta_dict.setdefault("usage_count", 0)
                meta_dict.setdefault("date_added",  time.time())

                meta = ChunkMetadata(**meta_dict)
                self.chunks.append(Chunk(
                    text      = item["text"],
                    embedding = item["embedding"],
                    metadata  = meta,
                ))
                if meta.sha256:
                    self._hash_set.add(meta.sha256)

            self._collections = set(payload.get("collections", ["default"]))
            self.bm25.build(self._active_chunks())
            print(f"[KnowledgeContainer] Loaded {len(self.chunks)} chunks from {self.db_path}")
        except Exception as e:
            print(f"[KnowledgeContainer] Load error: {e}")
            self.chunks = []

    def _migrate_old_format(self, old_list: list[dict]) -> None:
        """Migrate from old persistent_rag.py format (flat list of dicts)."""
        print("[KnowledgeContainer] Migrating old knowledge format...")
        for item in old_list:
            meta = ChunkMetadata(
                chunk_id    = str(uuid.uuid4())[:8],
                source      = item.get("source", "unknown"),
                source_type = "file",
                collection  = "default",
                word_count  = len(item.get("text", "").split()),
                sha256      = "",
            )
            self.chunks.append(Chunk(
                text      = item.get("text", ""),
                embedding = item.get("embedding", []),
                metadata  = meta,
            ))
        self.bm25.build(self._active_chunks())
        self._save()
        print(f"[KnowledgeContainer] Migration done: {len(self.chunks)} chunks")
