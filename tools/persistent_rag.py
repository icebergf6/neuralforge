"""
Advanced RAG Engine v2:
1. Semantic Chunking (potong berdasarkan makna, bukan kata)
2. Re-ranking (cross-encoder style scoring setelah retrieval)
3. Multi-hop RAG (chain query untuk pertanyaan kompleks)
4. Parent-Child Chunks (simpan kecil, inject besar)
5. Episodic Memory (compress riwayat percakapan lama ke summary)
"""

import os
import re
import json
import math
import hashlib
import urllib.request
import numpy as np

DB_FILE = "models/knowledge_db.json"


# ─── Semantic Chunker ──────────────────────────────────────────────────────────
class SemanticChunker:
    """
    Potong teks berdasarkan batas makna semantik:
    1. Gunakan kalimat sebagai unit dasar
    2. Gabungkan kalimat yang koherensi vektornya tinggi
    3. Potong saat ada "lompatan semantik" (cosine drop > threshold)
    """

    def __init__(self, embedding_fn, max_chunk_tokens=120, threshold=0.35):
        self.embedding_fn  = embedding_fn
        self.max_tokens    = max_chunk_tokens
        self.threshold     = threshold

    def split_sentences(self, text: str) -> list[str]:
        # Split on sentence boundaries: '.', '!', '?', '\n\n'
        sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def chunk(self, text: str) -> list[str]:
        sentences = self.split_sentences(text)
        if not sentences:
            return []

        # Compute embeddings for each sentence
        embeddings = [self.embedding_fn(s) for s in sentences]

        chunks = []
        current_sents = [sentences[0]]
        current_len   = len(sentences[0].split())

        for i in range(1, len(sentences)):
            # Check cosine similarity with previous sentence
            sim = float(np.dot(embeddings[i], embeddings[i - 1]))
            sent_len = len(sentences[i].split())

            # Continue chunk if similar enough AND not too long
            if sim >= self.threshold and current_len + sent_len <= self.max_tokens:
                current_sents.append(sentences[i])
                current_len += sent_len
            else:
                # Semantic boundary detected → finalize chunk
                chunks.append(" ".join(current_sents))
                current_sents = [sentences[i]]
                current_len   = sent_len

        if current_sents:
            chunks.append(" ".join(current_sents))

        return [c for c in chunks if len(c.strip()) > 10]


# ─── Cross-Encoder Re-ranker ───────────────────────────────────────────────────
class ReRanker:
    """
    Re-rank retrieved candidates using a more accurate scoring signal.
    Uses: keyword overlap + length normalization + position bias.
    (Production: replace with a small BERT cross-encoder for better accuracy)
    """

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        if not candidates:
            return []

        q_words  = set(re.findall(r'\w+', query.lower()))
        q_bigrams = set()
        q_list = list(q_words)
        for i in range(len(q_list) - 1):
            q_bigrams.add(q_list[i] + "_" + q_list[i + 1])

        scored = []
        for c in candidates:
            text   = c.get("text", "")
            t_words = re.findall(r'\w+', text.lower())
            t_set   = set(t_words)

            # Unigram Jaccard
            overlap = len(q_words & t_set)
            union   = len(q_words | t_set)
            jaccard = overlap / (union + 1e-6)

            # Bigram bonus
            t_bigrams = set()
            for i in range(len(t_words) - 1):
                t_bigrams.add(t_words[i] + "_" + t_words[i + 1])
            bigram_overlap = len(q_bigrams & t_bigrams) / (len(q_bigrams) + 1e-6)

            # Length normalization: prefer medium-length chunks
            words = len(t_words)
            length_penalty = 1.0 - abs(words - 60) / 200.0
            length_penalty = max(0.3, length_penalty)

            # Combined score
            score = 0.5 * jaccard + 0.3 * bigram_overlap + 0.2 * length_penalty
            scored.append((score, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]


# ─── Persistent Knowledge Hub v2 (Advanced RAG) ───────────────────────────────
class PersistentKnowledgeHub:
    """
    Hybrid Search + Semantic Chunking + Re-ranking + Parent-Child Chunks.
    """

    def __init__(self, embedding_dim=288, db_path=DB_FILE):
        self.embedding_dim = embedding_dim
        self.db_path       = db_path
        self.chunks        = []        # child chunks (small, for retrieval)
        self.parents       = {}        # parent chunks (large, for context injection)
        self.doc_lengths   = []
        self.avg_doc_len   = 0.0
        self.vocab_df      = {}

        self.reranker = ReRanker()
        self.chunker  = SemanticChunker(
            embedding_fn=self.compute_embedding,
            max_chunk_tokens=80,
            threshold=0.30
        )

        # Episodic memory: compressed conversation summaries
        self.episodic_memory: list[dict] = []

        self.load_from_disk()

    # ── HTML Cleaning ─────────────────────────────────────────────────────────
    def clean_html(self, html: str) -> str:
        cleaned = re.sub(r'<script.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<style.*?</style>',  '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    # ── Web Scraper ───────────────────────────────────────────────────────────
    def scrape_url(self, url: str) -> str:
        req = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                charset = r.headers.get_content_charset() or 'utf-8'
                html    = r.read().decode(charset, errors='replace')
                return self.clean_html(html)
        except Exception as e:
            return f"Error fetching URL: {e}"

    # ── Dense Embedding (hash-based, 288-dim) ────────────────────────────────
    def compute_embedding(self, text: str) -> np.ndarray:
        vec   = np.zeros(self.embedding_dim, dtype=np.float32)
        words = re.findall(r'\w+', text.lower())
        if not words:
            return vec
        # Token unigrams
        for w in words:
            vec[hash(w) % self.embedding_dim] += 1.0
        # Bigram enrichment
        for i in range(len(words) - 1):
            bigram = words[i] + "_" + words[i + 1]
            vec[hash(bigram) % self.embedding_dim] += 0.5
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    # ── Add Document (semantic chunking + parent-child) ───────────────────────
    def add_document(self, source_name: str, content: str) -> int:
        self.delete_document(source_name, auto_save=False)

        # Parent: full document (for context injection)
        parent_id = hashlib.md5(source_name.encode()).hexdigest()[:8]
        self.parents[parent_id] = {
            "source": source_name,
            "text":   content[:2000]   # truncate to 2000 chars max for parent
        }

        # Children: semantic chunks (for retrieval)
        child_chunks = self.chunker.chunk(content)
        start_id     = len(self.chunks)
        added        = 0
        for idx, ch in enumerate(child_chunks):
            emb = self.compute_embedding(ch)
            self.chunks.append({
                "id":        start_id + idx,
                "source":    source_name,
                "parent_id": parent_id,
                "text":      ch,
                "embedding": emb.tolist()
            })
            added += 1

        self.rebuild_bm25_index()
        self.save_to_disk()
        print(f"[RAG] Added '{source_name}': {added} semantic chunks (parent stored).")
        return added

    # ── Delete Document ───────────────────────────────────────────────────────
    def delete_document(self, source_name: str, auto_save: bool = True) -> int:
        initial = len(self.chunks)
        self.chunks = [c for c in self.chunks if c["source"] != source_name]
        # Remove parent
        self.parents = {k: v for k, v in self.parents.items()
                        if v["source"] != source_name}
        removed = initial - len(self.chunks)
        if removed > 0:
            self.rebuild_bm25_index()
            if auto_save:
                self.save_to_disk()
        return removed

    # ── BM25 Index ────────────────────────────────────────────────────────────
    def rebuild_bm25_index(self):
        """Rebuild BM25 stats from current self.chunks. Always in-sync."""
        self.vocab_df.clear()
        self.doc_lengths.clear()
        total_len = 0
        for ch in self.chunks:
            tokens  = ch["text"].lower().split()
            tok_set = set(tokens)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len
            for t in tok_set:
                self.vocab_df[t] = self.vocab_df.get(t, 0) + 1
        self.avg_doc_len = (total_len / len(self.chunks)) if self.chunks else 0.0

    def compute_bm25_score(self, query_tokens: list[str], chunk_idx: int,
                           k1: float = 1.5, b: float = 0.75) -> float:
        doc_len = self.doc_lengths[chunk_idx] if chunk_idx < len(self.doc_lengths) \
                  else max(1, int(self.avg_doc_len))
        N = len(self.chunks)
        if N == 0:
            return 0.0
        chunk_text_lower = self.chunks[chunk_idx]["text"].lower()
        score = 0.0
        for q in query_tokens:
            if q not in self.vocab_df:
                continue
            df  = self.vocab_df[q]
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
            tf  = chunk_text_lower.count(q)
            num = tf * (k1 + 1)
            den = tf + k1 * (1 - b + b * (doc_len / (self.avg_doc_len + 1e-5))) + 1e-10
            score += idf * (num / den)
        return score

    # ── Hybrid Search (Dense + BM25) ─────────────────────────────────────────
    def search_hybrid(self, query: str, top_k: int = 3,
                      alpha: float = 0.6) -> list[tuple[float, dict]]:
        if not self.chunks:
            return []
        q_emb    = self.compute_embedding(query)
        q_tokens = re.findall(r'\w+', query.lower())

        results = []
        for idx, ch in enumerate(self.chunks):
            emb        = np.array(ch["embedding"], dtype=np.float32)
            dense_sim  = float(np.dot(q_emb, emb))
            bm25       = self.compute_bm25_score(q_tokens, idx)
            combined   = alpha * dense_sim + (1.0 - alpha) * min(bm25 / 5.0, 1.0)
            results.append((combined, ch))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k * 2]   # retrieve 2x then rerank

    # ── Multi-hop Search ─────────────────────────────────────────────────────
    def search_multihop(self, query: str, hops: int = 2, top_k: int = 2) -> list[dict]:
        """
        Multi-hop RAG:
        1. Initial retrieval for original query
        2. Expand query with keywords from retrieved context
        3. Second retrieval for enriched query
        """
        all_chunks = []
        current_query = query

        for hop in range(hops):
            results = self.search_hybrid(current_query, top_k=top_k * 2, alpha=0.6)
            if not results:
                break

            hop_chunks = [ch for _, ch in results]
            hop_chunks = self.reranker.rerank(current_query, hop_chunks, top_k=top_k)
            all_chunks.extend(hop_chunks)

            # Expand query with key terms from retrieved context
            if hop < hops - 1:
                context_text = " ".join(c["text"] for c in hop_chunks[:2])
                # Extract high-IDF keywords from context
                context_words = re.findall(r'\b[a-zA-Z]{4,}\b', context_text.lower())
                unique_words  = list(dict.fromkeys(context_words))[:5]  # top 5 unique
                current_query = query + " " + " ".join(unique_words)

        # Deduplicate by text
        seen = set()
        unique_chunks = []
        for c in all_chunks:
            key = c["text"][:50]
            if key not in seen:
                seen.add(key)
                unique_chunks.append(c)

        return unique_chunks[:top_k]

    # ── Get Parent Context ────────────────────────────────────────────────────
    def get_parent_context(self, chunk: dict) -> str:
        """Return parent document text for richer context injection."""
        parent_id = chunk.get("parent_id")
        if parent_id and parent_id in self.parents:
            return self.parents[parent_id]["text"]
        return chunk["text"]

    # ── Retrieve with Re-ranking + Parent Injection ───────────────────────────
    def retrieve(self, query: str, top_k: int = 2,
                 use_multihop: bool = False,
                 use_parent: bool  = True) -> list[dict]:
        """Full retrieval pipeline: hybrid → rerank → parent expansion."""
        if use_multihop:
            candidates = self.search_multihop(query, hops=2, top_k=top_k * 3)
        else:
            raw = self.search_hybrid(query, top_k=top_k * 3, alpha=0.6)
            candidates = [ch for _, ch in raw]

        reranked = self.reranker.rerank(query, candidates, top_k=top_k)

        if use_parent:
            for c in reranked:
                c["_context"] = self.get_parent_context(c)
        else:
            for c in reranked:
                c["_context"] = c["text"]

        return reranked

    # ── Episodic Memory ───────────────────────────────────────────────────────
    def add_episodic_memory(self, conversation_turns: list[dict]) -> None:
        """
        Compress a list of conversation turns into a memory entry.
        turns: [{"role": "user"/"assistant", "content": "..."}]
        """
        # Simple extractive summary: join all content
        summary = " | ".join(
            f"{t['role']}: {t['content'][:100]}"
            for t in conversation_turns
        )
        emb = self.compute_embedding(summary)
        self.episodic_memory.append({
            "summary":   summary,
            "embedding": emb.tolist(),
            "turns":     len(conversation_turns)
        })
        print(f"[Memory] Compressed {len(conversation_turns)} turns into episodic memory.")

    def search_episodic_memory(self, query: str, top_k: int = 2) -> list[str]:
        """Retrieve relevant past conversation summaries."""
        if not self.episodic_memory:
            return []
        q_emb = self.compute_embedding(query)
        scored = []
        for mem in self.episodic_memory:
            emb = np.array(mem["embedding"], dtype=np.float32)
            sim = float(np.dot(q_emb, emb))
            scored.append((sim, mem["summary"]))
        scored.sort(reverse=True)
        return [s for _, s in scored[:top_k]]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_to_disk(self):
        try:
            payload = {
                "chunks":          self.chunks,
                "parents":         self.parents,
                "episodic_memory": self.episodic_memory
            }
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            print(f"[VectorDB] Saved {len(self.chunks)} chunks to {self.db_path}")
        except Exception as e:
            print(f"[VectorDB] Save error: {e}")

    def load_from_disk(self):
        if not os.path.exists(self.db_path):
            return
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            # Support old format (list) and new format (dict)
            if isinstance(payload, list):
                self.chunks = payload
                self.parents = {}
                self.episodic_memory = []
            else:
                self.chunks          = payload.get("chunks", [])
                self.parents         = payload.get("parents", {})
                self.episodic_memory = payload.get("episodic_memory", [])
            self.rebuild_bm25_index()
            print(f"[VectorDB] Loaded {len(self.chunks)} chunks, "
                  f"{len(self.parents)} parents, "
                  f"{len(self.episodic_memory)} episodic memories.")
        except Exception as e:
            print(f"[VectorDB] Load error: {e}")
            self.chunks = []
            self.parents = {}
            self.episodic_memory = []

    # ── Stats ─────────────────────────────────────────────────────────────────
    def get_stats(self) -> dict:
        source_counts: dict[str, int] = {}
        for c in self.chunks:
            s = c["source"]
            source_counts[s] = source_counts.get(s, 0) + 1
        return {
            "total_chunks":    len(self.chunks),
            "total_parents":   len(self.parents),
            "episodic_memory": len(self.episodic_memory),
            "sources":         [{"source": k, "chunks": v} for k, v in source_counts.items()],
            "persistent_path": self.db_path
        }
