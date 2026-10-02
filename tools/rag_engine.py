"""
Knowledge Pipeline: File Ingestion, Web Scraper, Text Chunking, dan Semantic Vector Indexing
Memproses file teks, PDF, atau halaman web menjadi potongan teks ber-vektor
untuk disuntikkan ke dalam sistem RAG.
"""

import re
import urllib.request
import urllib.parse
import numpy as np

class KnowledgeEngine:
    def __init__(self, embedding_dim=288):
        self.embedding_dim = embedding_dim
        self.chunks = []

    def clean_html(self, html_content):
        # Hapus tag script dan style
        cleaned = re.sub(r'<script.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<style.*?</style>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Hapus tag HTML lainnya
        cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
        # Normalisasi spasi dan baris baru
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def scrape_url(self, url):
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                content_type = response.headers.get_content_charset() or 'utf-8'
                html = response.read().decode(content_type, errors='replace')
                return self.clean_html(html)
        except Exception as e:
            return f"Error fetching URL: {str(e)}"

    def chunk_text(self, text, max_words=60, overlap=10):
        words = text.split()
        chunks = []
        i = 0
        while i < len(words):
            chunk = " ".join(words[i : i + max_words])
            if len(chunk.strip()) > 10:
                chunks.append(chunk)
            i += (max_words - overlap)
        return chunks

    def compute_embedding(self, text):
        """
        Menghasilkan representasi vektor 288 dimensi berbasis hash token semantik
        (Ringan, deterministik, dan kompatibel langsung dengan dimensi model).
        """
        vec = np.zeros(self.embedding_dim, dtype=np.float32)
        words = re.findall(r'\w+', text.lower())
        if not words:
            return vec
        
        for w in words:
            h = hash(w) % self.embedding_dim
            vec[h] += 1.0

        # Normalisasi L2 vector agar memiliki magnitude 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def add_document(self, source_name, content):
        new_chunks = self.chunk_text(content)
        start_id = len(self.chunks)
        added_count = 0
        for idx, ch in enumerate(new_chunks):
            emb = self.compute_embedding(ch)
            self.chunks.append({
                "id": start_id + idx,
                "source": source_name,
                "text": ch,
                "embedding": emb.tolist()  # Store as list for JSON compatibility
            })
            added_count += 1
        return added_count

    def search_similar(self, query, top_k=2):
        if not self.chunks:
            return []
        
        q_emb = self.compute_embedding(query)
        scores = []
        for ch in self.chunks:
            # Cosine similarity — convert from list to numpy if needed
            emb = np.array(ch["embedding"], dtype=np.float32)
            dot = float(np.dot(q_emb, emb))
            scores.append((dot, ch))

        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_k]

    def get_stats(self):
        sources = set(c["source"] for c in self.chunks)
        return {
            "total_chunks": len(self.chunks),
            "sources": list(sources)
        }
