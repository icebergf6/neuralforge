#pragma once

#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "ops.hpp"

namespace lengine {

/**
 * Representasi sebuah potongan pengetahuan (Knowledge Chunk)
 */
struct KnowledgeChunk {
    int id;
    std::string source;      // Nama file atau URL sumber
    std::string text;        // Isi teks dokumen
    std::vector<float> embedding; // Vektor representasi semantik
};

/**
 * Custom In-Memory Vector Store & Cosine Similarity Engine
 * Ditenagai oleh instruksi AVX2 SIMD untuk pencarian super cepat (< 1ms)
 */
class VectorStore {
public:
    std::vector<KnowledgeChunk> chunks;
    int embedding_dim = 288; // Mengikuti dimensi model (288 float)

    VectorStore() = default;

    // Menambahkan dokumen baru ke dalam index
    void add_chunk(int id, const std::string& source, const std::string& text, const std::vector<float>& emb) {
        chunks.push_back({id, source, text, emb});
    }

    // Menghitung Cosine Similarity: dot(a, b) / (norm(a) * norm(b))
    static float cosine_similarity(const float* a, const float* b, int size) {
        float dot = Ops::dot_product(a, b, size);
        float norm_a = Ops::dot_product(a, a, size);
        float norm_b = Ops::dot_product(b, b, size);

        if (norm_a <= 0.0f || norm_b <= 0.0f) return 0.0f;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    // Mencari Top-K potongan pengetahuan yang paling relevan dengan query vektor
    std::vector<std::pair<float, KnowledgeChunk>> search(const std::vector<float>& query_embedding, int top_k = 3) const {
        std::vector<std::pair<float, KnowledgeChunk>> results;
        if (chunks.empty() || query_embedding.empty()) return results;

        for (const auto& item : chunks) {
            float sim = cosine_similarity(query_embedding.data(), item.embedding.data(), static_cast<int>(query_embedding.size()));
            results.emplace_back(sim, item);
        }

        // Urutkan berdasarkan kemiripan tertinggi (descending)
        std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        if (results.size() > static_cast<size_t>(top_k)) {
            results.resize(top_k);
        }

        return results;
    }

    size_t size() const {
        return chunks.size();
    }

    void clear() {
        chunks.clear();
    }
};

} // namespace lengine
