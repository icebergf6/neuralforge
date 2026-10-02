#pragma once

#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>
#include <thread>
#include <functional>

// AVX2 SIMD for inner dot product
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(_M_X64))
    #include <immintrin.h>
    #define FA_HAS_AVX2 1
#endif

namespace lengine {

/**
 * Flash Attention 2 — Cache-Efficient Tiled Online Softmax Kernel
 *
 * Standard attention: O(N²) RAM (materialisasi matriks attention [N, N]).
 * Flash Attention 2: O(N) RAM — hitung attention score tile-by-tile di SRAM
 *   tanpa materialisasi penuh. Cocok untuk seq_len panjang.
 *
 * Algoritma (Dao et al., 2022 / 2023):
 *   Untuk setiap blok key/value:
 *     1. Hitung S_block = Q * K_block^T / sqrt(d)
 *     2. Update running max m_new = max(m_prev, max(S_block))
 *     3. Rescale accumulator: acc *= exp(m_prev - m_new)
 *     4. Akumulasikan nilai baru:  acc += exp(S - m_new) * V_block
 *     5. Normalisasi akhir: out = acc / l (running softmax denominator)
 */
class FlashAttention {
public:
    // ─── Scalar dot product (fallback) ───────────────────────────────────────
    static inline float dot_scalar(const float* __restrict__ a,
                                   const float* __restrict__ b, int n) {
        float s = 0.0f;
        for (int i = 0; i < n; ++i) s += a[i] * b[i];
        return s;
    }

    // ─── SIMD dot product (AVX2 FMA, 8 floats per cycle) ────────────────────
    static inline float dot_simd(const float* __restrict__ a,
                                  const float* __restrict__ b, int n) {
        float sum = 0.0f;
        int i = 0;
#ifdef FA_HAS_AVX2
        __m256 acc = _mm256_setzero_ps();
        for (; i + 8 <= n; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, acc);
        for (int k = 0; k < 8; ++k) sum += buf[k];
#endif
        for (; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }

    /**
     * compute_tiled_attention — Single head, Flash Attention 2 style.
     *
     * @param out        Output vector [head_size] — weighted sum of V
     * @param q_head     Query vector [head_size]
     * @param k_cache    Key cache pointer [seq_len * head_size], contiguous
     * @param v_cache    Value cache pointer [seq_len * head_size], contiguous
     * @param kv_stride  Stride in floats between consecutive time-steps in cache
     *                   (= kv_dim for the full layer KV cache slice)
     * @param current_pos  Number of tokens processed so far (inclusive)
     * @param head_size    Dimension per head
     * @param block_size   Tile size (default 64 — fits in ~2KB L1 cache)
     */
    static void compute_tiled_attention(
        float* __restrict__ out,
        const float* __restrict__ q_head,
        const float* __restrict__ k_cache,
        const float* __restrict__ v_cache,
        int kv_stride,
        int current_pos,
        int head_size,
        int block_size = 64
    ) {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        const int seq_len = current_pos + 1;

        // Running online-softmax state
        float m = -1e30f;   // running max
        float l = 0.0f;     // running sum of exp(s - m)

        // Accumulator for weighted V (output)
        std::vector<float> acc(head_size, 0.0f);

        // Local score buffer for one tile
        std::vector<float> s_tile(block_size);

        for (int b = 0; b < seq_len; b += block_size) {
            const int b_end = std::min(b + block_size, seq_len);
            const int tile_len = b_end - b;

            // ── Step 1: Compute tile scores S = Q · K^T * scale ──────────
            float m_tile = -1e30f;
            for (int t = 0; t < tile_len; ++t) {
                const float* k_t = k_cache + static_cast<size_t>(b + t) * kv_stride;
                float score = dot_simd(q_head, k_t, head_size) * scale;
                s_tile[t] = score;
                if (score > m_tile) m_tile = score;
            }

            // ── Step 2: Online softmax rescaling ─────────────────────────
            float m_new = std::max(m, m_tile);
            float alpha = std::exp(m - m_new);   // rescale old acc

            // Rescale old accumulator
            for (int i = 0; i < head_size; ++i) acc[i] *= alpha;

            // ── Step 3: Accumulate new tile ───────────────────────────────
            float l_tile = 0.0f;
            for (int t = 0; t < tile_len; ++t) {
                float p = std::exp(s_tile[t] - m_new);
                l_tile += p;
                const float* v_t = v_cache + static_cast<size_t>(b + t) * kv_stride;
                for (int i = 0; i < head_size; ++i) {
                    acc[i] += p * v_t[i];
                }
            }

            // Update running state
            l = l * alpha + l_tile;
            m = m_new;
        }

        // ── Step 4: Normalize output ─────────────────────────────────────
        const float inv_l = (l > 1e-10f) ? 1.0f / l : 0.0f;
        for (int i = 0; i < head_size; ++i) {
            out[i] = acc[i] * inv_l;
        }
    }

    /**
     * compute_multi_head_flash — All attention heads in parallel using threads.
     *
     * @param out          Output [n_heads * head_size]
     * @param q            Query [n_heads * head_size]
     * @param k_cache_layer  Key cache for this layer [seq_len * kv_dim]
     * @param v_cache_layer  Value cache for this layer [seq_len * kv_dim]
     * @param n_heads      Number of query heads
     * @param n_kv_heads   Number of KV heads (GQA support: n_kv_heads <= n_heads)
     * @param head_size    Dimension per head
     * @param current_pos  Current sequence position
     * @param num_threads  Threads to use for multi-head parallelism
     */
    static void compute_multi_head_flash(
        float* __restrict__ out,
        const float* __restrict__ q,
        const float* __restrict__ k_cache_layer,
        const float* __restrict__ v_cache_layer,
        int n_heads,
        int n_kv_heads,
        int head_size,
        int current_pos,
        int num_threads = 4
    ) {
        const int kv_dim    = n_kv_heads * head_size;
        const int kv_mul    = n_heads / n_kv_heads;  // GQA group size

        auto process_head = [&](int h_start, int h_end) {
            for (int h = h_start; h < h_end; ++h) {
                const float* q_h     = q + h * head_size;
                float*       out_h   = out + h * head_size;
                int kv_h             = h / kv_mul;  // GQA: which KV head to use
                const float* k_h     = k_cache_layer + kv_h * head_size;
                const float* v_h     = v_cache_layer + kv_h * head_size;

                compute_tiled_attention(
                    out_h, q_h,
                    k_h, v_h,
                    kv_dim,           // stride between time-steps
                    current_pos,
                    head_size
                );
            }
        };

        if (num_threads <= 1 || n_heads < num_threads) {
            process_head(0, n_heads);
            return;
        }

        // Dispatch heads across threads
        std::vector<std::thread> workers;
        int chunk = (n_heads + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            int hs = t * chunk;
            int he = std::min(hs + chunk, n_heads);
            if (hs >= he) break;
            workers.emplace_back(process_head, hs, he);
        }
        for (auto& th : workers) {
            if (th.joinable()) th.join();
        }
    }
};

} // namespace lengine
