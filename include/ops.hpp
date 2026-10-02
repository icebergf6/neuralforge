#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <functional>
#include <thread>

// AVX2 SIMD
#if defined(__AVX2__) || (defined(_MSC_VER) && defined(_M_X64))
    #include <immintrin.h>
    #define LENGINE_HAS_AVX2 1
#endif

namespace lengine {

/**
 * High-Performance Math Kernel dengan Optimasi:
 * 1. SIMD Vectorization (AVX2 256-bit FMA)
 * 2. Multi-threaded Parallel Matrix Multiplication
 * 3. 8-Bit Integer Quantization (Q8_0)
 * 4. 4-Bit Integer Quantization (Q4_0) — NEW
 * 5. Fused Kernels untuk memory bandwidth reduction
 */
class Ops {
public:
    // ─── 1. RMSNorm (Root Mean Square Normalization) ─────────────────────────
    static void rmsnorm(float* out, const float* x, const float* weight, int size, float eps = 1e-5f) {
        float sum_sq = 0.0f;
        int i = 0;

#if defined(LENGINE_HAS_AVX2)
        __m256 v_sum = _mm256_setzero_ps();
        for (; i + 8 <= size; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            v_sum = _mm256_fmadd_ps(vx, vx, v_sum);
        }
        alignas(32) float buffer[8];
        _mm256_storeu_ps(buffer, v_sum);
        for (int b = 0; b < 8; ++b) sum_sq += buffer[b];
#endif

        for (; i < size; ++i) sum_sq += x[i] * x[i];

        float scale = 1.0f / std::sqrt((sum_sq / size) + eps);
        i = 0;

#if defined(LENGINE_HAS_AVX2)
        __m256 v_scale = _mm256_set1_ps(scale);
        for (; i + 8 <= size; i += 8) {
            __m256 vx = _mm256_loadu_ps(x + i);
            __m256 vw = _mm256_loadu_ps(weight + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_mul_ps(vx, v_scale), vw));
        }
#endif
        for (; i < size; ++i) out[i] = x[i] * scale * weight[i];
    }

    // ─── 2. SIMD Vector Dot Product ──────────────────────────────────────────
    static inline float dot_product(const float* a, const float* b, int size) {
        float sum = 0.0f;
        int j = 0;

#if defined(LENGINE_HAS_AVX2)
        __m256 v_acc = _mm256_setzero_ps();
        for (; j + 8 <= size; j += 8) {
            __m256 va = _mm256_loadu_ps(a + j);
            __m256 vb = _mm256_loadu_ps(b + j);
            v_acc = _mm256_fmadd_ps(va, vb, v_acc);
        }
        alignas(32) float buf[8];
        _mm256_storeu_ps(buf, v_acc);
        for (int k = 0; k < 8; ++k) sum += buf[k];
#endif
        for (; j < size; ++j) sum += a[j] * b[j];
        return sum;
    }

    // ─── 3. Matrix Multiply: out = W * x  [rows x cols] ─────────────────────
    static void matmul(float* out, const float* x, const float* w, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            out[i] = dot_product(w + (size_t)i * cols, x, cols);
        }
    }

    // ─── 4. Multi-Threaded Parallel Matmul ───────────────────────────────────
    static void matmul_parallel(float* out, const float* x, const float* w,
                                int rows, int cols, int num_threads = 4) {
        if (num_threads <= 1 || rows < 128) {
            matmul(out, x, w, rows, cols);
            return;
        }

        std::vector<std::thread> workers;
        int chunk = (rows + num_threads - 1) / num_threads;

        for (int t = 0; t < num_threads; ++t) {
            int rs = t * chunk;
            int re = std::min(rs + chunk, rows);
            if (rs >= re) break;
            workers.emplace_back([=]() {
                for (int i = rs; i < re; ++i)
                    out[i] = dot_product(w + (size_t)i * cols, x, cols);
            });
        }
        for (auto& th : workers) if (th.joinable()) th.join();
    }

    // ─── 5. Rotary Positional Embeddings (RoPE) ───────────────────────────────
    static void rope(float* vec, int pos, int head_dim) {
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / std::pow(10000.0f, (float)i / (float)head_dim);
            float val  = (float)pos * freq;
            float c = std::cos(val), s = std::sin(val);
            float v0 = vec[i], v1 = vec[i + 1];
            vec[i]     = v0 * c - v1 * s;
            vec[i + 1] = v0 * s + v1 * c;
        }
    }

    // ─── 6. Numerically Stable Softmax ───────────────────────────────────────
    static void softmax(float* x, int size) {
        if (size <= 0) return;
        float mx = x[0];
        for (int i = 1; i < size; ++i) if (x[i] > mx) mx = x[i];

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) { x[i] = std::exp(x[i] - mx); sum += x[i]; }

        float inv = 1.0f / sum;
        for (int i = 0; i < size; ++i) x[i] *= inv;
    }

    // ─── 7. SiLU & SwiGLU Activation ─────────────────────────────────────────
    static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

    static void swiglu(float* out, const float* gate, const float* up, int size) {
        int i = 0;
#if defined(LENGINE_HAS_AVX2)
        // Approximate SiLU via tanh for vectorization: silu(x) ≈ 0.5x*(1+tanh(0.7978*(x+0.0446x^3)))
        // For accuracy we do scalar SiLU; SIMD handles the multiply step only
        for (; i + 8 <= size; i += 8) {
            alignas(32) float g_buf[8], u_buf[8];
            _mm256_storeu_ps(g_buf, _mm256_loadu_ps(gate + i));
            _mm256_storeu_ps(u_buf, _mm256_loadu_ps(up + i));
            alignas(32) float o_buf[8];
            for (int k = 0; k < 8; ++k) o_buf[k] = silu(g_buf[k]) * u_buf[k];
            _mm256_storeu_ps(out + i, _mm256_loadu_ps(o_buf));
        }
#endif
        for (; i < size; ++i) out[i] = silu(gate[i]) * up[i];
    }

    // ─── 8. Vector Add ────────────────────────────────────────────────────────
    static void add(float* out, const float* a, const float* b, int size) {
        int i = 0;
#if defined(LENGINE_HAS_AVX2)
        for (; i + 8 <= size; i += 8) {
            _mm256_storeu_ps(out + i,
                _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
#endif
        for (; i < size; ++i) out[i] = a[i] + b[i];
    }

    // ─── 9. Element-wise multiply-accumulate ──────────────────────────────────
    static void mul_acc(float* __restrict__ acc, const float* __restrict__ a,
                         const float* __restrict__ b, int size) {
        int i = 0;
#if defined(LENGINE_HAS_AVX2)
        for (; i + 8 <= size; i += 8) {
            __m256 va  = _mm256_loadu_ps(a + i);
            __m256 vb  = _mm256_loadu_ps(b + i);
            __m256 vc  = _mm256_loadu_ps(acc + i);
            _mm256_storeu_ps(acc + i, _mm256_fmadd_ps(va, vb, vc));
        }
#endif
        for (; i < size; ++i) acc[i] += a[i] * b[i];
    }
};

// ─── Q8_0 Quantization Block (8-bit) ─────────────────────────────────────────
struct BlockQ8_0 {
    float   scale;
    int8_t  qs[32];
};

// ─── Q4_0 Quantization Block (4-bit) — NEW ───────────────────────────────────
// Stores 32 int4 values packed into 16 bytes + 1 float scale
struct BlockQ4_0 {
    uint16_t scale_f16;     // scale stored as fp16
    uint8_t  qs[16];        // 32 x 4-bit values packed as lo|hi pairs

    float scale_as_f32() const {
        // fp16 → fp32 conversion
        uint32_t sign     = (scale_f16 >> 15) & 1;
        uint32_t exponent = (scale_f16 >> 10) & 0x1F;
        uint32_t mantissa =  scale_f16        & 0x3FF;
        uint32_t f;
        if (exponent == 0)       f = (sign<<31)|((127-14)<<23)|(mantissa<<13);
        else if (exponent == 31) f = (sign<<31)|(0xFF<<23)|(mantissa<<13);
        else                     f = (sign<<31)|((exponent+127-15)<<23)|(mantissa<<13);
        float r; std::memcpy(&r, &f, 4); return r;
    }
};

class Quantization {
public:
    // ── Q8_0 ─────────────────────────────────────────────────────────────────
    static void quantize_q8_block(const float* src, BlockQ8_0* block) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float v = std::abs(src[i]);
            if (v > amax) amax = v;
        }
        float d  = amax / 127.0f;
        block->scale = d;
        float id = d > 0.0f ? (1.0f / d) : 0.0f;
        for (int i = 0; i < 32; ++i)
            block->qs[i] = static_cast<int8_t>(std::round(src[i] * id));
    }

    static float dot_q8(const BlockQ8_0* b, const float* x) {
        float sum = 0.0f;
        for (int i = 0; i < 32; ++i) sum += (b->qs[i] * b->scale) * x[i];
        return sum;
    }

    // ── Q4_0 — NEW ───────────────────────────────────────────────────────────
    static void quantize_q4_block(const float* src, BlockQ4_0* block) {
        // Find absolute max for scale
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float v = std::abs(src[i]);
            if (v > amax) amax = v;
        }
        // Scale: maps [-amax, +amax] → [-7, +7] (4-bit signed with offset 8)
        float d  = amax / 7.0f;
        float id = d > 0.0f ? (1.0f / d) : 0.0f;

        // Store scale as fp16
        // Simple float→fp16 (round-to-nearest, no subnormal handling)
        uint32_t fi; std::memcpy(&fi, &d, 4);
        uint32_t sign     = (fi >> 31) & 1;
        int      exponent = ((fi >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = (fi >> 13) & 0x3FF;
        if (exponent <= 0) exponent = 0;
        if (exponent >= 31) exponent = 31;
        block->scale_f16 = static_cast<uint16_t>((sign << 15) | (exponent << 10) | mantissa);

        for (int i = 0; i < 16; ++i) {
            int8_t lo = static_cast<int8_t>(std::max(-8.0f, std::min(7.0f, std::round(src[i * 2 + 0] * id))));
            int8_t hi = static_cast<int8_t>(std::max(-8.0f, std::min(7.0f, std::round(src[i * 2 + 1] * id))));
            block->qs[i] = static_cast<uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
        }
    }

    static float dot_q4(const BlockQ4_0* b, const float* x) {
        float scale = b->scale_as_f32();
        float sum   = 0.0f;
        for (int i = 0; i < 16; ++i) {
            int8_t lo = static_cast<int8_t>((b->qs[i] & 0x0F) - 8);
            int8_t hi = static_cast<int8_t>((b->qs[i] >> 4)  - 8);
            sum += (lo * scale) * x[i * 2 + 0];
            sum += (hi * scale) * x[i * 2 + 1];
        }
        return sum;
    }

    // ── Quantize full weight matrix (float32 → Q4_0) ─────────────────────────
    // Returns number of blocks. outBlocks must be pre-allocated: (n_elements/32) blocks
    static size_t quantize_matrix_q4(const float* data, BlockQ4_0* out, size_t n_elements) {
        size_t n_blocks = n_elements / 32;
        for (size_t b = 0; b < n_blocks; ++b)
            quantize_q4_block(data + b * 32, out + b);
        return n_blocks;
    }

    // ── Quantize full weight matrix (float32 → Q8_0) ─────────────────────────
    static size_t quantize_matrix_q8(const float* data, BlockQ8_0* out, size_t n_elements) {
        size_t n_blocks = n_elements / 32;
        for (size_t b = 0; b < n_blocks; ++b)
            quantize_q8_block(data + b * 32, out + b);
        return n_blocks;
    }

    // ── Benchmark: memory saved with Q4 vs F32 ───────────────────────────────
    static void print_compression_stats(size_t n_elements) {
        size_t f32_bytes = n_elements * 4;
        size_t q8_bytes  = (n_elements / 32) * (1 * 4 + 32 * 1);  // scale + int8s
        size_t q4_bytes  = (n_elements / 32) * (2 + 16);           // f16 scale + 4-bit
        std::printf("[Quant Stats] F32: %zu MB | Q8: %zu MB (%.1fx) | Q4: %zu MB (%.1fx)\n",
            f32_bytes/(1024*1024), q8_bytes/(1024*1024), (float)f32_bytes/q8_bytes,
            q4_bytes/(1024*1024),  (float)f32_bytes/q4_bytes);
    }
};

} // namespace lengine
