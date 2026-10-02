#pragma once

#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <thread>
#include <cstring>
#include <cstdio>
#include "model.hpp"
#include "tokenizer.hpp"
#include "sampler.hpp"
#include "ops.hpp"
#include "flash_attn.hpp"

namespace lengine {

/**
 * Inference Engine v2 — dengan:
 *  - Flash Attention 2 (tiled, O(N) RAM)
 *  - Speculative Decoding (draft + verify)
 *  - Multi-threaded forward pass
 *  - Streaming token callback
 *  - KV-Cache dengan support GQA
 */
class Engine {
public:
    Model     model;
    Tokenizer tokenizer;
    Sampler   sampler;

    int  num_threads    = 4;
    bool use_flash_attn = true;   // Toggle Flash Attention 2

    // ── Activation Buffers (allocated once at init) ──────────────────────────
    std::vector<float> x, xb, xb2, hb, hb2, q, k, v, att, logits;

    // ── KV-Cache [n_layers × seq_len × kv_dim] ───────────────────────────────
    std::vector<float> key_cache;
    std::vector<float> value_cache;

    Engine() {
        unsigned hw = std::thread::hardware_concurrency();
        num_threads = (hw > 0) ? static_cast<int>(hw) : 4;
    }

    bool init(const std::string& model_path, const std::string& tokenizer_path) {
        if (!model.load(model_path))                                       return false;
        if (!tokenizer.load(tokenizer_path, model.config.vocab_size))      return false;

        const auto& c = model.config;
        int head_size = c.dim / c.n_heads;
        int kv_dim    = c.n_kv_heads * head_size;

        x.resize(c.dim);  xb.resize(c.dim);  xb2.resize(c.dim);
        hb.resize(c.hidden_dim); hb2.resize(c.hidden_dim);
        q.resize(c.dim);  k.resize(kv_dim);  v.resize(kv_dim);
        att.resize(c.n_heads * c.seq_len);
        logits.resize(c.vocab_size);

        size_t kv_sz = (size_t)c.n_layers * c.seq_len * kv_dim;
        key_cache.assign(kv_sz, 0.0f);
        value_cache.assign(kv_sz, 0.0f);

        std::cout << "[Engine] Init complete. Threads=" << num_threads
                  << "  FlashAttn=" << (use_flash_attn ? "ON" : "OFF") << "\n";
        return true;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // forward: single transformer forward pass for token at position pos
    // ─────────────────────────────────────────────────────────────────────────
    void forward(int token, int pos) {
        const auto& c  = model.config;
        const auto& w  = model.weights;
        int dim        = c.dim;
        int hidden_dim = c.hidden_dim;
        int head_size  = dim / c.n_heads;
        int kv_dim     = c.n_kv_heads * head_size;
        int kv_mul     = c.n_heads / c.n_kv_heads;

        // 1. Token embedding lookup
        float* row = w.token_embedding_table + (size_t)token * dim;
        std::memcpy(x.data(), row, dim * sizeof(float));

        for (int l = 0; l < c.n_layers; ++l) {
            // A. Attention RMSNorm
            Ops::rmsnorm(xb.data(), x.data(), w.rms_att_weight[l], dim);

            // B. QKV projections
            Ops::matmul_parallel(q.data(), xb.data(), w.wq[l], dim,    dim,    num_threads);
            Ops::matmul_parallel(k.data(), xb.data(), w.wk[l], kv_dim, dim,    num_threads);
            Ops::matmul_parallel(v.data(), xb.data(), w.wv[l], kv_dim, dim,    num_threads);

            // C. Apply RoPE
            for (int h = 0; h < c.n_heads;    ++h) Ops::rope(q.data() + h * head_size, pos, head_size);
            for (int h = 0; h < c.n_kv_heads; ++h) Ops::rope(k.data() + h * head_size, pos, head_size);

            // D. Write K,V to cache
            size_t layer_off = (size_t)l * c.seq_len * kv_dim + (size_t)pos * kv_dim;
            std::memcpy(key_cache.data()   + layer_off, k.data(), kv_dim * sizeof(float));
            std::memcpy(value_cache.data() + layer_off, v.data(), kv_dim * sizeof(float));

            // E. Multi-Head Attention
            if (use_flash_attn) {
                // ── Flash Attention 2: tiled, O(N) RAM, parallel heads ───────
                const float* k_layer = key_cache.data()   + (size_t)l * c.seq_len * kv_dim;
                const float* v_layer = value_cache.data() + (size_t)l * c.seq_len * kv_dim;

                FlashAttention::compute_multi_head_flash(
                    xb.data(), q.data(),
                    k_layer, v_layer,
                    c.n_heads, c.n_kv_heads, head_size, pos, num_threads
                );
            } else {
                // ── Standard Scaled Dot-Product Attention ────────────────────
                float scale = 1.0f / std::sqrt((float)head_size);
                for (int h = 0; h < c.n_heads; ++h) {
                    float* q_h   = q.data()   + h * head_size;
                    float* att_h = att.data()  + h * c.seq_len;
                    int kv_h     = h / kv_mul;

                    for (int t = 0; t <= pos; ++t) {
                        size_t k_off = (size_t)l * c.seq_len * kv_dim + (size_t)t * kv_dim + kv_h * head_size;
                        att_h[t] = Ops::dot_product(q_h, key_cache.data() + k_off, head_size) * scale;
                    }
                    Ops::softmax(att_h, pos + 1);

                    float* out_h = xb.data() + h * head_size;
                    std::memset(out_h, 0, head_size * sizeof(float));
                    for (int t = 0; t <= pos; ++t) {
                        size_t v_off = (size_t)l * c.seq_len * kv_dim + (size_t)t * kv_dim + kv_h * head_size;
                        float a = att_h[t];
                        for (int i = 0; i < head_size; ++i) out_h[i] += a * value_cache[v_off + i];
                    }
                }
            }

            // F. Output projection + Residual 1
            Ops::matmul_parallel(xb2.data(), xb.data(), w.wo[l], dim, dim, num_threads);
            Ops::add(x.data(), x.data(), xb2.data(), dim);

            // G. FFN RMSNorm
            Ops::rmsnorm(xb.data(), x.data(), w.rms_ffn_weight[l], dim);

            // H. SwiGLU FFN: w2(silu(w1(x)) * w3(x))
            Ops::matmul_parallel(hb.data(),  xb.data(), w.w1[l], hidden_dim, dim,        num_threads);
            Ops::matmul_parallel(hb2.data(), xb.data(), w.w3[l], hidden_dim, dim,        num_threads);
            Ops::swiglu(hb.data(), hb.data(), hb2.data(), hidden_dim);
            Ops::matmul_parallel(xb.data(),  hb.data(),  w.w2[l], dim,       hidden_dim, num_threads);

            // I. Residual 2
            Ops::add(x.data(), x.data(), xb.data(), dim);
        }

        // 3. Final RMSNorm + LM Head
        Ops::rmsnorm(x.data(), x.data(), w.rms_final_weight, dim);
        Ops::matmul_parallel(logits.data(), x.data(), w.wcls, c.vocab_size, dim, num_threads);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // generate: autoregressive loop with streaming callback
    //   on_token: called for each generated token string (for streaming UI)
    // ─────────────────────────────────────────────────────────────────────────
    void generate(
        const std::string& prompt,
        int   max_new_tokens = 64,
        float temperature    = 0.8f,
        float topp           = 0.9f,
        std::function<void(const std::string&)> on_token = nullptr
    ) {
        auto tokens = tokenizer.encode(prompt, true, false);
        if (tokens.empty()) return;

        std::cout << "\n[Engine] Prompt: \"" << prompt << "\"\n";
        std::cout << std::string(60, '-') << "\n";

        // Print prompt back
        for (size_t i = 1; i < tokens.size(); ++i)
            std::cout << tokenizer.decode(tokens[i]) << std::flush;

        int token      = tokens[0];
        int pos        = 0;
        int prompt_len = (int)tokens.size();
        int gen_count  = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        while (pos < model.config.seq_len && gen_count < max_new_tokens) {
            forward(token, pos);

            int next;
            if (pos + 1 < prompt_len) {
                next = tokens[pos + 1];
            } else {
                next = sampler.sample(logits.data(), model.config.vocab_size, temperature, topp);
                gen_count++;
                if (next == 2) break;  // EOS

                std::string word = tokenizer.decode(next);
                std::cout << word << std::flush;
                if (on_token) on_token(word);
            }
            token = next;
            pos++;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double tps     = (gen_count > 0 && elapsed > 0) ? gen_count / elapsed : 0.0;

        std::cout << "\n" << std::string(60, '-') << "\n";
        std::printf("[Stats] Tokens: %d | Time: %.2fs | Speed: %.1f tok/s | Threads: %d | Attn: %s\n",
            gen_count, elapsed, tps, num_threads, use_flash_attn ? "Flash2" : "Standard");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // speculative_decode: Draft model generates N tokens, verifier validates.
    //   draft_engine: smaller/faster engine used to propose tokens
    //   gamma:        number of draft tokens to propose per step
    //
    // Speed gain: ~2-3x when draft acceptance rate > 70%
    // ─────────────────────────────────────────────────────────────────────────
    void speculative_decode(
        Engine&            draft_engine,
        const std::string& prompt,
        int                max_new_tokens = 128,
        float              temperature    = 0.8f,
        int                gamma          = 5    // draft tokens per step
    ) {
        auto tokens = tokenizer.encode(prompt, true, false);
        if (tokens.empty()) return;

        std::cout << "\n[SpecDecode] Prompt: \"" << prompt << "\"  gamma=" << gamma << "\n";
        std::cout << std::string(60, '-') << "\n";

        // Print prompt
        for (size_t i = 1; i < tokens.size(); ++i)
            std::cout << tokenizer.decode(tokens[i]) << std::flush;

        // Sync prefill: run both engines through the prompt
        int pos = 0;
        int token = tokens[0];
        int prompt_len = (int)tokens.size();

        for (int t = 0; t < prompt_len; ++t) {
            forward(token, pos);
            draft_engine.forward(token, pos);
            if (t + 1 < prompt_len) token = tokens[t + 1];
            pos++;
        }

        int total_accepted = 0;
        int total_draft    = 0;
        int gen_count      = 0;

        auto t0 = std::chrono::high_resolution_clock::now();

        while (gen_count < max_new_tokens && pos < model.config.seq_len) {
            // ── Phase 1: Draft γ tokens using draft engine ──────────────────
            std::vector<int>   draft_tokens;
            std::vector<float> draft_probs;  // probability of each draft token

            int draft_pos = pos;
            int draft_tok = token;

            for (int g = 0; g < gamma; ++g) {
                draft_engine.forward(draft_tok, draft_pos);

                // Get draft prob for sampled token
                std::vector<float> d_logits = draft_engine.logits;
                int next = sampler.sample(d_logits.data(), model.config.vocab_size, temperature, 0.95f);

                // Compute softmax to get probability
                Ops::softmax(d_logits.data(), model.config.vocab_size);
                draft_probs.push_back(d_logits[next]);
                draft_tokens.push_back(next);

                draft_tok = next;
                draft_pos++;

                if (next == 2) break;  // EOS in draft
            }

            // ── Phase 2: Verify draft tokens with target engine ──────────────
            int accepted = 0;
            for (int g = 0; g < (int)draft_tokens.size(); ++g) {
                forward(token, pos);

                // Get target probability for draft token
                std::vector<float> t_logits = logits;
                Ops::softmax(t_logits.data(), model.config.vocab_size);
                float p_target = t_logits[draft_tokens[g]];
                float p_draft  = draft_probs[g];

                // Acceptance criterion: accept if p_target/p_draft >= uniform random
                float ratio = p_target / (p_draft + 1e-10f);
                float u     = (float)rand() / RAND_MAX;

                if (u < ratio) {
                    // Accept draft token
                    token = draft_tokens[g];
                    pos++;
                    accepted++;
                    total_accepted++;

                    std::string word = tokenizer.decode(token);
                    std::cout << word << std::flush;
                    gen_count++;
                    if (token == 2 || gen_count >= max_new_tokens) goto done;
                } else {
                    // Reject: sample from adjusted distribution
                    // p_adjusted[x] = max(0, p_target - p_draft)
                    std::vector<float> adj(model.config.vocab_size);
                    for (int i = 0; i < model.config.vocab_size; ++i) {
                        float p_t_i = t_logits[i];
                        std::vector<float> d_logits2 = draft_engine.logits;
                        Ops::softmax(d_logits2.data(), model.config.vocab_size);
                        adj[i] = std::max(0.0f, p_t_i - d_logits2[i]);
                    }
                    float sum_adj = 0;
                    for (float a : adj) sum_adj += a;
                    if (sum_adj > 0) for (float& a : adj) a /= sum_adj;

                    // Sample from adjusted
                    int corrected = sampler.sample(adj.data(), model.config.vocab_size, 1.0f, 0.0f);
                    token = corrected;
                    pos++;
                    std::string word = tokenizer.decode(token);
                    std::cout << word << std::flush;
                    gen_count++;
                    if (token == 2 || gen_count >= max_new_tokens) goto done;
                    break;
                }
            }
            total_draft += (int)draft_tokens.size();

            if (accepted == 0) {
                // No draft tokens accepted, sample one from target
                forward(token, pos);
                token = sampler.sample(logits.data(), model.config.vocab_size, temperature, 0.9f);
                if (token == 2) break;
                std::cout << tokenizer.decode(token) << std::flush;
                pos++;
                gen_count++;
            }
        }
        done:
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double tps = (gen_count > 0 && elapsed > 0) ? gen_count / elapsed : 0.0;
        float  acc_rate = total_draft > 0 ? (float)total_accepted / total_draft : 0.0f;

        std::cout << "\n" << std::string(60, '-') << "\n";
        std::printf("[SpecDecode Stats] Tokens: %d | Time: %.2fs | Speed: %.1f tok/s\n",
            gen_count, elapsed, tps);
        std::printf("  Accept Rate: %.1f%%  (accepted %d / %d draft)\n",
            acc_rate * 100.0f, total_accepted, total_draft);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Reset KV-cache (call between separate conversations)
    // ─────────────────────────────────────────────────────────────────────────
    void reset_kv_cache() {
        std::fill(key_cache.begin(),   key_cache.end(),   0.0f);
        std::fill(value_cache.begin(), value_cache.end(), 0.0f);
    }
};

} // namespace lengine
