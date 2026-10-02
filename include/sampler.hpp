#pragma once

#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

namespace lengine {

/**
 * Sampler: Mengubah logits output Transformer menjadi token ID berikutnya
 * Mendukung Temperature, Top-K, dan Top-P (Nucleus) Sampling.
 */
class Sampler {
private:
    std::mt19937 rng;

public:
    Sampler(unsigned int seed = 42) : rng(seed) {}

    int sample(float* logits, int vocab_size, float temperature = 0.8f, float topp = 0.9f) {
        // 1. Greedy sampling jika temperature == 0
        if (temperature <= 0.0f) {
            int max_i = 0;
            float max_p = logits[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logits[i] > max_p) {
                    max_p = logits[i];
                    max_i = i;
                }
            }
            return max_i;
        }

        // 2. Skalakan logits dengan temperature
        for (int i = 0; i < vocab_size; ++i) {
            logits[i] /= temperature;
        }

        // 3. Hitung Softmax probabilitas
        float max_logit = *std::max_element(logits, logits + vocab_size);
        float sum_exp = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            logits[i] = std::exp(logits[i] - max_logit);
            sum_exp += logits[i];
        }
        for (int i = 0; i < vocab_size; ++i) {
            logits[i] /= sum_exp;
        }

        // 4. Top-P (Nucleus) Filtering jika topp < 1.0
        if (topp > 0.0f && topp < 1.0f) {
            std::vector<std::pair<float, int>> probs;
            probs.reserve(vocab_size);
            for (int i = 0; i < vocab_size; ++i) {
                probs.emplace_back(logits[i], i);
            }

            // Urutkan probabilitas descending
            std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });

            // Akumulasi probabilitas kumulatif hingga mencapai cutoff topp
            float cumulative_prob = 0.0f;
            int last_idx = vocab_size - 1;
            for (int i = 0; i < vocab_size; ++i) {
                cumulative_prob += probs[i].first;
                if (cumulative_prob > topp) {
                    last_idx = i;
                    break;
                }
            }

            // Sample dari subset koin probabilitas yang lolos
            std::uniform_real_distribution<float> dist(0.0f, cumulative_prob);
            float r = dist(rng);
            float cdf = 0.0f;
            for (int i = 0; i <= last_idx; ++i) {
                cdf += probs[i].first;
                if (r <= cdf) {
                    return probs[i].second;
                }
            }
            return probs[0].second;
        }

        // 5. Standar Multinomial Sampling
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng);
        float cdf = 0.0f;
        for (int i = 0; i < vocab_size; ++i) {
            cdf += logits[i];
            if (r <= cdf) {
                return i;
            }
        }
        return vocab_size - 1;
    }
};

} // namespace lengine
