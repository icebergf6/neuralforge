#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <stdexcept>

namespace lengine {

/**
 * Byte-Pair Encoding (BPE) Tokenizer
 * Kompatibel dengan tokenizer.bin dari arsitektur Llama 2 / TinyStories.
 * 
 * Format tokenizer.bin:
 * - max_token_length (int32)
 * - N x [ score (float32), len (int32), bytes (char[]) ]
 */
class Tokenizer {
public:
    struct TokenInfo {
        std::string str;
        float score;
    };

    int vocab_size = 0;
    int max_token_length = 0;
    std::vector<TokenInfo> vocab;
    std::unordered_map<std::string, int> token_to_id;
    unsigned char byte_pieces[512] = {0};

    Tokenizer() = default;

    bool load(const std::string& path, int expected_vocab_size = 32000) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[Tokenizer] Error: Failed to open " << path << std::endl;
            return false;
        }

        vocab_size = expected_vocab_size;
        vocab.resize(vocab_size);

        if (!file.read(reinterpret_cast<char*>(&max_token_length), sizeof(int32_t))) {
            return false;
        }

        for (int i = 0; i < vocab_size; ++i) {
            float score = 0.0f;
            int32_t len = 0;
            file.read(reinterpret_cast<char*>(&score), sizeof(float));
            file.read(reinterpret_cast<char*>(&len), sizeof(int32_t));

            std::string piece(len, '\0');
            file.read(&piece[0], len);

            vocab[i] = {piece, score};
            token_to_id[piece] = i;
        }

        return true;
    }

    // Decode: ID token -> Teks
    std::string decode(int token_id) const {
        if (token_id < 0 || token_id >= vocab_size) {
            return "";
        }
        return vocab[token_id].str;
    }

    // Encode: Teks prompt -> list of Token IDs menggunakan BPE merge algorithm
    std::vector<int> encode(const std::string& text, bool bos = true, bool eos = false) const {
        std::vector<int> tokens;
        if (bos) {
            tokens.push_back(1); // 1 adalah BOS (Beginning-Of-Sentence) pada Llama
        }

        if (text.empty()) {
            return tokens;
        }

        // 1. Inisialisasi awal: setiap karakter string dipetakan ke token vocab (atau fallback byte)
        for (size_t i = 0; i < text.size(); ++i) {
            std::string single_char(1, text[i]);
            auto it = token_to_id.find(single_char);
            if (it != token_to_id.end()) {
                tokens.push_back(it->second);
            } else {
                // Fallback byte token: Llama memetakan byte <0x00>..<0xFF> ke index token 3..258
                tokens.push_back(static_cast<unsigned char>(text[i]) + 3);
            }
        }

        // 2. Iterasi BPE: cari pasangan token berurutan dengan merge score tertinggi
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;

            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                std::string merged = vocab[tokens[i]].str + vocab[tokens[i + 1]].str;
                auto it = token_to_id.find(merged);
                if (it != token_to_id.end()) {
                    float score = vocab[it->second].score;
                    if (score > best_score) {
                        best_score = score;
                        best_id = it->second;
                        best_idx = static_cast<int>(i);
                    }
                }
            }

            // Jika tidak ada lagi pasangan yang bisa digabungkan, hentikan loop
            if (best_idx == -1) {
                break;
            }

            // Gabungkan pasangan (merge pair)
            tokens[best_idx] = best_id;
            tokens.erase(tokens.begin() + best_idx + 1);
        }

        if (eos) {
            tokens.push_back(2); // 2 adalah EOS (End-Of-Sentence) pada Llama
        }

        return tokens;
    }
};

} // namespace lengine
