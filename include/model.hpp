#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <iostream>

namespace lengine {

/**
 * Hyperparameters arsitektur Llama Transformer
 * (Ukuran layer, hidden dimension, head count, dsb.)
 */
struct Config {
    int dim = 0;        // Transformer dimension (e.g. 288 di stories15M, 4096 di 7B)
    int hidden_dim = 0; // FFN hidden layer dimension (e.g. 768)
    int n_layers = 0;   // Jumlah Transformer blocks (e.g. 6)
    int n_heads = 0;    // Jumlah Query attention heads (e.g. 6)
    int n_kv_heads = 0; // Jumlah Key/Value heads (GQA support, e.g. 6)
    int vocab_size = 0; // Ukuran vocabulary (32000)
    int seq_len = 0;    // Max sequence length (e.g. 256)
};

/**
 * Model Weights Pointer
 * Menyimpan pointer ke memory buffer bobot model yang di-load dari file biner.
 */
struct TransformerWeights {
    // Token Embedding Table: [vocab_size, dim]
    float* token_embedding_table = nullptr;

    // Weights per layer (x n_layers)
    std::vector<float*> rms_att_weight; // [n_layers][dim]
    std::vector<float*> wq;             // [n_layers][dim, n_heads * head_size]
    std::vector<float*> wk;             // [n_layers][dim, n_kv_heads * head_size]
    std::vector<float*> wv;             // [n_layers][dim, n_kv_heads * head_size]
    std::vector<float*> wo;             // [n_layers][n_heads * head_size, dim]
    std::vector<float*> rms_ffn_weight; // [n_layers][dim]
    std::vector<float*> w1;             // [n_layers][hidden_dim, dim] (gate projection)
    std::vector<float*> w2;             // [n_layers][dim, hidden_dim] (down projection)
    std::vector<float*> w3;             // [n_layers][hidden_dim, dim] (up projection)

    // Final Output Head
    float* rms_final_weight = nullptr;  // [dim]
    float* wcls = nullptr;              // [vocab_size, dim] (classifier / LM head)
};

/**
 * Model Container & Loader
 */
class Model {
public:
    Config config;
    TransformerWeights weights;
    std::vector<float> data_buffer; // Menyimpan seluruh float weights di RAM

    bool load(const std::string& checkpoint_path) {
        std::ifstream file(checkpoint_path, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "[Model] Error: Failed to open model checkpoint " << checkpoint_path << std::endl;
            return false;
        }

        // 1. Baca Header Config (7 integer)
        if (!file.read(reinterpret_cast<char*>(&config), sizeof(Config))) {
            std::cerr << "[Model] Error: Failed to read model config header." << std::endl;
            return false;
        }

        std::cout << "========================================\n";
        std::cout << " [Model Loaded Config]\n";
        std::cout << "  - Dimension (dim):     " << config.dim << "\n";
        std::cout << "  - Hidden Dim:          " << config.hidden_dim << "\n";
        std::cout << "  - Layers:              " << config.n_layers << "\n";
        std::cout << "  - Attention Heads:     " << config.n_heads << "\n";
        std::cout << "  - KV Heads:            " << config.n_kv_heads << "\n";
        std::cout << "  - Vocab Size:          " << config.vocab_size << "\n";
        std::cout << "  - Max Seq Length:      " << config.seq_len << "\n";
        std::cout << "========================================\n";

        // 2. Baca seluruh sisa file bobot (weights)
        file.seekg(0, std::ios::end);
        size_t file_size = file.tellg();
        size_t weights_bytes = file_size - sizeof(Config);
        size_t num_floats = weights_bytes / sizeof(float);

        file.seekg(sizeof(Config), std::ios::beg);
        data_buffer.resize(num_floats);
        if (!file.read(reinterpret_cast<char*>(data_buffer.data()), weights_bytes)) {
            std::cerr << "[Model] Error reading weights data.\n";
            return false;
        }

        // 3. Map pointer bobot ke layer-layer Transformer
        float* ptr = data_buffer.data();
        weights.token_embedding_table = ptr;
        ptr += static_cast<size_t>(config.vocab_size) * config.dim;

        int L = config.n_layers;
        int d = config.dim;
        int hd = config.hidden_dim;
        int head_size = d / config.n_heads;
        int kv_dim = config.n_kv_heads * head_size;

        weights.rms_att_weight.resize(L);
        for (int i = 0; i < L; ++i) { weights.rms_att_weight[i] = ptr; ptr += d; }

        weights.wq.resize(L);
        for (int i = 0; i < L; ++i) { weights.wq[i] = ptr; ptr += static_cast<size_t>(d) * (config.n_heads * head_size); }

        weights.wk.resize(L);
        for (int i = 0; i < L; ++i) { weights.wk[i] = ptr; ptr += static_cast<size_t>(d) * kv_dim; }

        weights.wv.resize(L);
        for (int i = 0; i < L; ++i) { weights.wv[i] = ptr; ptr += static_cast<size_t>(d) * kv_dim; }

        weights.wo.resize(L);
        for (int i = 0; i < L; ++i) { weights.wo[i] = ptr; ptr += static_cast<size_t>(config.n_heads * head_size) * d; }

        weights.rms_ffn_weight.resize(L);
        for (int i = 0; i < L; ++i) { weights.rms_ffn_weight[i] = ptr; ptr += d; }

        weights.w1.resize(L);
        for (int i = 0; i < L; ++i) { weights.w1[i] = ptr; ptr += static_cast<size_t>(d) * hd; }

        weights.w2.resize(L);
        for (int i = 0; i < L; ++i) { weights.w2[i] = ptr; ptr += static_cast<size_t>(hd) * d; }

        weights.w3.resize(L);
        for (int i = 0; i < L; ++i) { weights.w3[i] = ptr; ptr += static_cast<size_t>(d) * hd; }

        weights.rms_final_weight = ptr; ptr += d;

        // Tentukan apakah classifier head (wcls) adalah shared dengan token embedding table
        // atau merupakan tensor terpisah (unshared). Kita cek sisa float yang tersisa di buffer.
        ptrdiff_t remaining = (data_buffer.data() + num_floats) - ptr;
        if (remaining == static_cast<ptrdiff_t>(config.vocab_size) * d) {
            weights.wcls = ptr; // Unshared classifier head
        } else {
            // Shared weights dengan token embedding (mayoritas model Llama/TinyStories)
            weights.wcls = weights.token_embedding_table;
        }

        std::cout << "[Model] Successfully loaded " << (file_size / (1024 * 1024)) << " MB of weights into memory.\n";
        return true;
    }
};

} // namespace lengine
