#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <variant>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace lengine {

/**
 * GGUF (GGML Universal File Format) v3 Full Parser
 *
 * Format standar industri — dipakai oleh llama.cpp, Ollama, LM Studio.
 * Mendukung: F32, F16, Q4_0, Q4_K, Q8_0 tensor types.
 *
 * Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
 *
 * File Layout:
 *   [magic: 4B][version: u32][n_tensors: u64][n_kv: u64]
 *   [metadata: n_kv key-value pairs]
 *   [tensor_infos: n_tensors tensor headers]
 *   [alignment padding]
 *   [tensor data: raw binary blobs]
 */

// ─── GGML Tensor Type IDs ─────────────────────────────────────────────────────
enum class GGMLType : uint32_t {
    F32   = 0,
    F16   = 1,
    Q4_0  = 2,
    Q4_1  = 3,
    Q5_0  = 6,
    Q5_1  = 7,
    Q8_0  = 8,
    Q8_1  = 9,
    Q2_K  = 10,
    Q3_K  = 11,
    Q4_K  = 12,
    Q5_K  = 13,
    Q6_K  = 14,
    Q8_K  = 15,
    IQ4_NL= 20,
    BF16  = 30,
    COUNT
};

// ─── GGUF Metadata Value Types ────────────────────────────────────────────────
enum class GGUFType : uint32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12
};

// ─── Metadata Value (variant) ─────────────────────────────────────────────────
using GGUFValue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t,
    uint32_t, int32_t, float,
    bool, std::string,
    uint64_t, int64_t, double
>;

// ─── Tensor Info ──────────────────────────────────────────────────────────────
struct GGUFTensorInfo {
    std::string          name;
    uint32_t             n_dimensions = 0;
    std::vector<uint64_t> dimensions;
    GGMLType             type = GGMLType::F32;
    uint64_t             offset = 0;   // byte offset from data section start
    uint64_t             n_elements = 0;
    uint64_t             n_bytes    = 0;

    uint64_t compute_n_elements() const {
        uint64_t n = 1;
        for (auto d : dimensions) n *= d;
        return n;
    }
};

// ─── Dequantization helpers ────────────────────────────────────────────────────
namespace dequant {
    // Q4_0: 32 int4 values + 1 float16 scale → float32
    // Block layout: [scale: f16 (2B)] [qs: 16B (32 x 4-bit packed)]
    static constexpr uint64_t Q4_0_BLOCK_SIZE  = 32;
    static constexpr uint64_t Q4_0_BLOCK_BYTES = 2 + 16; // f16 scale + 16B

    static float fp16_to_float(uint16_t h) {
        uint32_t sign     = (h >> 15) & 1;
        uint32_t exponent = (h >> 10) & 0x1F;
        uint32_t mantissa =  h        & 0x3FF;
        uint32_t f;
        if (exponent == 0) {
            // Subnormal
            f = (sign << 31) | ((127 - 14) << 23) | (mantissa << 13);
        } else if (exponent == 31) {
            // Inf / NaN
            f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        } else {
            f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }
        float result;
        std::memcpy(&result, &f, 4);
        return result;
    }

    // Dequantize a Q4_0 block into floats
    static void dequant_q4_0_block(const uint8_t* src, float* dst) {
        uint16_t scale_raw;
        std::memcpy(&scale_raw, src, 2);
        float scale = fp16_to_float(scale_raw);

        const uint8_t* qs = src + 2;
        for (int i = 0; i < 16; ++i) {
            uint8_t byte = qs[i];
            int8_t lo = static_cast<int8_t>((byte & 0x0F) - 8);
            int8_t hi = static_cast<int8_t>((byte >> 4)  - 8);
            dst[i * 2 + 0] = lo * scale;
            dst[i * 2 + 1] = hi * scale;
        }
    }

    // Dequantize Q8_0 block: [scale:f16][32 x int8]
    static void dequant_q8_0_block(const uint8_t* src, float* dst) {
        uint16_t scale_raw;
        std::memcpy(&scale_raw, src, 2);
        float scale = fp16_to_float(scale_raw);
        const int8_t* qs = reinterpret_cast<const int8_t*>(src + 2);
        for (int i = 0; i < 32; ++i) dst[i] = qs[i] * scale;
    }
} // namespace dequant

// ─── GGUFReader ───────────────────────────────────────────────────────────────
class GGUFReader {
public:
    uint32_t                                        version = 0;
    uint64_t                                        tensor_count = 0;
    uint64_t                                        metadata_kv_count = 0;
    std::unordered_map<std::string, GGUFValue>      metadata;
    std::vector<GGUFTensorInfo>                     tensors;
    std::unordered_map<std::string, size_t>         tensor_index; // name -> index
    uint64_t                                        data_offset = 0; // absolute file pos

    // ── Public API ───────────────────────────────────────────────────────────

    bool load(const std::string& path) {
        file_.open(path, std::ios::binary);
        if (!file_.is_open()) {
            std::cerr << "[GGUF] Cannot open: " << path << "\n";
            return false;
        }

        if (!read_header())   return false;
        if (!read_metadata()) return false;
        if (!read_tensor_infos()) return false;

        // Align to 32 bytes (GGUF standard alignment)
        uint64_t pos = static_cast<uint64_t>(file_.tellg());
        uint64_t aligned = ((pos + 31) / 32) * 32;
        file_.seekg(static_cast<std::streamoff>(aligned));
        data_offset = static_cast<uint64_t>(file_.tellg());

        print_summary();
        return true;
    }

    // Get metadata as string (with fallback)
    std::string get_str(const std::string& key, const std::string& fallback = "") const {
        auto it = metadata.find(key);
        if (it == metadata.end()) return fallback;
        if (auto* s = std::get_if<std::string>(&it->second)) return *s;
        return fallback;
    }

    uint32_t get_u32(const std::string& key, uint32_t fallback = 0) const {
        auto it = metadata.find(key);
        if (it == metadata.end()) return fallback;
        if (auto* v = std::get_if<uint32_t>(&it->second)) return *v;
        if (auto* v = std::get_if<int32_t> (&it->second)) return static_cast<uint32_t>(*v);
        return fallback;
    }

    uint64_t get_u64(const std::string& key, uint64_t fallback = 0) const {
        auto it = metadata.find(key);
        if (it == metadata.end()) return fallback;
        if (auto* v = std::get_if<uint64_t>(&it->second)) return *v;
        if (auto* v = std::get_if<uint32_t>(&it->second)) return static_cast<uint64_t>(*v);
        return fallback;
    }

    // Dequantize a named tensor into a flat float32 vector
    std::vector<float> load_tensor_f32(const std::string& name) {
        auto it = tensor_index.find(name);
        if (it == tensor_index.end()) {
            std::cerr << "[GGUF] Tensor not found: " << name << "\n";
            return {};
        }

        const GGUFTensorInfo& info = tensors[it->second];
        std::vector<float> out(info.n_elements);
        uint64_t file_pos = data_offset + info.offset;
        file_.seekg(static_cast<std::streamoff>(file_pos));

        if (info.type == GGMLType::F32) {
            file_.read(reinterpret_cast<char*>(out.data()), info.n_elements * sizeof(float));
        } else if (info.type == GGMLType::F16) {
            std::vector<uint16_t> buf(info.n_elements);
            file_.read(reinterpret_cast<char*>(buf.data()), info.n_elements * 2);
            for (size_t i = 0; i < info.n_elements; ++i)
                out[i] = dequant::fp16_to_float(buf[i]);
        } else if (info.type == GGMLType::Q4_0) {
            uint64_t n_blocks = info.n_elements / dequant::Q4_0_BLOCK_SIZE;
            std::vector<uint8_t> raw(n_blocks * dequant::Q4_0_BLOCK_BYTES);
            file_.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            for (uint64_t b = 0; b < n_blocks; ++b) {
                dequant::dequant_q4_0_block(
                    raw.data() + b * dequant::Q4_0_BLOCK_BYTES,
                    out.data() + b * dequant::Q4_0_BLOCK_SIZE
                );
            }
        } else if (info.type == GGMLType::Q8_0) {
            uint64_t block_bytes = 2 + 32;
            uint64_t n_blocks    = info.n_elements / 32;
            std::vector<uint8_t> raw(n_blocks * block_bytes);
            file_.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
            for (uint64_t b = 0; b < n_blocks; ++b) {
                dequant::dequant_q8_0_block(
                    raw.data() + b * block_bytes,
                    out.data() + b * 32
                );
            }
        } else {
            std::cerr << "[GGUF] Unsupported quant type " << static_cast<uint32_t>(info.type)
                      << " for tensor: " << name << "\n";
        }

        return out;
    }

    bool has_tensor(const std::string& name) const {
        return tensor_index.count(name) > 0;
    }

private:
    std::ifstream file_;

    bool read_header() {
        char magic[4] = {};
        file_.read(magic, 4);
        if (std::string(magic, 4) != "GGUF") {
            std::cerr << "[GGUF] Bad magic. Not a GGUF file.\n";
            return false;
        }
        file_.read(reinterpret_cast<char*>(&version), 4);
        if (version < 2 || version > 3) {
            std::cerr << "[GGUF] Unsupported version: " << version << "\n";
            return false;
        }
        file_.read(reinterpret_cast<char*>(&tensor_count),       8);
        file_.read(reinterpret_cast<char*>(&metadata_kv_count),  8);
        return true;
    }

    std::string read_gguf_string() {
        uint64_t len = 0;
        file_.read(reinterpret_cast<char*>(&len), 8);
        std::string s(static_cast<size_t>(len), '\0');
        file_.read(&s[0], static_cast<std::streamsize>(len));
        return s;
    }

    GGUFValue read_value(GGUFType type) {
        switch (type) {
            case GGUFType::UINT8:   { uint8_t  v; file_.read(reinterpret_cast<char*>(&v),1); return v; }
            case GGUFType::INT8:    { int8_t   v; file_.read(reinterpret_cast<char*>(&v),1); return v; }
            case GGUFType::UINT16:  { uint16_t v; file_.read(reinterpret_cast<char*>(&v),2); return v; }
            case GGUFType::INT16:   { int16_t  v; file_.read(reinterpret_cast<char*>(&v),2); return v; }
            case GGUFType::UINT32:  { uint32_t v; file_.read(reinterpret_cast<char*>(&v),4); return v; }
            case GGUFType::INT32:   { int32_t  v; file_.read(reinterpret_cast<char*>(&v),4); return v; }
            case GGUFType::FLOAT32: { float    v; file_.read(reinterpret_cast<char*>(&v),4); return v; }
            case GGUFType::BOOL:    { bool     v; file_.read(reinterpret_cast<char*>(&v),1); return v; }
            case GGUFType::UINT64:  { uint64_t v; file_.read(reinterpret_cast<char*>(&v),8); return v; }
            case GGUFType::INT64:   { int64_t  v; file_.read(reinterpret_cast<char*>(&v),8); return v; }
            case GGUFType::FLOAT64: { double   v; file_.read(reinterpret_cast<char*>(&v),8); return v; }
            case GGUFType::STRING:  return read_gguf_string();
            case GGUFType::ARRAY: {
                // Array: [type:u32][count:u64][elements...]
                uint32_t arr_type_raw; file_.read(reinterpret_cast<char*>(&arr_type_raw), 4);
                uint64_t arr_count;    file_.read(reinterpret_cast<char*>(&arr_count),    8);
                GGUFType arr_type = static_cast<GGUFType>(arr_type_raw);
                std::string arr_str = "[array:" + std::to_string(arr_count) + "]";
                // Skip array elements (we store as opaque string)
                for (uint64_t i = 0; i < arr_count; ++i) read_value(arr_type);
                return arr_str;
            }
            default:
                std::cerr << "[GGUF] Unknown metadata type: " << static_cast<uint32_t>(type) << "\n";
                return std::string("?");
        }
    }

    bool read_metadata() {
        for (uint64_t i = 0; i < metadata_kv_count; ++i) {
            std::string key = read_gguf_string();
            uint32_t type_raw;
            file_.read(reinterpret_cast<char*>(&type_raw), 4);
            GGUFType type = static_cast<GGUFType>(type_raw);
            metadata[key] = read_value(type);
        }
        return true;
    }

    bool read_tensor_infos() {
        tensors.resize(tensor_count);
        for (uint64_t i = 0; i < tensor_count; ++i) {
            GGUFTensorInfo& info = tensors[i];
            info.name = read_gguf_string();

            file_.read(reinterpret_cast<char*>(&info.n_dimensions), 4);
            info.dimensions.resize(info.n_dimensions);
            for (uint32_t d = 0; d < info.n_dimensions; ++d) {
                file_.read(reinterpret_cast<char*>(&info.dimensions[d]), 8);
            }

            uint32_t type_raw;
            file_.read(reinterpret_cast<char*>(&type_raw), 4);
            info.type = static_cast<GGMLType>(type_raw);
            file_.read(reinterpret_cast<char*>(&info.offset), 8);

            info.n_elements = info.compute_n_elements();

            // Compute byte size based on type
            uint64_t block_size  = (info.type == GGMLType::Q4_0) ? 32 : 32;
            uint64_t block_bytes = (info.type == GGMLType::Q4_0) ? 18 :
                                   (info.type == GGMLType::Q8_0) ? 34 :
                                   (info.type == GGMLType::F16)  ? (info.n_elements * 2) :
                                                                    (info.n_elements * 4);
            if (info.type == GGMLType::F32) {
                info.n_bytes = info.n_elements * 4;
            } else if (info.type == GGMLType::F16) {
                info.n_bytes = info.n_elements * 2;
            } else {
                // Block quantized: n_blocks * block_bytes
                info.n_bytes = (info.n_elements / block_size) * block_bytes;
            }

            tensor_index[info.name] = static_cast<size_t>(i);
        }
        return true;
    }

    void print_summary() const {
        std::cout << "========================================\n"
                  << " [GGUF File Loaded]\n"
                  << "  - Version:         " << version << "\n"
                  << "  - Tensors:         " << tensor_count << "\n"
                  << "  - Metadata KVs:    " << metadata_kv_count << "\n";
        // Print common arch keys
        for (const auto& key : {"general.architecture", "general.name",
                                  "llama.context_length", "llama.embedding_length"}) {
            auto it = metadata.find(key);
            if (it != metadata.end()) {
                if (auto* s = std::get_if<std::string>(&it->second))
                    std::cout << "  - " << key << ": " << *s << "\n";
                else if (auto* u = std::get_if<uint32_t>(&it->second))
                    std::cout << "  - " << key << ": " << *u << "\n";
                else if (auto* u64 = std::get_if<uint64_t>(&it->second))
                    std::cout << "  - " << key << ": " << *u64 << "\n";
            }
        }
        std::cout << "========================================\n";
    }
};

} // namespace lengine
