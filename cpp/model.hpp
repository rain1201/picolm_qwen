#ifndef MODEL_HPP
#define MODEL_HPP

#include "quant.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <fstream>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#endif

constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr uint32_t KVCACHE_MAGIC = 0x4B564350;
constexpr int MAX_LAYERS = 64;

struct ModelConfig {
    int n_embd = 0;
    int n_ffn = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int n_layers = 0;
    int vocab_size = 0;
    int max_seq_len = 2048;
    int head_dim = 0;
    float rope_freq_base = 10000.0f;
    int alignment = 32;
    GGUFType weight_type = GGUFType::F16;
};

struct LayerWeights {
    const void* attn_norm = nullptr;
    const void* attn_q = nullptr;
    const void* attn_k = nullptr;
    const void* attn_v = nullptr;
    const void* attn_q_b = nullptr;
    const void* attn_k_b = nullptr;
    const void* attn_v_b = nullptr;
    const void* attn_output_b = nullptr;
    const void* ffn_gate_b = nullptr;
    const void* ffn_down_b = nullptr;
    const void* ffn_up_b = nullptr;
    const void* attn_q_norm = nullptr;
    const void* attn_k_norm = nullptr;
    const void* attn_output = nullptr;
    const void* ffn_norm = nullptr;
    const void* ffn_gate = nullptr;
    const void* ffn_down = nullptr;
    const void* ffn_up = nullptr;
    GGUFType type_attn_norm = GGUFType::NONE;
    GGUFType type_attn_q = GGUFType::NONE;
    GGUFType type_attn_k = GGUFType::NONE;
    GGUFType type_attn_v = GGUFType::NONE;
    GGUFType type_attn_output = GGUFType::NONE;
    GGUFType type_ffn_norm = GGUFType::NONE;
    GGUFType type_ffn_gate = GGUFType::NONE;
    GGUFType type_ffn_down = GGUFType::NONE;
    GGUFType type_ffn_up = GGUFType::NONE;
    GGUFType type_attn_q_b = GGUFType::NONE;
    GGUFType type_attn_k_b = GGUFType::NONE;
    GGUFType type_attn_v_b = GGUFType::NONE;
    GGUFType type_attn_output_b = GGUFType::NONE;
    GGUFType type_ffn_gate_b = GGUFType::NONE;
    GGUFType type_ffn_down_b = GGUFType::NONE;
    GGUFType type_ffn_up_b = GGUFType::NONE;
    GGUFType type_attn_q_norm = GGUFType::NONE;
    GGUFType type_attn_k_norm = GGUFType::NONE;
};

struct ModelWeights {
    const void* token_embd = nullptr;
    GGUFType type_token_embd = GGUFType::NONE;
    const void* output_norm = nullptr;
    GGUFType type_output_norm = GGUFType::NONE;
    const void* output = nullptr;
    GGUFType type_output = GGUFType::NONE;
    LayerWeights layers[MAX_LAYERS];
};

struct RunState {
    std::vector<float> x, xb, xb2, q, hb, hb2, logits;
    std::vector<uint16_t> key_cache, val_cache;
    std::vector<float> dequant_scratch;
    std::vector<float> rope_cos, rope_sin;
    std::vector<float> norm_weights;
    std::vector<std::vector<float>> attn_norm_w, ffn_norm_w;
    std::vector<std::vector<float>> attn_q_bias, attn_k_bias, attn_v_bias, attn_output_bias;
    std::vector<std::vector<float>> ffn_gate_bias, ffn_up_bias, ffn_down_bias;
    std::vector<float> output_norm_w;
    std::vector<std::vector<float>> attn_q_norm_w, attn_k_norm_w;
    
    size_t mem_size = 0;
};

using RopeFn = void(*)(float*, float*, int, int, int, const float*, const float*);

struct Model {
    ModelConfig config;
    ModelWeights weights;
    RunState state;
    
    void* mmap_addr = nullptr;
    size_t mmap_size = 0;
    
    const void* tok_tokens_data = nullptr;
    uint64_t tok_n_tokens = 0;
    const void* tok_scores_data = nullptr;
    uint64_t tok_n_scores = 0;
    uint32_t tok_bos_id = 1;
    uint32_t tok_eos_id = 2;
    
    RopeFn rope_fn = nullptr;
    
#ifdef _WIN32
    HANDLE file_handle = nullptr;
    HANDLE map_handle = nullptr;
#else
    int fd = -1;
#endif

    ~Model() { free(); }

    int load(const char* path, int max_seq_len_override = 0) {
        weights = ModelWeights();  // Zero-initialize
        
        if (mmap_file(path) != 0) return -1;
        if (parse_gguf(max_seq_len_override) != 0) return -1;
        if (allocate_run_state() != 0) return -1;
        
        return 0;
    }

    float* forward(int token, int pos) {
        int dim = config.n_embd;
        int n_ffn = config.n_ffn;
        int n_heads = config.n_heads;
        int n_kv_heads = config.n_kv_heads;
        int head_dim = config.head_dim;
        int kv_dim = n_kv_heads * head_dim;
        int kv_mul = n_heads / n_kv_heads;
        int half_dim = head_dim / 2;
        int q_dim = n_heads * head_dim;

        const float* cos_pos = state.rope_cos.data() + (size_t)pos * half_dim;
        const float* sin_pos = state.rope_sin.data() + (size_t)pos * half_dim;

        // 1. Embedding lookup
        {
            size_t row_bytes = gguf_type_row_size(weights.type_token_embd, dim);
            const void* embd_row = (const uint8_t*)weights.token_embd + (size_t)token * row_bytes;
            dequantize_row(embd_row, state.x.data(), dim, weights.type_token_embd);
        }

        // 2. Transformer layers
        for (int l = 0; l < config.n_layers; l++) {
            LayerWeights* lw = &weights.layers[l];

            // Attention
            TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.attn_norm_w[l].data(), dim);
            
            // Q projection
            TensorOps::matmul_bias(state.q.data(), state.xb.data(), lw->attn_q, 
                                   state.attn_q_bias[l].data(), dim, q_dim, 
                                   lw->type_attn_q, lw->type_attn_q_b, state.dequant_scratch.data());
            
            // QK-Norm for Q (before RoPE)
            if (lw->attn_q_norm && state.attn_q_norm_w[l].data()) {
                for (int h = 0; h < n_heads; h++) {
                    float* qh = state.q.data() + h * head_dim;
                    TensorOps::rmsnorm(qh, qh, state.attn_q_norm_w[l].data(), head_dim);
                }
            }
            
            // K projection (use xb2 as temp buffer)
            TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), lw->attn_k,
                                   state.attn_k_bias[l].data(), dim, kv_dim,
                                   lw->type_attn_k, lw->type_attn_k_b, state.dequant_scratch.data());
            
            // QK-Norm for K (before RoPE)
            if (lw->attn_k_norm && state.attn_k_norm_w[l].data()) {
                for (int h = 0; h < n_kv_heads; h++) {
                    float* kh = state.xb2.data() + h * head_dim;
                    TensorOps::rmsnorm(kh, kh, state.attn_k_norm_w[l].data(), head_dim);
                }
            }

            // RoPE for Q and K
            rope_fn(state.q.data(), state.xb2.data(), head_dim, n_heads, n_kv_heads, cos_pos, sin_pos);

            // Store K as FP16
            uint16_t* kcache_layer = state.key_cache.data() + (size_t)l * config.max_seq_len * kv_dim;
            uint16_t* key_pos_fp16 = kcache_layer + (size_t)pos * kv_dim;
            for (int d = 0; d < kv_dim; d++) {
                key_pos_fp16[d] = fp32_to_fp16(state.xb2.data()[d]);
            }

            // V projection (reuse xb2 buffer after K is stored)
            TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), lw->attn_v,
                                   state.attn_v_bias[l].data(), dim, kv_dim,
                                   lw->type_attn_v, lw->type_attn_v_b, state.dequant_scratch.data());
            
            // Store V as FP16
            uint16_t* vcache_layer = state.val_cache.data() + (size_t)l * config.max_seq_len * kv_dim;
            uint16_t* val_pos_fp16 = vcache_layer + (size_t)pos * kv_dim;
            for (int d = 0; d < kv_dim; d++) {
                val_pos_fp16[d] = fp32_to_fp16(state.xb2.data()[d]);
            }

            // Flash Attention (online softmax)
            std::fill(state.xb.data(), state.xb.data() + q_dim, 0.0f);
            
            for (int h = 0; h < n_heads; h++) {
                float* qh = state.q.data() + h * head_dim;
                int kv_h = h / kv_mul;
                float* xbh = state.xb.data() + h * head_dim;

                float max_score = -1e30f;
                float sum_exp = 0.0f;
                std::vector<float> acc(head_dim, 0.0f);

                for (int t = 0; t <= pos; t++) {
                    const uint16_t* kt = kcache_layer + (size_t)t * kv_dim + kv_h * head_dim;
                    const uint16_t* vt = vcache_layer + (size_t)t * kv_dim + kv_h * head_dim;

                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        score += qh[d] * fp16_to_fp32(kt[d]);
                    }
                    score /= std::sqrt((float)head_dim);

                    if (score > max_score) {
                        float correction = std::exp(max_score - score);
                        sum_exp *= correction;
                        for (int d = 0; d < head_dim; d++) acc[d] *= correction;
                        sum_exp += 1.0f;
                        for (int d = 0; d < head_dim; d++) {
                            acc[d] += fp16_to_fp32(vt[d]);
                        }
                        max_score = score;
                    } else {
                        float w = std::exp(score - max_score);
                        sum_exp += w;
                        for (int d = 0; d < head_dim; d++) {
                            acc[d] += w * fp16_to_fp32(vt[d]);
                        }
                    }
                }

                for (int d = 0; d < head_dim; d++) {
                    xbh[d] = acc[d] / sum_exp;
                }
            }

            // Attention output projection: xb (attention output) -> xb2 (projected), then x += xb2
            TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), lw->attn_output,
                                   state.attn_output_bias[l].data(), q_dim, dim,
                                   lw->type_attn_output, lw->type_attn_output_b, state.dequant_scratch.data());
            
            TensorOps::vec_add(state.x.data(), state.xb2.data(), dim);

            // FFN
            TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.ffn_norm_w[l].data(), dim);
            TensorOps::matmul_bias(state.hb.data(), state.xb.data(), lw->ffn_gate,
                                   state.ffn_gate_bias[l].data(), dim, n_ffn,
                                   lw->type_ffn_gate, lw->type_ffn_gate_b, state.dequant_scratch.data());
            TensorOps::silu(state.hb.data(), n_ffn);
            TensorOps::matmul(state.hb2.data(), state.xb.data(), lw->ffn_up, dim, n_ffn, lw->type_ffn_up);
            TensorOps::elemwise_mul(state.hb.data(), state.hb.data(), state.hb2.data(), n_ffn);
            TensorOps::matmul_bias(state.xb.data(), state.hb.data(), lw->ffn_down,
                                   state.ffn_down_bias[l].data(), n_ffn, dim,
                                   lw->type_ffn_down, lw->type_ffn_down_b, state.dequant_scratch.data());
            TensorOps::vec_add(state.x.data(), state.xb.data(), dim);
        }

        // Final norm and output
        TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.output_norm_w.data(), dim);
        TensorOps::matmul(state.logits.data(), state.xb.data(), weights.output, 
                         dim, config.vocab_size, weights.type_output);

        return state.logits.data();
    }

    void free() {
        if (mmap_addr) {
#ifdef _WIN32
            UnmapViewOfFile(mmap_addr);
            if (map_handle) CloseHandle(map_handle);
            if (file_handle) CloseHandle(file_handle);
#else
            munmap(mmap_addr, mmap_size);
            if (fd >= 0) close(fd);
#endif
            mmap_addr = nullptr;
        }
    }

private:
    int mmap_file(const char* path) {
#ifdef _WIN32
        file_handle = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_handle == INVALID_HANDLE_VALUE) return -1;

        LARGE_INTEGER fsize;
        GetFileSizeEx(file_handle, &fsize);
        mmap_size = (size_t)fsize.QuadPart;

        map_handle = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map_handle) { CloseHandle(file_handle); return -1; }

        mmap_addr = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);
        if (!mmap_addr) { CloseHandle(map_handle); CloseHandle(file_handle); return -1; }
#else
        fd = open(path, O_RDONLY);
        if (fd < 0) return -1;

        struct stat st;
        fstat(fd, &st);
        mmap_size = (size_t)st.st_size;

        mmap_addr = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mmap_addr == MAP_FAILED) { close(fd); return -1; }
#endif
        return 0;
    }

    int parse_gguf(int max_seq_len_override) {
        const uint8_t* data = (const uint8_t*)mmap_addr;
        size_t pos = 0;

        uint32_t magic; std::memcpy(&magic, data + pos, 4); pos += 4;
        if (magic != GGUF_MAGIC) return -1;

        uint32_t version; std::memcpy(&version, data + pos, 4); pos += 4;
        if (version < 2 || version > 3) return -1;

        uint64_t n_tensors, n_metadata;
        std::memcpy(&n_tensors, data + pos, 8); pos += 8;
        std::memcpy(&n_metadata, data + pos, 8); pos += 8;

        config.rope_freq_base = 10000.0f;
        config.max_seq_len = 2048;
        config.head_dim = -1;

        // Parse metadata
        for (uint64_t i = 0; i < n_metadata; i++) {
            uint64_t key_len; std::memcpy(&key_len, data + pos, 8); pos += 8;
            if (pos + key_len > mmap_size) return -1;
            std::string key((const char*)data + pos, key_len); pos += key_len;

            uint32_t vtype; std::memcpy(&vtype, data + pos, 4); pos += 4;

            auto read_i32 = [&]() { int32_t v; std::memcpy(&v, data+pos, 4); pos += 4; return v; };
            auto read_f32 = [&]() { float v; std::memcpy(&v, data+pos, 4); pos += 4; return v; };
            auto read_u64 = [&]() { uint64_t v; std::memcpy(&v, data+pos, 8); pos += 8; return v; };

            auto skip_value = [&]() {
                switch (vtype) {
                    case 0: case 1: case 7: pos += 1; break;  // UINT8, INT8, BOOL
                    case 2: case 3: pos += 2; break;  // UINT16, INT16
                    case 4: case 5: pos += 4; break;  // UINT32, INT32
                    case 10: case 11: pos += 8; break;  // UINT64, INT64
                    case 6: pos += 4; break;  // FLOAT32
                    case 12: pos += 8; break;  // FLOAT64
                    case 8: {  // STRING
                        uint64_t slen = read_u64();
                        pos += slen;
                        break;
                    }
                    case 9: {  // ARRAY
                        uint32_t arr_type = read_i32();
                        uint64_t arr_len = read_u64();
                        for (uint64_t k = 0; k < arr_len; k++) {
                            uint32_t elem_type = arr_type;
                            switch (elem_type) {
                                case 0: case 1: case 7: pos += 1; break;
                                case 2: case 3: pos += 2; break;
                                case 4: case 5: pos += 4; break;
                                case 10: case 11: pos += 8; break;
                                case 6: pos += 4; break;
                                case 12: pos += 8; break;
                                case 8: { uint64_t sl = read_u64(); pos += sl; break; }
                                default: break;
                            }
                        }
                        break;
                    }
                    default: break;
                }
            };

            if (key.find("embedding_length") != std::string::npos) {
                config.n_embd = read_i32();
            } else if (key.find("feed_forward_length") != std::string::npos) {
                config.n_ffn = read_i32();
            } else if (key.find("attention.head_count") != std::string::npos &&
                       key.find("_kv") == std::string::npos) {
                config.n_heads = read_i32();
            } else if (key.find("attention.head_count_kv") != std::string::npos) {
                config.n_kv_heads = read_i32();
            } else if (key.find("block_count") != std::string::npos) {
                config.n_layers = read_i32();
            } else if (key.find("context_length") != std::string::npos) {
                config.max_seq_len = read_i32();
            } else if (key.find("rope.freq_base") != std::string::npos) {
                config.rope_freq_base = read_f32();
            } else if (key.find("general.alignment") != std::string::npos) {
                config.alignment = read_i32();
            } else if (key.find("tokenizer.ggml.bos_token_id") != std::string::npos) {
                tok_bos_id = (uint32_t)read_i32();
            } else if (key.find("tokenizer.ggml.eos_token_id") != std::string::npos) {
                tok_eos_id = (uint32_t)read_i32();
            } else if (key.find("tokenizer.ggml.tokens") != std::string::npos) {
                uint32_t arr_type = read_i32(); (void)arr_type;
                uint64_t arr_len = read_u64();
                tok_tokens_data = data + pos;
                tok_n_tokens = arr_len;
                // Skip string array
                for (uint64_t j = 0; j < arr_len; j++) {
                    uint64_t slen = read_u64();
                    pos += slen;
                }
            } else if (key.find("tokenizer.ggml.scores") != std::string::npos) {
                uint32_t arr_type = read_i32(); (void)arr_type;
                uint64_t arr_len = read_u64();
                tok_scores_data = data + pos;
                tok_n_scores = arr_len;
                // Skip float array
                pos += arr_len * 4;
            } else if (key.find("tokenizer.ggml.token_type") != std::string::npos) {
                // Skip token type array
                uint32_t arr_type = read_i32(); (void)arr_type;
                uint64_t arr_len = read_u64();
                pos += arr_len * 4;
            } else if (key.find("tokenizer.ggml.pre") != std::string::npos) {
                skip_value();
            } else if (key.find("attention.key_length") != std::string::npos) {
                config.head_dim = read_i32();
            } else {
                skip_value();
            }
        }

        if (tok_bos_id == 1 && config.vocab_size > 150000) {
            tok_bos_id = 151643;
            tok_eos_id = 151645;
        }

        // Limit max_seq_len for memory efficiency
        if (max_seq_len_override > 0 && max_seq_len_override < config.max_seq_len) {
            config.max_seq_len = max_seq_len_override;
        } else if (config.max_seq_len > 2048) {
            config.max_seq_len = 2048;  // Cap at 2048
        }
        if (config.head_dim < 1) config.head_dim = config.n_embd / config.n_heads;

        // Parse tensor info
        struct TensorInfo {
            std::string name;
            uint32_t n_dims;
            uint64_t dims[4];
            uint32_t type;
            uint64_t offset;
        };
        std::vector<TensorInfo> tinfos(n_tensors);

        for (uint64_t i = 0; i < n_tensors; i++) {
            uint64_t name_len; std::memcpy(&name_len, data+pos, 8); pos += 8;
            tinfos[i].name.assign((const char*)data+pos, name_len); pos += name_len;
            std::memcpy(&tinfos[i].n_dims, data+pos, 4); pos += 4;
            for (uint32_t d = 0; d < tinfos[i].n_dims; d++) {
                std::memcpy(&tinfos[i].dims[d], data+pos, 8); pos += 8;
            }
            std::memcpy(&tinfos[i].type, data+pos, 4); pos += 4;
            tinfos[i].type += 1;
            std::memcpy(&tinfos[i].offset, data+pos, 8); pos += 8;
        }

        size_t tensor_data_base = ((pos + config.alignment - 1) / config.alignment) * config.alignment;

        for (const auto& ti : tinfos) {
            const void* ptr = (const uint8_t*)mmap_addr + tensor_data_base + ti.offset;
            GGUFType qtype = (GGUFType)ti.type;

            if (ti.name == "token_embd.weight") {
                weights.token_embd = ptr; weights.type_token_embd = qtype;
            } else if (ti.name == "output_norm.weight") {
                weights.output_norm = ptr; weights.type_output_norm = qtype;
            } else if (ti.name == "output.weight") {
                weights.output = ptr; weights.type_output = qtype;
            } else if (ti.name.size() > 4 && ti.name.substr(0, 4) == "blk.") {
                size_t dot = ti.name.find('.', 4);
                if (dot != std::string::npos) {
                    int layer = std::stoi(ti.name.substr(4, dot - 4));
                    std::string suffix = ti.name.substr(dot + 1);
                    LayerWeights* lw = &weights.layers[layer];

                    if (suffix == "attn_norm.weight") { lw->attn_norm = ptr; lw->type_attn_norm = qtype; }
                    else if (suffix == "attn_q.weight") { lw->attn_q = ptr; lw->type_attn_q = qtype; }
                    else if (suffix == "attn_k.weight") { lw->attn_k = ptr; lw->type_attn_k = qtype; }
                    else if (suffix == "attn_v.weight") { lw->attn_v = ptr; lw->type_attn_v = qtype; }
                    else if (suffix == "attn_output.weight") { lw->attn_output = ptr; lw->type_attn_output = qtype; }
                    else if (suffix == "ffn_norm.weight") { lw->ffn_norm = ptr; lw->type_ffn_norm = qtype; }
                    else if (suffix == "ffn_gate.weight") { lw->ffn_gate = ptr; lw->type_ffn_gate = qtype; }
                    else if (suffix == "ffn_down.weight") { lw->ffn_down = ptr; lw->type_ffn_down = qtype; }
                    else if (suffix == "ffn_up.weight") { lw->ffn_up = ptr; lw->type_ffn_up = qtype; }
                    else if (suffix == "attn_q.bias") { lw->attn_q_b = ptr; lw->type_attn_q_b = qtype; }
                    else if (suffix == "attn_k.bias") { lw->attn_k_b = ptr; lw->type_attn_k_b = qtype; }
                    else if (suffix == "attn_v.bias") { lw->attn_v_b = ptr; lw->type_attn_v_b = qtype; }
                    else if (suffix == "attn_output.bias") { lw->attn_output_b = ptr; lw->type_attn_output_b = qtype; }
                    else if (suffix == "ffn_gate.bias") { lw->ffn_gate_b = ptr; lw->type_ffn_gate_b = qtype; }
                    else if (suffix == "ffn_down.bias") { lw->ffn_down_b = ptr; lw->type_ffn_down_b = qtype; }
                    else if (suffix == "ffn_up.bias") { lw->ffn_up_b = ptr; lw->type_ffn_up_b = qtype; }
                    else if (suffix == "attn_q_norm.weight") { lw->attn_q_norm = ptr; lw->type_attn_q_norm = qtype; }
                    else if (suffix == "attn_k_norm.weight") { lw->attn_k_norm = ptr; lw->type_attn_k_norm = qtype; }
                }
            }
        }

        if (!weights.output) {
            weights.output = weights.token_embd;
            weights.type_output = weights.type_token_embd;
        }

        if (config.vocab_size == 0) {
            for (const auto& ti : tinfos) {
                if (ti.name == "token_embd.weight" && ti.n_dims >= 2) {
                    config.vocab_size = (ti.dims[0] == (uint64_t)config.n_embd) ? (int)ti.dims[1] : (int)ti.dims[0];
                    break;
                }
            }
        }
        if (config.vocab_size == 0 && tok_n_tokens > 0) {
            config.vocab_size = (int)tok_n_tokens;
        }

        config.weight_type = weights.layers[0].type_attn_q;

        fprintf(stderr, "Model config:\n");
        fprintf(stderr, "  n_embd=%d, n_ffn=%d, n_heads=%d, n_kv_heads=%d\n",
                config.n_embd, config.n_ffn, config.n_heads, config.n_kv_heads);
        fprintf(stderr, "  n_layers=%d, vocab_size=%d, max_seq=%d\n",
                config.n_layers, config.vocab_size, config.max_seq_len);
        fprintf(stderr, "  head_dim=%d, rope_base=%.1f\n", config.head_dim, config.rope_freq_base);

        return 0;
    }

    int allocate_run_state() {
        int dim = config.n_embd;
        int n_ffn = config.n_ffn;
        int q_dim = config.n_heads * config.head_dim;
        int kv_dim = config.n_kv_heads * config.head_dim;
        int half_dim = config.head_dim / 2;
        int max_scratch = std::max({dim, q_dim, n_ffn, config.vocab_size});

        state.x.resize((size_t)max_scratch);
        state.xb.resize((size_t)max_scratch);
        state.xb2.resize((size_t)max_scratch);
        state.q.resize((size_t)q_dim);
        state.hb.resize((size_t)n_ffn);
        state.hb2.resize((size_t)n_ffn);
        state.logits.resize((size_t)config.vocab_size);
        state.dequant_scratch.resize((size_t)max_scratch);
        state.rope_cos.resize((size_t)config.max_seq_len * half_dim);
        state.rope_sin.resize((size_t)config.max_seq_len * half_dim);

        // Norm weights - use flat arrays for efficiency
        state.attn_norm_w.resize((size_t)config.n_layers);
        state.ffn_norm_w.resize((size_t)config.n_layers);
        state.output_norm_w.resize((size_t)dim);

        for (int l = 0; l < config.n_layers; l++) {
            state.attn_norm_w[l].resize((size_t)dim);
            state.ffn_norm_w[l].resize((size_t)dim);
            if (weights.layers[l].type_attn_norm != GGUFType::NONE) {
                dequantize_row(weights.layers[l].attn_norm, state.attn_norm_w[l].data(), dim, weights.layers[l].type_attn_norm);
            }
            if (weights.layers[l].type_ffn_norm != GGUFType::NONE) {
                dequantize_row(weights.layers[l].ffn_norm, state.ffn_norm_w[l].data(), dim, weights.layers[l].type_ffn_norm);
            }
        }
        if (weights.type_output_norm != GGUFType::NONE) {
            dequantize_row(weights.output_norm, state.output_norm_w.data(), dim, weights.type_output_norm);
        }

        // Biases - use flat arrays
        state.attn_q_bias.resize((size_t)config.n_layers);
        state.attn_k_bias.resize((size_t)config.n_layers);
        state.attn_v_bias.resize((size_t)config.n_layers);
        state.attn_output_bias.resize((size_t)config.n_layers);
        state.ffn_gate_bias.resize((size_t)config.n_layers);
        state.ffn_up_bias.resize((size_t)config.n_layers);
        state.ffn_down_bias.resize((size_t)config.n_layers);

        for (int l = 0; l < config.n_layers; l++) {
            state.attn_q_bias[l].resize((size_t)q_dim);
            state.attn_k_bias[l].resize((size_t)kv_dim);
            state.attn_v_bias[l].resize((size_t)kv_dim);
            state.attn_output_bias[l].resize((size_t)dim);
            state.ffn_gate_bias[l].resize((size_t)n_ffn);
            state.ffn_up_bias[l].resize((size_t)n_ffn);
            state.ffn_down_bias[l].resize((size_t)dim);

            LayerWeights* lw = &weights.layers[l];
            if (lw->attn_q_b && lw->type_attn_q_b != GGUFType::NONE)
                dequantize_row(lw->attn_q_b, state.attn_q_bias[l].data(), q_dim, lw->type_attn_q_b);
            if (lw->attn_k_b && lw->type_attn_k_b != GGUFType::NONE)
                dequantize_row(lw->attn_k_b, state.attn_k_bias[l].data(), kv_dim, lw->type_attn_k_b);
            if (lw->attn_v_b && lw->type_attn_v_b != GGUFType::NONE)
                dequantize_row(lw->attn_v_b, state.attn_v_bias[l].data(), kv_dim, lw->type_attn_v_b);
            if (lw->attn_output_b && lw->type_attn_output_b != GGUFType::NONE)
                dequantize_row(lw->attn_output_b, state.attn_output_bias[l].data(), dim, lw->type_attn_output_b);
            if (lw->ffn_gate_b && lw->type_ffn_gate_b != GGUFType::NONE)
                dequantize_row(lw->ffn_gate_b, state.ffn_gate_bias[l].data(), n_ffn, lw->type_ffn_gate_b);
            if (lw->ffn_up_b && lw->type_ffn_up_b != GGUFType::NONE)
                dequantize_row(lw->ffn_up_b, state.ffn_up_bias[l].data(), n_ffn, lw->type_ffn_up_b);
            if (lw->ffn_down_b && lw->type_ffn_down_b != GGUFType::NONE)
                dequantize_row(lw->ffn_down_b, state.ffn_down_bias[l].data(), dim, lw->type_ffn_down_b);
        }

        // QK-Norm weights
        state.attn_q_norm_w.resize((size_t)config.n_layers);
        state.attn_k_norm_w.resize((size_t)config.n_layers);
        for (int l = 0; l < config.n_layers; l++) {
            state.attn_q_norm_w[l].resize((size_t)config.head_dim);
            state.attn_k_norm_w[l].resize((size_t)config.head_dim);
            if (weights.layers[l].attn_q_norm && weights.layers[l].type_attn_q_norm != GGUFType::NONE) {
                dequantize_row(weights.layers[l].attn_q_norm, state.attn_q_norm_w[l].data(), config.head_dim, weights.layers[l].type_attn_q_norm);
            }
            if (weights.layers[l].attn_k_norm && weights.layers[l].type_attn_k_norm != GGUFType::NONE) {
                dequantize_row(weights.layers[l].attn_k_norm, state.attn_k_norm_w[l].data(), config.head_dim, weights.layers[l].type_attn_k_norm);
            }
        }

        // KV Cache - use config.max_seq_len for proper indexing
        size_t kv_elements = (size_t)config.n_layers * (size_t)config.max_seq_len * (size_t)kv_dim;
        state.key_cache.resize(kv_elements);
        state.val_cache.resize(kv_elements);

        // RoPE tables
        for (int pos = 0; pos < config.max_seq_len; pos++) {
            float* cos_row = state.rope_cos.data() + (size_t)pos * half_dim;
            float* sin_row = state.rope_sin.data() + (size_t)pos * half_dim;
            for (int i = 0; i < half_dim; i++) {
                float theta = (float)pos / std::pow(config.rope_freq_base, (float)(2 * i) / (float)config.head_dim);
                cos_row[i] = std::cos(theta);
                sin_row[i] = std::sin(theta);
            }
        }

        state.mem_size = state.x.capacity() * sizeof(float) * 8 +
                        state.rope_cos.capacity() * sizeof(float) * 2 +
                        kv_elements * sizeof(uint16_t) * 2;

        fprintf(stderr, "Allocating %.2f MB for runtime state\n", state.mem_size / (1024.0 * 1024.0));
        return 0;
    }
};

#endif // MODEL_HPP
