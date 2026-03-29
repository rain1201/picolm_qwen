#ifndef MODEL_HPP
#define MODEL_HPP

#include "quant.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"
#include "prefetcher.hpp"
#include "gguf_metadata.hpp"
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <fstream>
#include <unordered_map>
#include <memory>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <memoryapi.h>
#else
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

constexpr uint32_t GGUF_MAGIC = 0x46554747;
constexpr int MAX_LAYERS = 64;

// 配置键名常量
namespace cfg {
    constexpr const char* N_EMBD = "n_embd";
    constexpr const char* N_FFN = "n_ffn";
    constexpr const char* N_HEADS = "n_heads";
    constexpr const char* N_KV_HEADS = "n_kv_heads";
    constexpr const char* N_LAYERS = "n_layers";
    constexpr const char* VOCAB_SIZE = "vocab_size";
    constexpr const char* MAX_SEQ_LEN = "max_seq_len";
    constexpr const char* HEAD_DIM = "head_dim";
    constexpr const char* ROPE_FREQ_BASE = "rope_freq_base";
    constexpr const char* ALIGNMENT = "alignment";
    constexpr const char* WEIGHT_TYPE = "weight_type";
    constexpr const char* USE_PREFETCH = "use_prefetch";
    constexpr const char* USE_EVICT = "use_evict";
}

// 辅助函数声明（在 gguf_metadata.hpp 中定义）
using parser::getInt;
using parser::getFloat;
using parser::getUint;
using parser::getBool;

// 层权重键名常量
namespace layer {
    constexpr const char* ATTN_NORM = "attn_norm";
    constexpr const char* FFN_NORM = "ffn_norm";
    constexpr const char* ATTN_Q = "attn_q";
    constexpr const char* ATTN_K = "attn_k";
    constexpr const char* ATTN_V = "attn_v";
    constexpr const char* ATTN_OUTPUT = "attn_output";
    constexpr const char* ATTN_Q_B = "attn_q_b";
    constexpr const char* ATTN_K_B = "attn_k_b";
    constexpr const char* ATTN_V_B = "attn_v_b";
    constexpr const char* ATTN_OUTPUT_B = "attn_output_b";
    constexpr const char* ATTN_Q_NORM = "attn_q_norm";
    constexpr const char* ATTN_K_NORM = "attn_k_norm";
    constexpr const char* FFN_GATE = "ffn_gate";
    constexpr const char* FFN_DOWN = "ffn_down";
    constexpr const char* FFN_UP = "ffn_up";
    constexpr const char* FFN_GATE_B = "ffn_gate_b";
    constexpr const char* FFN_DOWN_B = "ffn_down_b";
    constexpr const char* FFN_UP_B = "ffn_up_b";
}

// Prefetch 权重列表（按访问顺序）
inline const std::vector<std::pair<const char*, bool>> PREFETCH_KEYS = {
    {layer::ATTN_Q,       false},  // attn_size
    {layer::ATTN_K,       false},  // attn_size
    {layer::ATTN_V,       false},  // attn_size
    {layer::ATTN_OUTPUT,  false},  // attn_size
    {layer::FFN_GATE,     true},   // ffn_size
    {layer::FFN_UP,       true},   // ffn_size
    {layer::FFN_DOWN,     true},   // ffn_size
};

// Evict 权重列表（按驱逐顺序）
inline const std::vector<std::pair<const char*, bool>> EVICT_KEYS = {
    {layer::ATTN_Q,       false},  // attn_size
    {layer::ATTN_K,       false},  // kv_size
    {layer::ATTN_V,       false},  // kv_size
    {layer::ATTN_OUTPUT,  false},  // q_dim * dim
    {layer::FFN_GATE,     true},   // ffn_size
    {layer::FFN_UP,       true},   // ffn_size
    {layer::FFN_DOWN,     true},   // n_ffn * dim
};

// 输出层权重键名常量
namespace output {
    constexpr const char* TOKEN_EMBD = "token_embd";
    constexpr const char* OUTPUT_NORM = "output_norm";
    constexpr const char* OUTPUT = "output";
}

// ============================================================================
// 模型配置（直接读写 map）
// ============================================================================
struct ModelConfig {
    metadata::ModelConfig config;
    
    // 访问底层 map
    auto& ints() { return config.ints; }
    auto& floats() { return config.floats; }
    auto& uints() { return config.uints; }
    auto& metadata() { return config.metadata; }
    const auto& ints() const { return config.ints; }
    const auto& floats() const { return config.floats; }
    const auto& uints() const { return config.uints; }
    const auto& metadata() const { return config.metadata; }
};

// ============================================================================
// 层权重（完全基于 map）
// ============================================================================
using LayerWeights = metadata::LayerWeights;

// ============================================================================
// 模型权重（完全基于 map）
// ============================================================================
struct ModelWeights {
    metadata::ModelWeightMap weights;
    LayerWeights layers[MAX_LAYERS];
    
    void set(const std::string& name, const void* ptr, uint32_t type) {
        if (ptr) {
            weights[name] = metadata::WeightEntry{ptr, type};
        }
    }
    
    metadata::WeightEntry get(const std::string& name) const {
        auto it = weights.find(name);
        return (it != weights.end()) ? it->second : metadata::WeightEntry{};
    }
    
    LayerWeights& get_layer(int layer_id) {
        return layers[layer_id];
    }
    
    const LayerWeights& get_layer(int layer_id) const {
        return layers[layer_id];
    }
};

// ============================================================================
// 运行状态
// ============================================================================
struct RunState {
    std::vector<float> x, xb, xb2, q, hb, hb2, logits;
    std::vector<uint16_t> key_cache, val_cache;
    std::vector<float> dequant_scratch;
    std::vector<float> rope_cos, rope_sin;
    std::vector<float> norm_weights;
    
    // 按层存储的归一化权重和偏置
    std::vector<std::vector<float>> attn_norm_w, ffn_norm_w;
    std::vector<std::vector<float>> attn_q_bias, attn_k_bias, attn_v_bias, attn_output_bias;
    std::vector<std::vector<float>> ffn_gate_bias, ffn_up_bias, ffn_down_bias;
    std::vector<float> output_norm_w;
    std::vector<std::vector<float>> attn_q_norm_w, attn_k_norm_w;
    
    float acc[512];
    size_t mem_size = 0;
};

using RopeFn = void(*)(float*, float*, int, int, int, const float*, const float*);

// ============================================================================
// 模型主类
// ============================================================================
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

    int load(const char* path, int max_seq_len_override = 0, bool prefetch = false, bool evict = false) {
        config.ints()[cfg::USE_PREFETCH] = prefetch ? 1 : 0;
        config.ints()[cfg::USE_EVICT] = evict ? 1 : 0;

        if (mmap_file(path) != 0) {
            fprintf(stderr, "[Load Error] mmap_file failed.\n");
            return -1;
        }
        if (parse_gguf(max_seq_len_override) != 0) {
            fprintf(stderr, "[Load Error] parse_gguf failed. (Check if it is a valid GGUF file)\n");
            return -1;
        }
        if (allocate_run_state() != 0) {
            fprintf(stderr, "[Load Error] allocate_run_state failed.\n");
            return -1;
        }

        return 0;
    }

    void prefetch(int l) {
        if (!getBool(config.ints(), cfg::USE_PREFETCH) || l + 1 >= getInt(config.ints(), cfg::N_LAYERS)) {
            return;
        }

        int dim = getInt(config.ints(), cfg::N_EMBD);
        int n_ffn = getInt(config.ints(), cfg::N_FFN);
        int head_dim = getInt(config.ints(), cfg::HEAD_DIM);
        int n_heads = getInt(config.ints(), cfg::N_HEADS);
        int q_dim = n_heads * head_dim;

        LayerWeights& next_lw = weights.get_layer(l + 1);
        size_t attn_size = (size_t)gguf_type_row_size((GGUFType)next_lw.get(layer::ATTN_Q).type, dim) * q_dim;
        size_t ffn_size = (size_t)gguf_type_row_size((GGUFType)next_lw.get(layer::FFN_GATE).type, dim) * n_ffn;

        for (const auto& [key, use_ffn_size] : PREFETCH_KEYS) {
            size_t size = use_ffn_size ? ffn_size : attn_size;
            MemoryPrefetcher::prefetch_layer(next_lw.get(key).ptr, size);
        }
    }

    void evict(int l) {
        if (!getBool(config.ints(), cfg::USE_EVICT)) {
            return;
        }

        LayerWeights& lw = weights.get_layer(l);
        int dim = getInt(config.ints(), cfg::N_EMBD);
        int n_ffn = getInt(config.ints(), cfg::N_FFN);
        int head_dim = getInt(config.ints(), cfg::HEAD_DIM);
        int n_heads = getInt(config.ints(), cfg::N_HEADS);
        int q_dim = n_heads * head_dim;
        int kv_dim = getInt(config.ints(), cfg::N_KV_HEADS) * head_dim;

        auto evict_tensor = [](const void* ptr, size_t size) {
            if (!ptr || size == 0) return;
#ifdef _WIN32
            VirtualUnlock((LPVOID)ptr, size);
#else
            madvise((void*)ptr, size, MADV_DONTNEED);
#endif
        };

        // 计算各权重的大小
        auto get_size = [&](const char* key) -> size_t {
            auto entry = lw.get(key);
            if (strcmp(key, layer::ATTN_Q) == 0) return gguf_type_row_size((GGUFType)entry.type, dim) * q_dim;
            if (strcmp(key, layer::ATTN_K) == 0) return gguf_type_row_size((GGUFType)entry.type, dim) * kv_dim;
            if (strcmp(key, layer::ATTN_V) == 0) return gguf_type_row_size((GGUFType)entry.type, dim) * kv_dim;
            if (strcmp(key, layer::ATTN_OUTPUT) == 0) return gguf_type_row_size((GGUFType)entry.type, q_dim) * dim;
            if (strcmp(key, layer::FFN_GATE) == 0) return gguf_type_row_size((GGUFType)entry.type, dim) * n_ffn;
            if (strcmp(key, layer::FFN_UP) == 0) return gguf_type_row_size((GGUFType)entry.type, dim) * n_ffn;
            if (strcmp(key, layer::FFN_DOWN) == 0) return gguf_type_row_size((GGUFType)entry.type, n_ffn) * dim;
            return (size_t)0;
        };

        for (const auto& [key, _] : EVICT_KEYS) {
            evict_tensor(lw.get(key).ptr, get_size(key));
        }
    }

    float* forward(int token, int pos) {
        int dim = getInt(config.ints(), cfg::N_EMBD);
        int n_ffn = getInt(config.ints(), cfg::N_FFN);
        int n_heads = getInt(config.ints(), cfg::N_HEADS);
        int n_kv_heads = getInt(config.ints(), cfg::N_KV_HEADS);
        int head_dim = getInt(config.ints(), cfg::HEAD_DIM);
        int kv_dim = n_kv_heads * head_dim;
        int kv_mul = n_heads / n_kv_heads;
        int half_dim = head_dim / 2;
        int q_dim = n_heads * head_dim;

        const float* cos_pos = state.rope_cos.data() + (size_t)pos * half_dim;
        const float* sin_pos = state.rope_sin.data() + (size_t)pos * half_dim;

        // 1. Embedding lookup
        {
            auto embd_entry = weights.get(output::TOKEN_EMBD);
            size_t row_bytes = gguf_type_row_size((GGUFType)embd_entry.type, dim);
            const void* embd_row = (const uint8_t*)embd_entry.ptr + (size_t)token * row_bytes;
            dequantize_row(embd_row, state.x.data(), dim, (GGUFType)embd_entry.type);
        }

        // 2. Transformer layers
        for (int l = 0; l < getInt(config.ints(), cfg::N_LAYERS); l++) {
            LayerWeights& lw = weights.get_layer(l);

            prefetch(l);

            // Attention norm
            TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.attn_norm_w[l].data(), dim);

            // Q projection
            auto q_entry = lw.get(layer::ATTN_Q);
            auto q_b_entry = lw.get(layer::ATTN_Q_B);
            TensorOps::matmul_bias(state.q.data(), state.xb.data(), q_entry.ptr,
                                   state.attn_q_bias[l].data(), dim, q_dim,
                                   (GGUFType)q_entry.type, (GGUFType)q_b_entry.type, state.dequant_scratch.data());

            // QK-Norm for Q
            auto q_norm_entry = lw.get(layer::ATTN_Q_NORM);
            if (q_norm_entry && state.attn_q_norm_w[l].data()) {
                for (int h = 0; h < n_heads; h++) {
                    float* qh = state.q.data() + h * head_dim;
                    TensorOps::rmsnorm(qh, qh, state.attn_q_norm_w[l].data(), head_dim);
                }
            }

            // K projection
            auto k_entry = lw.get(layer::ATTN_K);
            auto k_b_entry = lw.get(layer::ATTN_K_B);
            TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), k_entry.ptr,
                                   state.attn_k_bias[l].data(), dim, kv_dim,
                                   (GGUFType)k_entry.type, (GGUFType)k_b_entry.type, state.dequant_scratch.data());

            // QK-Norm for K
            auto k_norm_entry = lw.get(layer::ATTN_K_NORM);
            if (k_norm_entry && state.attn_k_norm_w[l].data()) {
                for (int h = 0; h < n_kv_heads; h++) {
                    float* kh = state.xb2.data() + h * head_dim;
                    TensorOps::rmsnorm(kh, kh, state.attn_k_norm_w[l].data(), head_dim);
                }
            }

            // RoPE for Q and K
            rope_fn(state.q.data(), state.xb2.data(), head_dim, n_heads, n_kv_heads, cos_pos, sin_pos);

            // Store K as FP16
            uint16_t* kcache_layer = state.key_cache.data() + (size_t)l * getInt(config.ints(), cfg::MAX_SEQ_LEN) * kv_dim;
            uint16_t* key_pos_fp16 = kcache_layer + (size_t)pos * kv_dim;
            for (int d = 0; d < kv_dim; d++) {
                key_pos_fp16[d] = fp32_to_fp16(state.xb2.data()[d]);
            }

            // V projection
            auto v_entry = lw.get(layer::ATTN_V);
            auto v_b_entry = lw.get(layer::ATTN_V_B);
            TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), v_entry.ptr,
                                   state.attn_v_bias[l].data(), dim, kv_dim,
                                   (GGUFType)v_entry.type, (GGUFType)v_b_entry.type, state.dequant_scratch.data());

            // Store V as FP16
            uint16_t* vcache_layer = state.val_cache.data() + (size_t)l * getInt(config.ints(), cfg::MAX_SEQ_LEN) * kv_dim;
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
                std::fill(state.acc, state.acc + head_dim, 0.0f);

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
                        for (int d = 0; d < head_dim; d++) state.acc[d] *= correction;
                        sum_exp += 1.0f;
                        for (int d = 0; d < head_dim; d++) {
                            state.acc[d] += fp16_to_fp32(vt[d]);
                        }
                        max_score = score;
                    } else {
                        float w = std::exp(score - max_score);
                        sum_exp += w;
                        for (int d = 0; d < head_dim; d++) {
                            state.acc[d] += w * fp16_to_fp32(vt[d]);
                        }
                    }
                }

                for (int d = 0; d < head_dim; d++) {
                    xbh[d] = state.acc[d] / sum_exp;
                }
            }

            // Attention output projection
            auto attn_out_entry = lw.get(layer::ATTN_OUTPUT);
            auto attn_out_b_entry = lw.get(layer::ATTN_OUTPUT_B);
            TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), attn_out_entry.ptr,
                                   state.attn_output_bias[l].data(), q_dim, dim,
                                   (GGUFType)attn_out_entry.type, (GGUFType)attn_out_b_entry.type, state.dequant_scratch.data());

            TensorOps::vec_add(state.x.data(), state.xb2.data(), dim);

            // FFN
            TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.ffn_norm_w[l].data(), dim);

            auto gate_entry = lw.get(layer::FFN_GATE);
            auto gate_b_entry = lw.get(layer::FFN_GATE_B);
            TensorOps::matmul_bias(state.hb.data(), state.xb.data(), gate_entry.ptr,
                                   state.ffn_gate_bias[l].data(), dim, n_ffn,
                                   (GGUFType)gate_entry.type, (GGUFType)gate_b_entry.type, state.dequant_scratch.data());
            TensorOps::silu(state.hb.data(), n_ffn);
            
            auto up_entry = lw.get(layer::FFN_UP);
            TensorOps::matmul(state.hb2.data(), state.xb.data(), up_entry.ptr, dim, n_ffn, (GGUFType)up_entry.type);
            TensorOps::elemwise_mul(state.hb.data(), state.hb.data(), state.hb2.data(), n_ffn);

            auto down_entry = lw.get(layer::FFN_DOWN);
            auto down_b_entry = lw.get(layer::FFN_DOWN_B);
            TensorOps::matmul_bias(state.xb.data(), state.hb.data(), down_entry.ptr,
                                   state.ffn_down_bias[l].data(), n_ffn, dim,
                                   (GGUFType)down_entry.type, (GGUFType)down_b_entry.type, state.dequant_scratch.data());
            TensorOps::vec_add(state.x.data(), state.xb.data(), dim);

            evict(l);
        }

        // Final norm and output
        TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.output_norm_w.data(), dim);
        auto output_entry = weights.get(output::OUTPUT);
        TensorOps::matmul(state.logits.data(), state.xb.data(), output_entry.ptr,
                         dim, getInt(config.ints(), cfg::VOCAB_SIZE), (GGUFType)output_entry.type);

        return state.logits.data();
    }

    float* forward_batch(const std::vector<int>& tokens, int start_pos) {
        int num_tokens = (int)tokens.size();
        if (num_tokens == 0) return nullptr;

        int dim = getInt(config.ints(), cfg::N_EMBD);
        int n_ffn = getInt(config.ints(), cfg::N_FFN);
        int n_heads = getInt(config.ints(), cfg::N_HEADS);
        int n_kv_heads = getInt(config.ints(), cfg::N_KV_HEADS);
        int head_dim = getInt(config.ints(), cfg::HEAD_DIM);
        int kv_dim = n_kv_heads * head_dim;
        int kv_mul = n_heads / n_kv_heads;
        int half_dim = head_dim / 2;
        int q_dim = n_heads * head_dim;

        std::vector<float> batch_x((size_t)num_tokens * dim, 0.0f);

        // 1. Embedding lookup
        {
            auto embd_entry = weights.get(output::TOKEN_EMBD);
            size_t row_bytes = gguf_type_row_size((GGUFType)embd_entry.type, dim);
            for (int i = 0; i < num_tokens; i++) {
                const void* embd_row = (const uint8_t*)embd_entry.ptr + (size_t)tokens[i] * row_bytes;
                dequantize_row(embd_row, batch_x.data() + (size_t)i * dim, dim, (GGUFType)embd_entry.type);
            }
        }

        // 2. Transformer layers
        for (int l = 0; l < getInt(config.ints(), cfg::N_LAYERS); l++) {
            LayerWeights& lw = weights.get_layer(l);
            prefetch(l);

            for (int i = 0; i < num_tokens; i++) {
                int pos = start_pos + i;
                std::memcpy(state.x.data(), batch_x.data() + (size_t)i * dim, dim * sizeof(float));

                // Attention norm
                TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.attn_norm_w[l].data(), dim);

                // Q, K, V
                auto q_entry = lw.get(layer::ATTN_Q);
                auto q_b_entry = lw.get(layer::ATTN_Q_B);
                TensorOps::matmul_bias(state.q.data(), state.xb.data(), q_entry.ptr,
                                       state.attn_q_bias[l].data(), dim, q_dim,
                                       (GGUFType)q_entry.type, (GGUFType)q_b_entry.type, state.dequant_scratch.data());

                auto q_norm_entry = lw.get(layer::ATTN_Q_NORM);
                if (q_norm_entry && state.attn_q_norm_w[l].data()) {
                    for (int h = 0; h < n_heads; h++) {
                        float* qh = state.q.data() + h * head_dim;
                        TensorOps::rmsnorm(qh, qh, state.attn_q_norm_w[l].data(), head_dim);
                    }
                }

                auto k_entry = lw.get(layer::ATTN_K);
                auto k_b_entry = lw.get(layer::ATTN_K_B);
                TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), k_entry.ptr,
                                       state.attn_k_bias[l].data(), dim, kv_dim,
                                       (GGUFType)k_entry.type, (GGUFType)k_b_entry.type, state.dequant_scratch.data());

                auto k_norm_entry = lw.get(layer::ATTN_K_NORM);
                if (k_norm_entry && state.attn_k_norm_w[l].data()) {
                    for (int h = 0; h < n_kv_heads; h++) {
                        float* kh = state.xb2.data() + h * head_dim;
                        TensorOps::rmsnorm(kh, kh, state.attn_k_norm_w[l].data(), head_dim);
                    }
                }

                const float* cos_pos = state.rope_cos.data() + (size_t)pos * half_dim;
                const float* sin_pos = state.rope_sin.data() + (size_t)pos * half_dim;
                rope_fn(state.q.data(), state.xb2.data(), head_dim, n_heads, n_kv_heads, cos_pos, sin_pos);

                // Store K, V
                uint16_t* kcache_layer = state.key_cache.data() + (size_t)l * getInt(config.ints(), cfg::MAX_SEQ_LEN) * kv_dim;
                uint16_t* key_pos_fp16 = kcache_layer + (size_t)pos * kv_dim;
                for (int d = 0; d < kv_dim; d++) {
                    key_pos_fp16[d] = fp32_to_fp16(state.xb2.data()[d]);
                }

                auto v_entry = lw.get(layer::ATTN_V);
                auto v_b_entry = lw.get(layer::ATTN_V_B);
                TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), v_entry.ptr,
                                       state.attn_v_bias[l].data(), dim, kv_dim,
                                       (GGUFType)v_entry.type, (GGUFType)v_b_entry.type, state.dequant_scratch.data());

                uint16_t* vcache_layer = state.val_cache.data() + (size_t)l * getInt(config.ints(), cfg::MAX_SEQ_LEN) * kv_dim;
                uint16_t* val_pos_fp16 = vcache_layer + (size_t)pos * kv_dim;
                for (int d = 0; d < kv_dim; d++) {
                    val_pos_fp16[d] = fp32_to_fp16(state.xb2.data()[d]);
                }

                // Flash Attention
                std::fill(state.xb.data(), state.xb.data() + q_dim, 0.0f);
                for (int h = 0; h < n_heads; h++) {
                    float* qh = state.q.data() + h * head_dim;
                    int kv_h = h / kv_mul;
                    float* xbh = state.xb.data() + h * head_dim;

                    float max_score = -1e30f;
                    float sum_exp = 0.0f;
                    std::fill(state.acc, state.acc + head_dim, 0.0f);

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
                            for (int d = 0; d < head_dim; d++) state.acc[d] *= correction;
                            sum_exp += 1.0f;
                            for (int d = 0; d < head_dim; d++) {
                                state.acc[d] += fp16_to_fp32(vt[d]);
                            }
                            max_score = score;
                        } else {
                            float w = std::exp(score - max_score);
                            sum_exp += w;
                            for (int d = 0; d < head_dim; d++) {
                                state.acc[d] += w * fp16_to_fp32(vt[d]);
                            }
                        }
                    }
                    for (int d = 0; d < head_dim; d++) {
                        xbh[d] = state.acc[d] / sum_exp;
                    }
                }

                auto attn_out_entry = lw.get(layer::ATTN_OUTPUT);
                auto attn_out_b_entry = lw.get(layer::ATTN_OUTPUT_B);
                TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), attn_out_entry.ptr,
                                       state.attn_output_bias[l].data(), q_dim, dim,
                                       (GGUFType)attn_out_entry.type, (GGUFType)attn_out_b_entry.type, state.dequant_scratch.data());
                TensorOps::vec_add(state.x.data(), state.xb2.data(), dim);

                // FFN
                TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.ffn_norm_w[l].data(), dim);

                auto gate_entry = lw.get(layer::FFN_GATE);
                auto gate_b_entry = lw.get(layer::FFN_GATE_B);
                TensorOps::matmul_bias(state.hb.data(), state.xb.data(), gate_entry.ptr,
                                       state.ffn_gate_bias[l].data(), dim, n_ffn,
                                       (GGUFType)gate_entry.type, (GGUFType)gate_b_entry.type, state.dequant_scratch.data());
                TensorOps::silu(state.hb.data(), n_ffn);

                auto up_entry = lw.get(layer::FFN_UP);
                TensorOps::matmul(state.hb2.data(), state.xb.data(), up_entry.ptr, dim, n_ffn, (GGUFType)up_entry.type);
                TensorOps::elemwise_mul(state.hb.data(), state.hb.data(), state.hb2.data(), n_ffn);

                auto down_entry = lw.get(layer::FFN_DOWN);
                auto down_b_entry = lw.get(layer::FFN_DOWN_B);
                TensorOps::matmul_bias(state.xb.data(), state.hb.data(), down_entry.ptr,
                                       state.ffn_down_bias[l].data(), n_ffn, dim,
                                       (GGUFType)down_entry.type, (GGUFType)down_b_entry.type, state.dequant_scratch.data());
                TensorOps::vec_add(state.x.data(), state.xb.data(), dim);

                std::memcpy(batch_x.data() + (size_t)i * dim, state.x.data(), dim * sizeof(float));
            }
            evict(l);
        }

        // Final norm and output
        std::memcpy(state.x.data(), batch_x.data() + (size_t)(num_tokens - 1) * dim, dim * sizeof(float));
        TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.output_norm_w.data(), dim);
        auto output_entry = weights.get(output::OUTPUT);
        TensorOps::matmul(state.logits.data(), state.xb.data(), output_entry.ptr,
                         dim, getInt(config.ints(), cfg::VOCAB_SIZE), (GGUFType)output_entry.type);

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
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (file_handle == INVALID_HANDLE_VALUE) {
            fprintf(stderr, "[Error] CreateFileA failed for '%s'. Windows Error Code: %lu\n", path, GetLastError());
            return -1;
        }

        LARGE_INTEGER fsize;
        if (!GetFileSizeEx(file_handle, &fsize)) {
            fprintf(stderr, "[Error] GetFileSizeEx failed. Windows Error Code: %lu\n", GetLastError());
            CloseHandle(file_handle);
            return -1;
        }
        mmap_size = (size_t)fsize.QuadPart;

        map_handle = CreateFileMappingA(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map_handle) { 
            fprintf(stderr, "[Error] CreateFileMappingA failed. Windows Error Code: %lu\n", GetLastError());
            CloseHandle(file_handle); 
            return -1; 
        }

        mmap_addr = MapViewOfFile(map_handle, FILE_MAP_READ, 0, 0, 0);
        if (!mmap_addr) { 
            fprintf(stderr, "[Error] MapViewOfFile failed. Windows Error Code: %lu\n", GetLastError());
            CloseHandle(map_handle); 
            CloseHandle(file_handle); 
            return -1; 
        }
#else
        fd = open(path, O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "[Error] open() failed for '%s'\n", path);
            return -1;
        }

        struct stat st;
        fstat(fd, &st);
        mmap_size = (size_t)st.st_size;

        mmap_addr = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mmap_addr == MAP_FAILED) { 
            fprintf(stderr, "[Error] mmap() failed\n");
            close(fd); 
            return -1; 
        }
#endif
        return 0;
    }

    int parse_gguf(int max_seq_len_override) {
        const uint8_t* data = (const uint8_t*)mmap_addr;
        size_t pos = 0;

        uint32_t magic; std::memcpy(&magic, data + pos, 4); pos += 4;
        if (magic != GGUF_MAGIC) {
            fprintf(stderr, "Invalid GGUF magic: 0x%08X\n", magic);
            return -1;
        }

        uint32_t version; std::memcpy(&version, data + pos, 4); pos += 4;
        if (version < 2 || version > 3) {
            fprintf(stderr, "Invalid GGUF version: %u\n", version);
            return -1;
        }

        uint64_t n_tensors, n_metadata;
        std::memcpy(&n_tensors, data + pos, 8); pos += 8;
        std::memcpy(&n_metadata, data + pos, 8); pos += 8;

        config.floats()[cfg::ROPE_FREQ_BASE] = 10000.0f;
        config.ints()[cfg::MAX_SEQ_LEN] = 2048;
        config.ints()[cfg::HEAD_DIM] = -1;

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
                    case 0: case 1: case 7: pos += 1; break;
                    case 2: case 3: pos += 2; break;
                    case 4: case 5: pos += 4; break;
                    case 10: case 11: pos += 8; break;
                    case 6: pos += 4; break;
                    case 12: pos += 8; break;
                    case 8: {
                        uint64_t slen = read_u64();
                        pos += slen;
                        break;
                    }
                    case 9: {
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

            // 使用统一的解析函数
            if (key.find("tokenizer.ggml.bos_token_id") != std::string::npos) {
                tok_bos_id = (uint32_t)read_i32();
                config.metadata()[key] = tok_bos_id;
            } else if (key.find("tokenizer.ggml.eos_token_id") != std::string::npos) {
                tok_eos_id = (uint32_t)read_i32();
                config.metadata()[key] = tok_eos_id;
            } else if (key.find("tokenizer.ggml.tokens") != std::string::npos) {
                uint32_t arr_type = read_i32(); (void)arr_type;
                uint64_t arr_len = read_u64();
                tok_tokens_data = data + pos;
                tok_n_tokens = arr_len;
                for (uint64_t j = 0; j < arr_len; j++) {
                    uint64_t slen = read_u64();
                    pos += slen;
                }
                config.metadata()[key] = arr_len;
            } else if (key.find("tokenizer.ggml.scores") != std::string::npos) {
                uint32_t arr_type = read_i32(); (void)arr_type;
                uint64_t arr_len = read_u64();
                tok_scores_data = data + pos;
                tok_n_scores = arr_len;
                pos += arr_len * 4;
                config.metadata()[key] = arr_len;
            } else if (key.find("tokenizer.ggml.token_type") != std::string::npos) {
                uint32_t arr_type = read_i32(); (void)arr_type;
                uint64_t arr_len = read_u64();
                pos += arr_len * 4;
            } else if (key.find("tokenizer.ggml.pre") != std::string::npos) {
                skip_value();
            } else if (vtype == 4 || vtype == 5) {
                int32_t val = read_i32();
                parser::parse_config_value(key, val, config.config);
            } else if (vtype == 6) {
                float val = read_f32();
                parser::parse_config_value(key, val, config.config);
            } else {
                skip_value();
            }
        }

        // BOS/EOS 默认值调整
        if (tok_bos_id == 1 && getInt(config.ints(), cfg::VOCAB_SIZE) > 150000) {
            tok_bos_id = 151643;
            tok_eos_id = 151645;
        }

        // 限制 max_seq_len
        if (max_seq_len_override > 0 && max_seq_len_override < getInt(config.ints(), cfg::MAX_SEQ_LEN)) {
            config.ints()[cfg::MAX_SEQ_LEN] = max_seq_len_override;
        } else if (getInt(config.ints(), cfg::MAX_SEQ_LEN) > 2048) {
            config.ints()[cfg::MAX_SEQ_LEN] = 2048;
        }
        if (getInt(config.ints(), cfg::HEAD_DIM) < 1) {
            config.ints()[cfg::HEAD_DIM] = getInt(config.ints(), cfg::N_EMBD) / getInt(config.ints(), cfg::N_HEADS);
        }

        // Parse tensor info
        std::vector<metadata::TensorInfo> tinfos(n_tensors);

        for (uint64_t i = 0; i < n_tensors; i++) {
            uint64_t name_len; std::memcpy(&name_len, data+pos, 8); pos += 8;
            if (pos + name_len > mmap_size) {
                fprintf(stderr, "\nError: name_len exceeds mmap_size\n");
                return -1;
            }
            tinfos[i].name.assign((const char*)data+pos, name_len); pos += name_len;
            std::memcpy(&tinfos[i].n_dims, data+pos, 4); pos += 4;
            for (uint32_t d = 0; d < tinfos[i].n_dims; d++) {
                std::memcpy(&tinfos[i].dims[d], data+pos, 8); pos += 8;
            }
            std::memcpy(&tinfos[i].type, data+pos, 4); pos += 4;
            tinfos[i].type += 1;  // GGUF 类型值需要 +1 才能匹配枚举
            std::memcpy(&tinfos[i].offset, data+pos, 8); pos += 8;
        }

        int alignment = getInt(config.ints(), cfg::ALIGNMENT);
        if (alignment <= 0) alignment = 32;
        size_t tensor_data_base = ((pos + alignment - 1) / alignment) * alignment;

        // 使用映射表加载张量
        for (const auto& ti : tinfos) {
            const void* ptr = (const uint8_t*)mmap_addr + tensor_data_base + ti.offset;
            GGUFType qtype = (GGUFType)ti.type;

            // 处理输出层权重
            if (parser::is_output_weight(ti.name)) {
                std::string weight_type = parser::get_output_type(ti.name);
                weights.set(weight_type, ptr, (uint32_t)qtype);
            }
            // 处理层权重
            else if (parser::is_layer_weight(ti.name)) {
                auto [layer_id, suffix] = parser::parse_layer_name(ti.name);
                if (layer_id >= 0 && layer_id < getInt(config.ints(), cfg::N_LAYERS)) {
                    // 移除 ".weight" 后缀
                    if (suffix.size() > 7 && suffix.compare(suffix.size() - 7, 7, ".weight") == 0) {
                        suffix = suffix.substr(0, suffix.size() - 7);
                    }
                    weights.get_layer(layer_id).set(suffix, ptr, (uint32_t)qtype);
                }
            }
        }

        // output 默认指向 token_embd
        if (!weights.get(output::OUTPUT)) {
            auto embd = weights.get(output::TOKEN_EMBD);
            weights.set(output::OUTPUT, embd.ptr, embd.type);
        }

        // 获取 vocab_size
        if (getInt(config.ints(), cfg::VOCAB_SIZE) == 0) {
            for (const auto& ti : tinfos) {
                std::string embd_name = std::string(output::TOKEN_EMBD) + ".weight";
                if (ti.name == embd_name && ti.n_dims >= 2) {
                    config.ints()[cfg::VOCAB_SIZE] = (ti.dims[0] == (uint64_t)getInt(config.ints(), cfg::N_EMBD)) ? (int)ti.dims[1] : (int)ti.dims[0];
                    break;
                }
            }
        }
        if (getInt(config.ints(), cfg::VOCAB_SIZE) == 0 && tok_n_tokens > 0) {
            config.ints()[cfg::VOCAB_SIZE] = (int)tok_n_tokens;
        }

        config.ints()[cfg::WEIGHT_TYPE] = weights.get_layer(0).get(layer::ATTN_Q).type;

        return 0;
    }

    int allocate_run_state() {
        int dim = getInt(config.ints(), cfg::N_EMBD);
        int n_ffn = getInt(config.ints(), cfg::N_FFN);
        int q_dim = getInt(config.ints(), cfg::N_HEADS) * getInt(config.ints(), cfg::HEAD_DIM);
        int kv_dim = getInt(config.ints(), cfg::N_KV_HEADS) * getInt(config.ints(), cfg::HEAD_DIM);
        int half_dim = getInt(config.ints(), cfg::HEAD_DIM) / 2;
        int max_scratch = std::max({dim, q_dim, n_ffn, getInt(config.ints(), cfg::VOCAB_SIZE)});

        state.x.resize((size_t)max_scratch, 0.0f);
        state.xb.resize((size_t)max_scratch, 0.0f);
        state.xb2.resize((size_t)max_scratch, 0.0f);
        state.q.resize((size_t)q_dim, 0.0f);
        state.hb.resize((size_t)n_ffn, 0.0f);
        state.hb2.resize((size_t)n_ffn, 0.0f);
        state.logits.resize((size_t)getInt(config.ints(), cfg::VOCAB_SIZE), 0.0f);
        state.dequant_scratch.resize((size_t)max_scratch, 0.0f);
        state.rope_cos.resize((size_t)getInt(config.ints(), cfg::MAX_SEQ_LEN) * half_dim, 0.0f);
        state.rope_sin.resize((size_t)getInt(config.ints(), cfg::MAX_SEQ_LEN) * half_dim, 0.0f);

        // 归一化权重
        state.attn_norm_w.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        state.ffn_norm_w.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        state.output_norm_w.resize((size_t)dim);

        for (int l = 0; l < getInt(config.ints(), cfg::N_LAYERS); l++) {
            state.attn_norm_w[l].resize((size_t)dim);
            state.ffn_norm_w[l].resize((size_t)dim);

            auto attn_norm = weights.get_layer(l).get(layer::ATTN_NORM);
            auto ffn_norm = weights.get_layer(l).get(layer::FFN_NORM);

            if (attn_norm) {
                dequantize_row(attn_norm.ptr, state.attn_norm_w[l].data(), dim, (GGUFType)attn_norm.type);
            }
            if (ffn_norm) {
                dequantize_row(ffn_norm.ptr, state.ffn_norm_w[l].data(), dim, (GGUFType)ffn_norm.type);
            }
        }

        auto output_norm = weights.get(output::OUTPUT_NORM);
        if (output_norm) {
            dequantize_row(output_norm.ptr, state.output_norm_w.data(), dim, (GGUFType)output_norm.type);
        }

        // 偏置
        state.attn_q_bias.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        state.attn_k_bias.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        state.attn_v_bias.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        state.attn_output_bias.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        state.ffn_gate_bias.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        state.ffn_up_bias.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        state.ffn_down_bias.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));

        for (int l = 0; l < getInt(config.ints(), cfg::N_LAYERS); l++) {
            state.attn_q_bias[l].resize((size_t)q_dim, 0.0f);
            state.attn_k_bias[l].resize((size_t)kv_dim, 0.0f);
            state.attn_v_bias[l].resize((size_t)kv_dim, 0.0f);
            state.attn_output_bias[l].resize((size_t)dim, 0.0f);
            state.ffn_gate_bias[l].resize((size_t)n_ffn, 0.0f);
            state.ffn_up_bias[l].resize((size_t)n_ffn, 0.0f);
            state.ffn_down_bias[l].resize((size_t)dim, 0.0f);

            LayerWeights& lw = weights.get_layer(l);
            auto q_b = lw.get(layer::ATTN_Q_B);
            auto k_b = lw.get(layer::ATTN_K_B);
            auto v_b = lw.get(layer::ATTN_V_B);
            auto out_b = lw.get(layer::ATTN_OUTPUT_B);
            auto gate_b = lw.get(layer::FFN_GATE_B);
            auto up_b = lw.get(layer::FFN_UP_B);
            auto down_b = lw.get(layer::FFN_DOWN_B);

            if (q_b) dequantize_row(q_b.ptr, state.attn_q_bias[l].data(), q_dim, (GGUFType)q_b.type);
            if (k_b) dequantize_row(k_b.ptr, state.attn_k_bias[l].data(), kv_dim, (GGUFType)k_b.type);
            if (v_b) dequantize_row(v_b.ptr, state.attn_v_bias[l].data(), kv_dim, (GGUFType)v_b.type);
            if (out_b) dequantize_row(out_b.ptr, state.attn_output_bias[l].data(), dim, (GGUFType)out_b.type);
            if (gate_b) dequantize_row(gate_b.ptr, state.ffn_gate_bias[l].data(), n_ffn, (GGUFType)gate_b.type);
            if (up_b) dequantize_row(up_b.ptr, state.ffn_up_bias[l].data(), n_ffn, (GGUFType)up_b.type);
            if (down_b) dequantize_row(down_b.ptr, state.ffn_down_bias[l].data(), dim, (GGUFType)down_b.type);
        }

        // QK-Norm 权重
        state.attn_q_norm_w.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        state.attn_k_norm_w.resize((size_t)getInt(config.ints(), cfg::N_LAYERS));
        for (int l = 0; l < getInt(config.ints(), cfg::N_LAYERS); l++) {
            state.attn_q_norm_w[l].resize((size_t)getInt(config.ints(), cfg::HEAD_DIM));
            state.attn_k_norm_w[l].resize((size_t)getInt(config.ints(), cfg::HEAD_DIM));

            LayerWeights& lw = weights.get_layer(l);
            auto q_norm = lw.get(layer::ATTN_Q_NORM);
            auto k_norm = lw.get(layer::ATTN_K_NORM);

            if (q_norm) {
                dequantize_row(q_norm.ptr, state.attn_q_norm_w[l].data(), getInt(config.ints(), cfg::HEAD_DIM), (GGUFType)q_norm.type);
            }
            if (k_norm) {
                dequantize_row(k_norm.ptr, state.attn_k_norm_w[l].data(), getInt(config.ints(), cfg::HEAD_DIM), (GGUFType)k_norm.type);
            }
        }

        // KV Cache
        size_t kv_elements = (size_t)getInt(config.ints(), cfg::N_LAYERS) * (size_t)getInt(config.ints(), cfg::MAX_SEQ_LEN) * (size_t)kv_dim;
        state.key_cache.resize(kv_elements, 0);
        state.val_cache.resize(kv_elements, 0);

        // RoPE 表
        for (int pos = 0; pos < getInt(config.ints(), cfg::MAX_SEQ_LEN); pos++) {
            float* cos_row = state.rope_cos.data() + (size_t)pos * half_dim;
            float* sin_row = state.rope_sin.data() + (size_t)pos * half_dim;
            for (int i = 0; i < half_dim; i++) {
                float theta = (float)pos / std::pow(getFloat(config.floats(), cfg::ROPE_FREQ_BASE), (float)(2 * i) / (float)getInt(config.ints(), cfg::HEAD_DIM));
                cos_row[i] = std::cos(theta);
                sin_row[i] = std::sin(theta);
            }
        }

        state.mem_size = state.x.capacity() * sizeof(float) * 8 +
                        state.rope_cos.capacity() * sizeof(float) * 2 +
                        kv_elements * sizeof(uint16_t) * 2;

        // ================= 终极内存锁定逻辑 (带精准计算和容错检查) =================
        
        size_t total_lock_bytes = 0;

        // 阶段 1: 仅仅统计所需锁定的字节数，不执行锁定
        auto calc_vec_1d = [&total_lock_bytes](auto& vec) {
            if (!vec.empty()) total_lock_bytes += vec.capacity() * sizeof(vec[0]);
        };
        auto calc_vec_2d = [&](auto& vec2d) {
            calc_vec_1d(vec2d);
            for (auto& inner : vec2d) calc_vec_1d(inner);
        };

        calc_vec_1d(state.x); calc_vec_1d(state.xb); calc_vec_1d(state.xb2);
        calc_vec_1d(state.q); calc_vec_1d(state.hb); calc_vec_1d(state.hb2);
        calc_vec_1d(state.logits); calc_vec_1d(state.dequant_scratch);
        calc_vec_1d(state.key_cache); calc_vec_1d(state.val_cache);
        calc_vec_1d(state.rope_cos); calc_vec_1d(state.rope_sin);
        calc_vec_1d(state.output_norm_w);
        calc_vec_2d(state.attn_norm_w); calc_vec_2d(state.ffn_norm_w);
        calc_vec_2d(state.attn_q_bias); calc_vec_2d(state.attn_k_bias);
        calc_vec_2d(state.attn_v_bias); calc_vec_2d(state.attn_output_bias);
        calc_vec_2d(state.ffn_gate_bias); calc_vec_2d(state.ffn_up_bias);
        calc_vec_2d(state.ffn_down_bias);
        calc_vec_2d(state.attn_q_norm_w); calc_vec_2d(state.attn_k_norm_w);

#ifdef _WIN32
        // 阶段 2: 暴力扩充 Windows 工作集，额外预留 512MB 缓冲防越界
        HANDLE hProcess = GetCurrentProcess();
        SIZE_T min_sz, max_sz;
        if (GetProcessWorkingSetSize(hProcess, &min_sz, &max_sz)) {
            SIZE_T needed_extra = total_lock_bytes + (512ULL * 1024ULL * 1024ULL); // +512MB
            if (!SetProcessWorkingSetSize(hProcess, min_sz + needed_extra, max_sz + needed_extra)) {
                fprintf(stderr, "[Warning] SetProcessWorkingSetSize failed! Code: %lu\n", GetLastError());
            }
        }
#endif

        // 阶段 3: 真正执行锁定，并带上报错机制
        auto do_lock_1d = [](auto& vec, const char* name) {
            if (!vec.empty()) {
                size_t bytes = vec.capacity() * sizeof(vec[0]);
#ifdef _WIN32
                if (!VirtualLock((LPVOID)vec.data(), bytes)) {
                    fprintf(stderr, "[Warning] VirtualLock failed on %s! Code: %lu\n", name, GetLastError());
                }
#else
                if (mlock((void*)vec.data(), bytes) != 0) {
                    fprintf(stderr, "[Warning] mlock failed on %s!\n", name);
                }
#endif
            }
        };
        auto do_lock_2d = [&](auto& vec2d, const char* name) {
            do_lock_1d(vec2d, name);
            for (auto& inner : vec2d) do_lock_1d(inner, name);
        };

        do_lock_1d(state.x, "state.x"); do_lock_1d(state.xb, "state.xb"); do_lock_1d(state.xb2, "state.xb2");
        do_lock_1d(state.q, "state.q"); do_lock_1d(state.hb, "state.hb"); do_lock_1d(state.hb2, "state.hb2");
        do_lock_1d(state.logits, "state.logits"); do_lock_1d(state.dequant_scratch, "state.dequant_scratch");
        do_lock_1d(state.key_cache, "state.key_cache"); do_lock_1d(state.val_cache, "state.val_cache");
        do_lock_1d(state.rope_cos, "state.rope_cos"); do_lock_1d(state.rope_sin, "state.rope_sin");
        do_lock_1d(state.output_norm_w, "state.output_norm_w");
        do_lock_2d(state.attn_norm_w, "state.attn_norm_w"); do_lock_2d(state.ffn_norm_w, "state.ffn_norm_w");
        do_lock_2d(state.attn_q_bias, "state.attn_q_bias"); do_lock_2d(state.attn_k_bias, "state.attn_k_bias");
        do_lock_2d(state.attn_v_bias, "state.attn_v_bias"); do_lock_2d(state.attn_output_bias, "state.attn_output_bias");
        do_lock_2d(state.ffn_gate_bias, "state.ffn_gate_bias"); do_lock_2d(state.ffn_up_bias, "state.ffn_up_bias");
        do_lock_2d(state.ffn_down_bias, "state.ffn_down_bias");
        do_lock_2d(state.attn_q_norm_w, "state.attn_q_norm_w"); do_lock_2d(state.attn_k_norm_w, "state.attn_k_norm_w");

        // ==============================================================================
        
        fprintf(stderr, "Allocating %.2f MB for runtime state (Locked: %.2f MB)\n", 
                state.mem_size / (1024.0 * 1024.0), total_lock_bytes / (1024.0 * 1024.0));
        return 0;
    }
};

#endif // MODEL_HPP
