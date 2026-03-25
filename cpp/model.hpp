#ifndef MODEL_HPP
#define MODEL_HPP

#include "quant.hpp"
#include "tensor.hpp"
#include "tokenizer.hpp"
#include "prefetcher.hpp"
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

    int ssm_conv_kernel = 4;
    int ssm_state_size = 128;
    int ssm_inner_size = 4096;
    int ssm_time_step_rank = 32;
    int full_attention_interval = 4;

    bool use_prefetch = false;
    bool use_evict = false;
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

    const void* ssm_a = nullptr;          // blk.0.ssm_a
    const void* ssm_dt_bias = nullptr;    // blk.0.ssm_dt.bias
    const void* ssm_conv1d = nullptr;     // blk.0.ssm_conv1d.weight
    const void* ssm_alpha = nullptr;      // blk.0.ssm_alpha.weight (或 x_proj / dt_proj)
    const void* ssm_beta = nullptr;       // blk.0.ssm_beta.weight
    const void* ssm_norm = nullptr;       // blk.0.ssm_norm.weight
    const void* ssm_out = nullptr;        // blk.0.ssm_out.weight

    const void* attn_qkv = nullptr;       // 融合的 QKV 投影 或 SSM 的 in_proj
    const void* attn_gate = nullptr;      // 门控
    const void* post_attention_norm = nullptr; // 取代原有的 ffn_norm

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
    GGUFType type_ssm_a = GGUFType::NONE;
    GGUFType type_ssm_dt_bias = GGUFType::NONE;
    GGUFType type_ssm_conv1d = GGUFType::NONE;
    GGUFType type_ssm_alpha = GGUFType::NONE;
    GGUFType type_ssm_beta = GGUFType::NONE;
    GGUFType type_ssm_norm = GGUFType::NONE;
    GGUFType type_ssm_out = GGUFType::NONE;
    GGUFType type_attn_qkv = GGUFType::NONE;
    GGUFType type_attn_gate = GGUFType::NONE;
    GGUFType type_post_attention_norm = GGUFType::NONE;

    layer_type type = layer_type::TRANSFORMER;
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
    float acc[512];

    std::vector<std::vector<float>> post_attn_norm_w, ssm_norm_w;
    std::vector<float> ssm_conv_state;   // [ssm_layer][(kernel_size-1)*qkv_dim]
    std::vector<float> ssm_hidden_state; //[ssm_layer][n_heads*head_dim*head_dim]
    
    size_t mem_size = 0;
};

using RopeFn = void(*)(float*, float*, int, int, int, const float*, const float*);
#include "ssm.hpp"
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
        weights = ModelWeights();  // Zero-initialize
        config.use_prefetch = prefetch;
        config.use_evict = evict;

        if (mmap_file(path) != 0) return -1;
        if (parse_gguf(max_seq_len_override) != 0) return -1;
        if (allocate_run_state() != 0) return -1;

        return 0;
    }

    void prefetch(int l) {
        int dim = config.n_embd;
        int n_ffn = config.n_ffn;
        int head_dim = config.head_dim;
        int n_heads = config.n_heads;
        int q_dim = n_heads * head_dim;


        // 预取下一层
        if (config.use_prefetch && l + 1 < config.n_layers) {
            LayerWeights* next_lw = &weights.layers[l + 1];
            size_t attn_size = (size_t)gguf_type_row_size(next_lw->type_attn_q, dim) * q_dim;
            size_t ffn_size = (size_t)gguf_type_row_size(next_lw->type_ffn_gate, dim) * n_ffn;
            MemoryPrefetcher::prefetch_layer(next_lw->attn_q, attn_size);
            MemoryPrefetcher::prefetch_layer(next_lw->attn_k, attn_size);
            MemoryPrefetcher::prefetch_layer(next_lw->attn_v, attn_size);
            MemoryPrefetcher::prefetch_layer(next_lw->attn_output, attn_size);
            MemoryPrefetcher::prefetch_layer(next_lw->ffn_gate, ffn_size);
            MemoryPrefetcher::prefetch_layer(next_lw->ffn_up, ffn_size);
            MemoryPrefetcher::prefetch_layer(next_lw->ffn_down, ffn_size);
        }
    }

    void evict(int l){
        LayerWeights* lw = &weights.layers[l];
        int dim = config.n_embd;
        int n_ffn = config.n_ffn;
        int head_dim = config.head_dim;
        int n_heads = config.n_heads;
        int q_dim = n_heads * head_dim;
        int kv_dim = config.n_kv_heads * head_dim;

        // 定义操作系统级内存驱逐 Helper (用后即弃)
        auto evict_tensor =[](const void* ptr, size_t size) {
            if (!ptr || size == 0) return;
#ifdef _WIN32
            VirtualUnlock((LPVOID)ptr, size);
#else
            madvise((void*)ptr, size, MADV_DONTNEED);
#endif
        };

        if (config.use_evict) {
                evict_tensor(lw->attn_q,      gguf_type_row_size(lw->type_attn_q, dim) * q_dim);
                evict_tensor(lw->attn_k,      gguf_type_row_size(lw->type_attn_k, dim) * kv_dim);
                evict_tensor(lw->attn_v,      gguf_type_row_size(lw->type_attn_v, dim) * kv_dim);
                evict_tensor(lw->attn_output, gguf_type_row_size(lw->type_attn_output, q_dim) * dim);
                evict_tensor(lw->ffn_gate,    gguf_type_row_size(lw->type_ffn_gate, dim) * n_ffn);
                evict_tensor(lw->ffn_up,      gguf_type_row_size(lw->type_ffn_up, dim) * n_ffn);
                evict_tensor(lw->ffn_down,    gguf_type_row_size(lw->type_ffn_down, n_ffn) * dim);
        }
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

        int ssm_layer_idx = 0;

        // 1. Embedding lookup
        {
            size_t row_bytes = gguf_type_row_size(weights.type_token_embd, dim);
            const void* embd_row = (const uint8_t*)weights.token_embd + (size_t)token * row_bytes;
            dequantize_row(embd_row, state.x.data(), dim, weights.type_token_embd);
        }

        // 2. Transformer layers
        for (int l = 0; l < config.n_layers; l++) {
            LayerWeights* lw = &weights.layers[l];
            prefetch(l);
            switch (lw->type) {
                case (layer_type::TRANSFORMER):{
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
                    //std::vector<float> acc(head_dim, 0.0f);
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
                break;}
                case (layer_type::SSM):{
                    // 1. SSM (Gated DeltaNet) 替代注意力块
                    TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.attn_norm_w[l].data(), dim);

                    int qkv_dim = config.n_heads * config.head_dim + 2 * config.n_kv_heads * config.head_dim;
                    float* curr_conv_state = state.ssm_conv_state.data() + l * (config.ssm_conv_kernel - 1) * qkv_dim;
                    float* curr_hidden_state = state.ssm_hidden_state.data() + l * config.n_heads * config.head_dim * config.head_dim;
                    
                    SSMOps::forward_ssm_step(
                        state.xb2.data(),      
                        state.xb.data(),       
                        lw,                    
                        dim,
                        qkv_dim,
                        config.n_heads,
                        config.n_kv_heads,
                        config.head_dim,
                        config.ssm_conv_kernel,
                        curr_conv_state,
                        curr_hidden_state,
                        state.ssm_norm_w[l].data(),
                        state.dequant_scratch.data() 
                    );

                    // 残差连接
                    TensorOps::vec_add(state.x.data(), state.xb2.data(), dim);

                    // 2. 补充被遗漏的 Post Attention Norm 
                    TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.post_attn_norm_w[l].data(), dim);

                    // 3. 补充被遗漏的常规 FFN 层！
                    TensorOps::matmul_bias(state.hb.data(), state.xb.data(), lw->ffn_gate,
                                        state.ffn_gate_bias[l].data(), dim, n_ffn,
                                        lw->type_ffn_gate, lw->type_ffn_gate_b, state.dequant_scratch.data());
                    TensorOps::silu(state.hb.data(), n_ffn);
                    TensorOps::matmul(state.hb2.data(), state.xb.data(), lw->ffn_up, dim, n_ffn, lw->type_ffn_up);
                    TensorOps::elemwise_mul(state.hb.data(), state.hb.data(), state.hb2.data(), n_ffn);
                    TensorOps::matmul_bias(state.xb.data(), state.hb.data(), lw->ffn_down,
                                        state.ffn_down_bias[l].data(), n_ffn, dim,
                                        lw->type_ffn_down, lw->type_ffn_down_b, state.dequant_scratch.data());
                    // FFN 残差连接
                    TensorOps::vec_add(state.x.data(), state.xb.data(), dim);
                    break;}
                default:
                    throw std::runtime_error("Unsupported layer type");
            }
        }

        // Final norm and output
        TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.output_norm_w.data(), dim);
        TensorOps::matmul(state.logits.data(), state.xb.data(), weights.output, 
                         dim, config.vocab_size, weights.type_output);

        return state.logits.data();
    }

    // ------------------- 新增：Layer-wise 批处理前向传播 (用于 Prefill 阶段) -------------------
    float* forward_batch(const std::vector<int>& tokens, int start_pos) {
        float* ret= nullptr;
        for(auto i:tokens){
            ret = forward(i,start_pos++);
        }
        return ret;
        int num_tokens = (int)tokens.size();
        if (num_tokens == 0) return nullptr;

        int dim = config.n_embd;
        int n_ffn = config.n_ffn;
        int n_heads = config.n_heads;
        int n_kv_heads = config.n_kv_heads;
        int head_dim = config.head_dim;
        int kv_dim = n_kv_heads * head_dim;
        int kv_mul = n_heads / n_kv_heads;
        int half_dim = head_dim / 2;
        int q_dim = n_heads * head_dim;

        // 为这一批 token 分配跨层的中间状态 Buffer
        // 大小为 num_tokens * dim，通常只有几十MB，完全可以放进内存
        std::vector<float> batch_x((size_t)num_tokens * dim, 0.0f);

        // 1. 一次性完成所有 Token 的 Embedding Lookup
        for (int i = 0; i < num_tokens; i++) {
            size_t row_bytes = gguf_type_row_size(weights.type_token_embd, dim);
            const void* embd_row = (const uint8_t*)weights.token_embd + (size_t)tokens[i] * row_bytes;
            dequantize_row(embd_row, batch_x.data() + (size_t)i * dim, dim, weights.type_token_embd);
        }


        // 2. Transformer 层循环 (外层为 Layer，内层为 Token)
        for (int l = 0; l < config.n_layers; l++) {
            LayerWeights* lw = &weights.layers[l];

            prefetch(l);

            // 对该层依次处理所有 Token
            for (int i = 0; i < num_tokens; i++) {
                int pos = start_pos + i;
                
                // 将当前 Token 的状态载入 state.x 以复用后续的单 Token 运算逻辑
                std::memcpy(state.x.data(), batch_x.data() + (size_t)i * dim, dim * sizeof(float));

                // ---------- Attention ----------
                TensorOps::rmsnorm(state.xb.data(), state.x.data(), state.attn_norm_w[l].data(), dim);
                
                // Q, K, V 计算...
                TensorOps::matmul_bias(state.q.data(), state.xb.data(), lw->attn_q, 
                                       state.attn_q_bias[l].data(), dim, q_dim, 
                                       lw->type_attn_q, lw->type_attn_q_b, state.dequant_scratch.data());
                
                if (lw->attn_q_norm && state.attn_q_norm_w[l].data()) {
                    for (int h = 0; h < n_heads; h++) {
                        float* qh = state.q.data() + h * head_dim;
                        TensorOps::rmsnorm(qh, qh, state.attn_q_norm_w[l].data(), head_dim);
                    }
                }
                
                TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), lw->attn_k,
                                       state.attn_k_bias[l].data(), dim, kv_dim,
                                       lw->type_attn_k, lw->type_attn_k_b, state.dequant_scratch.data());
                
                if (lw->attn_k_norm && state.attn_k_norm_w[l].data()) {
                    for (int h = 0; h < n_kv_heads; h++) {
                        float* kh = state.xb2.data() + h * head_dim;
                        TensorOps::rmsnorm(kh, kh, state.attn_k_norm_w[l].data(), head_dim);
                    }
                }

                // RoPE
                const float* cos_pos = state.rope_cos.data() + (size_t)pos * half_dim;
                const float* sin_pos = state.rope_sin.data() + (size_t)pos * half_dim;
                rope_fn(state.q.data(), state.xb2.data(), head_dim, n_heads, n_kv_heads, cos_pos, sin_pos);

                // 将 K 存入 KV Cache
                uint16_t* kcache_layer = state.key_cache.data() + (size_t)l * config.max_seq_len * kv_dim;
                uint16_t* key_pos_fp16 = kcache_layer + (size_t)pos * kv_dim;
                for (int d = 0; d < kv_dim; d++) {
                    key_pos_fp16[d] = fp32_to_fp16(state.xb2.data()[d]);
                }

                TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), lw->attn_v,
                                       state.attn_v_bias[l].data(), dim, kv_dim,
                                       lw->type_attn_v, lw->type_attn_v_b, state.dequant_scratch.data());
                
                // 将 V 存入 KV Cache
                uint16_t* vcache_layer = state.val_cache.data() + (size_t)l * config.max_seq_len * kv_dim;
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
                            for (int d = 0; d < head_dim; d++) state.acc[d] += fp16_to_fp32(vt[d]);
                            max_score = score;
                        } else {
                            float w = std::exp(score - max_score);
                            sum_exp += w;
                            for (int d = 0; d < head_dim; d++) state.acc[d] += w * fp16_to_fp32(vt[d]);
                        }
                    }
                    for (int d = 0; d < head_dim; d++) {
                        xbh[d] = state.acc[d] / sum_exp;
                    }
                }

                // 输出映射
                TensorOps::matmul_bias(state.xb2.data(), state.xb.data(), lw->attn_output,
                                       state.attn_output_bias[l].data(), q_dim, dim,
                                       lw->type_attn_output, lw->type_attn_output_b, state.dequant_scratch.data());
                TensorOps::vec_add(state.x.data(), state.xb2.data(), dim);

                // ---------- FFN ----------
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

                // 算完了，把状态写回 batch_x
                std::memcpy(batch_x.data() + (size_t)i * dim, state.x.data(), dim * sizeof(float));
            } // end of token loop

            evict(l);
        } // end of layer loop

        // 3. Final norm and output (只计算最后一个 Token 的 Logits 即可)
        std::memcpy(state.x.data(), batch_x.data() + (size_t)(num_tokens - 1) * dim, dim * sizeof(float));
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
                    
                    else if (suffix == "ssm_norm.weight") {lw->type=layer_type::SSM; lw->ssm_norm = ptr; lw->type_ssm_norm = qtype;}
                    else if (suffix == "ssm_a") { lw->ssm_a = ptr; lw->type_ssm_a = qtype;}
                    else if (suffix == "ssm_alpha.weight") { lw->ssm_alpha= ptr; lw->type_ssm_alpha = qtype;}
                    else if (suffix == "ssm_beta.weight") { lw->ssm_beta = ptr; lw->type_ssm_beta = qtype;}
                    else if (suffix == "ssm_conv1d.weight") { lw->ssm_conv1d = ptr; lw->type_ssm_conv1d = qtype;}
                    else if (suffix == "ssm_dt.bias") { lw->ssm_dt_bias = ptr; lw->type_ssm_dt_bias = qtype;}
                    else if (suffix == "ssm_out.weight") { lw->ssm_out = ptr; lw->type_ssm_out = qtype;}

                    else if (suffix == "attn_qkv.weight") { lw->attn_qkv = ptr; lw->type_attn_qkv = qtype; }
                    else if (suffix == "attn_gate.weight") { lw->attn_gate = ptr; lw->type_attn_gate = qtype; }
                    else if (suffix == "post_attention_norm.weight") { lw->post_attention_norm = ptr; lw->type_post_attention_norm = qtype; }

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
        int ssm_inner = config.ssm_inner_size;
        int max_scratch = std::max({dim, 2 * q_dim, n_ffn, config.vocab_size, ssm_inner * 6}); 

        state.x.resize((size_t)max_scratch, 0.0f);
        state.xb.resize((size_t)max_scratch, 0.0f);
        state.xb2.resize((size_t)max_scratch, 0.0f);
        state.q.resize((size_t)q_dim, 0.0f);
        state.hb.resize((size_t)n_ffn, 0.0f);
        state.hb2.resize((size_t)n_ffn, 0.0f);
        state.logits.resize((size_t)config.vocab_size, 0.0f);
        state.dequant_scratch.resize((size_t)max_scratch, 0.0f);
        state.rope_cos.resize((size_t)config.max_seq_len * half_dim, 0.0f);
        state.rope_sin.resize((size_t)config.max_seq_len * half_dim, 0.0f);

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
            state.attn_q_bias[l].resize((size_t)q_dim, 0.0f);
            state.attn_k_bias[l].resize((size_t)kv_dim, 0.0f);
            state.attn_v_bias[l].resize((size_t)kv_dim, 0.0f);
            state.attn_output_bias[l].resize((size_t)dim, 0.0f);
            state.ffn_gate_bias[l].resize((size_t)n_ffn, 0.0f);
            state.ffn_up_bias[l].resize((size_t)n_ffn, 0.0f);
            state.ffn_down_bias[l].resize((size_t)dim, 0.0f);

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

        state.attn_norm_w.resize((size_t)config.n_layers);
        state.ffn_norm_w.resize((size_t)config.n_layers);
        state.post_attn_norm_w.resize((size_t)config.n_layers);
        state.ssm_norm_w.resize((size_t)config.n_layers);
        state.output_norm_w.resize((size_t)dim);

        for (int l = 0; l < config.n_layers; l++) {
            state.attn_norm_w[l].resize((size_t)dim);
            state.ffn_norm_w[l].resize((size_t)dim);
            state.post_attn_norm_w[l].resize((size_t)dim, 0.0f);
            state.ssm_norm_w[l].resize((size_t)config.head_dim, 0.0f);

            if (weights.layers[l].type_attn_norm != GGUFType::NONE)
                dequantize_row(weights.layers[l].attn_norm, state.attn_norm_w[l].data(), dim, weights.layers[l].type_attn_norm);
            if (weights.layers[l].type_ffn_norm != GGUFType::NONE)
                dequantize_row(weights.layers[l].ffn_norm, state.ffn_norm_w[l].data(), dim, weights.layers[l].type_ffn_norm);
            
            // 新增加载
            if (weights.layers[l].post_attention_norm && weights.layers[l].type_post_attention_norm != GGUFType::NONE)
                dequantize_row(weights.layers[l].post_attention_norm, state.post_attn_norm_w[l].data(), dim, weights.layers[l].type_post_attention_norm);
            if (weights.layers[l].ssm_norm && weights.layers[l].type_ssm_norm != GGUFType::NONE)
                dequantize_row(weights.layers[l].ssm_norm, state.ssm_norm_w[l].data(), config.head_dim, weights.layers[l].type_ssm_norm);
        }

        // KV Cache - use config.max_seq_len for proper indexing
        size_t num_attn_layers = config.n_layers;  // Assuming all layers have attention; adjust if not
        size_t kv_elements = (size_t)config.n_layers * (size_t)config.max_seq_len * (size_t)kv_dim;
        state.key_cache.resize(kv_elements, 0);
        state.val_cache.resize(kv_elements, 0);

        int num_ssm_layers = config.n_layers;// - num_attn_layers;
        // 1D 卷积缓存: 每一层需要保存前 (kernel_size - 1) 个 token 的隐状态
        state.ssm_conv_state.resize(num_ssm_layers * (config.ssm_conv_kernel - 1) * config.ssm_inner_size, 0.0f);
        // SSM 循环隐状态: h_t 矩阵
        state.ssm_hidden_state.resize(num_ssm_layers * config.ssm_inner_size * config.ssm_state_size, 0.0f);
        
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
                        kv_elements * sizeof(uint16_t) * 2 +
                        state.ssm_conv_state.capacity() * sizeof(float) +
                        state.ssm_hidden_state.capacity() * sizeof(float);

        fprintf(stderr, "Allocating %.2f MB for runtime state\n", state.mem_size / (1024.0 * 1024.0));
        return 0;
    }
};

#endif // MODEL_HPP
