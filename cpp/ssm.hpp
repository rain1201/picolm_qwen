#ifndef SSM_HPP
#define SSM_HPP
#include "quant.hpp"
#include "tensor.hpp"

class SSMOps {
public:
    static inline float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }

    static inline float softplus(float x) {
        if (x > 20.0f) return x;
        return std::log1p(std::exp(x));
    }

    // 修复后的 Gated DeltaNet / Mamba-2 混合推断步
    static void forward_ssm_step(
        float* out, float* x_in, LayerWeights* lw,
        int dim, int ssm_qkv_dim, int ssm_n_heads, int ssm_n_kv_heads, int ssm_head_dim, int kernel_size,
        float* conv_state, float* hidden_state, 
        const float* ssm_norm_w, const float* conv1d_w, int conv1d_len, // 【新增参数】
        float* scratch
    ) {
        float* qkv      = scratch;                                     // Size: ssm_qkv_dim (6144)
        float* qkv_conv = scratch + ssm_qkv_dim;                       // Size: ssm_qkv_dim (6144)
        float* alpha_p  = scratch + ssm_qkv_dim * 2;                   // Size: ssm_n_heads (16)
        float* beta_p   = scratch + ssm_qkv_dim * 2 + ssm_n_heads;     // Size: ssm_n_heads (16)
        float* gate     = scratch + ssm_qkv_dim * 2 + ssm_n_heads * 2; // Size: dim (2048)
        float* ssm_out  = scratch + ssm_qkv_dim * 2 + ssm_n_heads * 2 + dim; // Size: dim (2048)

        // 1. QKV 投影
        TensorOps::matmul(qkv, x_in, lw->attn_qkv, dim, ssm_qkv_dim, lw->type_attn_qkv);

        // 2. 1D 因果卷积
        // 算出真正受卷积影响的通道数 (有时 V 不参与 1D 卷积，以防越界)
        int max_c = ssm_qkv_dim;
        if (conv1d_len > 0 && conv1d_w != nullptr) {
            max_c = conv1d_len / kernel_size;
        }

        for (int c = 0; c < ssm_qkv_dim; c++) {
            if (c < max_c && conv1d_w != nullptr) {
                float sum = qkv[c] * conv1d_w[c * kernel_size + (kernel_size - 1)];
                for (int k = 0; k < kernel_size - 1; k++) {
                    sum += conv_state[k * ssm_qkv_dim + c] * conv1d_w[c * kernel_size + k];
                }
                qkv_conv[c] = silu(sum);
            } else {
                qkv_conv[c] = silu(qkv[c]); // 超出权重的通道 (通常是 V)，不进行卷积
            }
        }

        // 滚动更新 1D 卷积缓存
        for (int k = 0; k < kernel_size - 2; k++) {
            std::memcpy(conv_state + k * ssm_qkv_dim, conv_state + (k + 1) * ssm_qkv_dim, ssm_qkv_dim * sizeof(float));
        }
        std::memcpy(conv_state + (kernel_size - 2) * ssm_qkv_dim, qkv, ssm_qkv_dim * sizeof(float));

        // 3. 计算时间步(dt)与衰减因子
        TensorOps::matmul(alpha_p, x_in, lw->ssm_alpha, dim, ssm_n_heads, lw->type_ssm_alpha);
        if (lw->ssm_beta) TensorOps::matmul(beta_p, x_in, lw->ssm_beta, dim, ssm_n_heads, lw->type_ssm_beta);

        std::vector<float> a_val(ssm_n_heads), b_val(ssm_n_heads);
        
        // 解析 Mamba 的连续化参数
        std::vector<float> dt_bias_buf(ssm_n_heads, 0.0f), a_buf(ssm_n_heads, 0.0f);
        if (lw->ssm_dt_bias) dequantize_row(lw->ssm_dt_bias, dt_bias_buf.data(), ssm_n_heads, lw->type_ssm_dt_bias);
        if (lw->ssm_a) dequantize_row(lw->ssm_a, a_buf.data(), ssm_n_heads, lw->type_ssm_a);

        for(int h = 0; h < ssm_n_heads; h++) {
            if (lw->ssm_dt_bias && lw->ssm_a) {
                float dt = softplus(alpha_p[h] + dt_bias_buf[h]);
                a_val[h] = std::exp(dt * a_buf[h]);
                b_val[h] = dt; 
                if (lw->ssm_beta) b_val[h] = dt * (1.0f / (1.0f + std::exp(-beta_p[h])));
            } else {
                a_val[h] = 1.0f / (1.0f + std::exp(-alpha_p[h]));
                b_val[h] = lw->ssm_beta ? (1.0f / (1.0f + std::exp(-beta_p[h]))) : 1.0f;
            }
        }

        // 4. RNN 前向传播 (协方差矩阵状态)
        float* q = qkv_conv;
        float* k = qkv_conv + ssm_n_heads * ssm_head_dim; 
        float* v = qkv_conv + ssm_n_heads * ssm_head_dim + ssm_n_kv_heads * ssm_head_dim;
        int kv_mul = ssm_n_heads / ssm_n_kv_heads;
        
       for (int h = 0; h < ssm_n_heads; h++) {
            int kv_h = h / kv_mul;
            float* q_h = q + h * ssm_head_dim;
            float* k_h = k + kv_h * ssm_head_dim;
            float* v_h = v + kv_h * ssm_head_dim;
            float* S_h = hidden_state + h * ssm_head_dim * ssm_head_dim;
            float a = a_val[h], b = b_val[h];

            // ======== 【新增修改】对 Q 和 K 进行 RMSNorm 归一化 ========
            // 防止 b > 1 时产生指数爆炸导致 NaN
            float sq = 0.0f, sk = 0.0f;
            for(int i = 0; i < ssm_head_dim; i++) {
                sq += q_h[i] * q_h[i];
                sk += k_h[i] * k_h[i];
            }
            sq = 1.0f / std::sqrt(sq / ssm_head_dim + 1e-6f);
            sk = 1.0f / std::sqrt(sk / ssm_head_dim + 1e-6f);
            for(int i = 0; i < ssm_head_dim; i++) {
                q_h[i] *= sq;
                k_h[i] *= sk;
            }
            // =========================================================

            std::vector<float> p(ssm_head_dim, 0.0f);
            
            for (int i = 0; i < ssm_head_dim; i++) {
                for (int j = 0; j < ssm_head_dim; j++) {
                    p[j] += S_h[i * ssm_head_dim + j] * k_h[i];
                }
            }
            
            // 2. 使用 Delta (v_h - p) 更新状态 S_t
            for (int i = 0; i < ssm_head_dim; i++) {
                for (int j = 0; j < ssm_head_dim; j++) {
                    S_h[i * ssm_head_dim + j] = a * S_h[i * ssm_head_dim + j] + b * k_h[i] * (v_h[j] - p[j]);
                }
            }

            float* out_h = ssm_out + h * ssm_head_dim;
            for (int j = 0; j < ssm_head_dim; j++) {
                float sum = 0.0f;
                for (int i = 0; i < ssm_head_dim; i++) {
                    sum += q_h[i] * S_h[i * ssm_head_dim + j];
                }
                out_h[j] = sum;
            }
        }

        // 5. GroupNorm (修复指针偏移量，不要所有头全用第一套 norm 权重)
        if (ssm_norm_w) {
            for (int h = 0; h < ssm_n_heads; h++) {
                float* out_h = ssm_out + h * ssm_head_dim;
                const float* norm_w_h = ssm_norm_w + h * ssm_head_dim; // 【修复】指向各自头的归一化权重
                TensorOps::rmsnorm(out_h, out_h, norm_w_h, ssm_head_dim);
            }
        }

        // 6. Gated 门控
        if (lw->attn_gate && lw->type_attn_gate != GGUFType::NONE) {
            TensorOps::matmul(gate, x_in, lw->attn_gate, dim, dim, lw->type_attn_gate);
            for (int i = 0; i < dim; i++) ssm_out[i] *= silu(gate[i]);
        }

        // 7. 线性映射
        TensorOps::matmul(out, ssm_out, lw->ssm_out, dim, dim, lw->type_ssm_out);
    }
};
#endif // SSM_HPP