#ifndef SSM_HPP
#define SSM_HPP
#include "quant.hpp"
#include "tensor.hpp"

class SSMOps {
public:
    static inline float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }

    // Gated DeltaNet (线性注意力) 核心推断步
    static void forward_ssm_step(
        float* out,                  // [dim] (如 4096)
        float* x_in,                 // [dim] (如 4096)
        LayerWeights* lw,
        int dim,                     // dim (e.g. 4096)
        int qkv_dim,                 // Q + 2*K/V dim (e.g. 8192)
        int n_heads,                 // heads (e.g. 32)
        int n_kv_heads,              // kv heads (e.g. 16)
        int head_dim,                // head dim (e.g. 128)
        int kernel_size,             // Conv kernel (e.g. 4)
        float* conv_state,           //[kernel_size-1, qkv_dim]
        float* hidden_state,         //[n_heads, head_dim, head_dim] (协方差矩阵状态)
        const float* ssm_norm_w,     // [head_dim]
        float* scratch               // 工作内存
    ) {
        float* qkv      = scratch;                             // Size: qkv_dim
        float* qkv_conv = scratch + qkv_dim;                   // Size: qkv_dim
        float* alpha    = scratch + qkv_dim * 2;               // Size: n_heads
        float* beta     = scratch + qkv_dim * 2 + n_heads;     // Size: n_heads
        float* gate     = scratch + qkv_dim * 2 + n_heads * 2; // Size: dim
        float* ssm_out  = scratch + qkv_dim * 2 + n_heads * 2 + dim; // Size: dim

        // 1. QKV 投影 (生成完整未激活的 q, k, v 并联向量)
        TensorOps::matmul(qkv, x_in, lw->attn_qkv, dim, qkv_dim, lw->type_attn_qkv);

        // 2. 1D 因果卷积作用于全体 QKV 维度并进行 SiLU 激活
        const float* conv1d_w = (const float*)lw->ssm_conv1d; 
        for (int c = 0; c < qkv_dim; c++) {
            float sum = qkv[c] * conv1d_w[c * kernel_size + (kernel_size - 1)];
            for (int k = 0; k < kernel_size - 1; k++) {
                sum += conv_state[k * qkv_dim + c] * conv1d_w[c * kernel_size + k];
            }
            qkv_conv[c] = silu(sum);
        }

        // 滚动更新 1D 卷积状态缓存
        for (int k = 0; k < kernel_size - 2; k++) {
            std::memcpy(conv_state + k * qkv_dim, 
                        conv_state + (k + 1) * qkv_dim, 
                        qkv_dim * sizeof(float));
        }
        std::memcpy(conv_state + (kernel_size - 2) * qkv_dim, qkv, qkv_dim * sizeof(float));

        // 3. 计算各 Head 步长因子 Alpha 与 Beta (数据依赖型衰减)
        TensorOps::matmul(alpha, x_in, lw->ssm_alpha, dim, n_heads, lw->type_ssm_alpha);
        TensorOps::matmul(beta, x_in, lw->ssm_beta, dim, n_heads, lw->type_ssm_beta);

        for(int h = 0; h < n_heads; h++) {
            alpha[h] = 1.0f / (1.0f + std::exp(-alpha[h])); // Sigmoid 激活控制衰减
            beta[h]  = 1.0f / (1.0f + std::exp(-beta[h]));  // Sigmoid 激活控制增量
        }

        // 4. 执行 Delta Rule 的 RNN 前向传播
        float* q = qkv_conv;
        float* k = qkv_conv + n_heads * head_dim; 
        float* v = qkv_conv + n_heads * head_dim + n_kv_heads * head_dim;

        int kv_mul = n_heads / n_kv_heads;

        for (int h = 0; h < n_heads; h++) {
            int kv_h = h / kv_mul;
            float* q_h = q + h * head_dim;
            float* k_h = k + kv_h * head_dim;
            float* v_h = v + kv_h * head_dim;
            float* S_h = hidden_state + h * head_dim * head_dim; // 协方差矩阵

            float a = alpha[h];
            float b = beta[h];

            // 状态更新矩阵方程: S_t = a * S_{t-1} + b * (k_t^T \otimes v_t)
            for (int i = 0; i < head_dim; i++) {
                for (int j = 0; j < head_dim; j++) {
                    S_h[i * head_dim + j] = a * S_h[i * head_dim + j] + b * k_h[i] * v_h[j];
                }
            }

            // 输出提取: out_h = q_t @ S_t
            float* out_h = ssm_out + h * head_dim;
            for (int j = 0; j < head_dim; j++) {
                float sum = 0.0f;
                for (int i = 0; i < head_dim; i++) {
                    sum += q_h[i] * S_h[i * head_dim + j];
                }
                out_h[j] = sum;
            }
        }

        // 5. 对每个头独立施加 GroupNorm/Head-wise RMSNorm (对应 ssm_norm)
        if (ssm_norm_w) {
            for (int h = 0; h < n_heads; h++) {
                float* out_h = ssm_out + h * head_dim;
                TensorOps::rmsnorm(out_h, out_h, ssm_norm_w, head_dim);
            }
        }

        // 6. 独立门控点乘 (Gated DeltaNet 特色门控机制)
        TensorOps::matmul(gate, x_in, lw->attn_gate, dim, dim, lw->type_attn_gate);
        for (int i = 0; i < dim; i++) {
            ssm_out[i] *= silu(gate[i]);
        }

        // 7. 输出线性映射
        TensorOps::matmul(out, ssm_out, lw->ssm_out, dim, dim, lw->type_ssm_out);
    }
};
#endif // SSM_HPP