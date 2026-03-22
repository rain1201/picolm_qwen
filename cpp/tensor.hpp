#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "quant.hpp"
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>

class TensorOps {
public:
    static void set_threads(int t) {
        n_threads = std::max(1, std::min(t, MAX_THREADS));
    }
    
    static int get_threads() { return n_threads; }

    // Matrix-vector multiply: out[d] = W[d, n] @ x[n]
    static void matmul(float* out, const float* x, const void* W, int n, int d, GGUFType qtype) {
        size_t row_bytes = gguf_type_row_size(qtype, n);
        const char* wptr = (const char*)W;

        if (n_threads <= 1 || d < 4) {
            for (int i = 0; i < d; i++) {
                out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
            }
            return;
        }

        int nt = std::min(n_threads, d);
        std::vector<std::thread> threads;
        int rows_per = d / nt;
        int extra = d % nt;
        int row = 0;

        for (int t = 0; t < nt; t++) {
            int start = row;
            row += rows_per + (t < extra ? 1 : 0);
            int end = row;
            threads.emplace_back([=, &out, &wptr, &x]() {
                for (int i = start; i < end; i++) {
                    out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
                }
            });
        }

        for (auto& t : threads) t.join();
    }

    static void matmul_bias(float* out, const float* x, const void* W, const void* b,
                           int n, int d, GGUFType w_type, GGUFType b_type, float* scratch) {
        matmul(out, x, W, n, d, w_type);
        if (b_type != GGUFType::NONE && b != nullptr) {
            dequantize_row(b, scratch, d, b_type);
            vec_add(out, scratch, d);
        }
    }

    // RMS normalization: out[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
    static void rmsnorm(float* out, const float* x, const float* weight, int size) {
        float ss = 0.0f;
        
#ifdef PICOLM_SSE2
        __m128 acc = _mm_setzero_ps();
        int i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 v = _mm_loadu_ps(x + i);
            acc = _mm_add_ps(acc, _mm_mul_ps(v, v));
        }
        ss = hsum_sse(acc);
        for (; i < size; i++) ss += x[i] * x[i];
#else
        for (int i = 0; i < size; i++) ss += x[i] * x[i];
#endif

        ss = 1.0f / std::sqrt(ss / (float)size + 1e-6f);

#ifdef PICOLM_SSE2
        __m128 scale_v = _mm_set1_ps(ss);
        int j = 0;
        for (; j + 3 < size; j += 4) {
            __m128 v = _mm_loadu_ps(x + j);
            __m128 w = _mm_loadu_ps(weight + j);
            _mm_storeu_ps(out + j, _mm_mul_ps(_mm_mul_ps(v, scale_v), w));
        }
        for (; j < size; j++) out[j] = x[j] * ss * weight[j];
#else
        for (int i = 0; i < size; i++) out[i] = x[i] * ss * weight[i];
#endif
    }

    // In-place softmax
    static void softmax(float* x, int size) {
        float max_val = *std::max_element(x, x + size);
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            x[i] = std::exp(x[i] - max_val);
            sum += x[i];
        }
        float inv = 1.0f / sum;

#ifdef PICOLM_SSE2
        __m128 inv_v = _mm_set1_ps(inv);
        int i = 0;
        for (; i + 3 < size; i += 4) {
            _mm_storeu_ps(x + i, _mm_mul_ps(_mm_loadu_ps(x + i), inv_v));
        }
        for (; i < size; i++) x[i] *= inv;
#else
        for (int i = 0; i < size; i++) x[i] *= inv;
#endif
    }

    // RoPE for Qwen3
    static void rope_qwen(float* q, float* k, int head_dim, int n_heads, int n_kv_heads,
                          const float* cos_pos, const float* sin_pos) {
        int half = head_dim / 2;

        for (int h = 0; h < n_heads; h++) {
            float* qh = q + h * head_dim;
            for (int i = 0; i < half; i++) {
                float q0 = qh[i];
                float q1 = qh[i + half];
                qh[i] = q0 * cos_pos[i] - q1 * sin_pos[i];
                qh[i + half] = q0 * sin_pos[i] + q1 * cos_pos[i];
            }
        }

        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = k + h * head_dim;
            for (int i = 0; i < half; i++) {
                float k0 = kh[i];
                float k1 = kh[i + half];
                kh[i] = k0 * cos_pos[i] - k1 * sin_pos[i];
                kh[i + half] = k0 * sin_pos[i] + k1 * cos_pos[i];
            }
        }
    }

    // RoPE for LLaMA
    static void rope_llama(float* q, float* k, int head_dim, int n_heads, int n_kv_heads,
                           const float* cos_pos, const float* sin_pos) {
        int half = head_dim / 2;

        for (int h = 0; h < n_heads; h++) {
            float* qh = q + h * head_dim;
            for (int i = 0; i < half; i++) {
                float q0 = qh[i * 2];
                float q1 = qh[i * 2 + 1];
                qh[i * 2] = q0 * cos_pos[i] - q1 * sin_pos[i];
                qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
            }
        }

        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = k + h * head_dim;
            for (int i = 0; i < half; i++) {
                float k0 = kh[i * 2];
                float k1 = kh[i * 2 + 1];
                kh[i * 2] = k0 * cos_pos[i] - k1 * sin_pos[i];
                kh[i * 2 + 1] = k0 * sin_pos[i] + k1 * cos_pos[i];
            }
        }
    }

    static void silu(float* x, int size) {
        for (int i = 0; i < size; i++) {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
    }

    static void elemwise_mul(float* out, const float* a, const float* b, int size) {
#ifdef PICOLM_SSE2
        int i = 0;
        for (; i + 3 < size; i += 4) {
            _mm_storeu_ps(out + i, _mm_mul_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
        }
        for (; i < size; i++) out[i] = a[i] * b[i];
#else
        for (int i = 0; i < size; i++) out[i] = a[i] * b[i];
#endif
    }

    static void vec_add(float* a, const float* b, int size) {
#ifdef PICOLM_SSE2
        int i = 0;
        for (; i + 3 < size; i += 4) {
            _mm_storeu_ps(a + i, _mm_add_ps(_mm_loadu_ps(a + i), _mm_loadu_ps(b + i)));
        }
        for (; i < size; i++) a[i] += b[i];
#else
        for (int i = 0; i < size; i++) a[i] += b[i];
#endif
    }

private:
    static constexpr int MAX_THREADS = 16;
    static inline int n_threads = 1;
};

#endif // TENSOR_HPP
