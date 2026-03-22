#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "quant.hpp"
#include <vector>
#include <thread>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>

class TensorOps {
private:
    static constexpr int MAX_THREADS = 16;
    static int n_threads;

    // Thread pool for parallel operations
    struct ThreadPool {
        std::vector<std::thread> workers;
        std::vector<std::function<void()>> tasks;
        std::mutex queue_mutex;
        std::condition_variable condition;
        std::condition_variable done_condition;
        std::atomic<bool> stop{false};
        int tasks_done{0};
        int tasks_expected{0};

        ThreadPool(int n) {
            if (n <= 1) return;
            for (int i = 0; i < n; i++) {
                workers.emplace_back([this]() {
                    while (true) {
                        std::function<void()> task;
                        {
                            std::unique_lock<std::mutex> lock(queue_mutex);
                            condition.wait(lock, [this]() {
                                return stop || !tasks.empty();
                            });
                            if (stop && tasks.empty()) return;
                            task = std::move(tasks.back());
                            tasks.pop_back();
                        }
                        task();
                        {
                            std::lock_guard<std::mutex> lock(queue_mutex);
                            tasks_done++;
                            if (tasks_done >= tasks_expected) {
                                done_condition.notify_all();
                            }
                        }
                    }
                });
            }
        }

        ~ThreadPool() {
            stop = true;
            condition.notify_all();
            for (auto& w : workers) {
                if (w.joinable()) w.join();
            }
        }

        void run_all(int n, std::function<void(int)> task_fn) {
            if (n <= 1 || workers.empty()) {
                for (int i = 0; i < n; i++) task_fn(i);
                return;
            }

            tasks_done = 0;
            tasks_expected = n;

            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                for (int i = 0; i < n; i++) {
                    tasks.emplace_back([i, &task_fn]() {
                        task_fn(i);
                    });
                }
            }
            condition.notify_all();

            // Wait for all tasks to complete
            std::unique_lock<std::mutex> lock(queue_mutex);
            done_condition.wait(lock, [this]() {
                return tasks_done >= tasks_expected;
            });
        }
    };
    
    static ThreadPool* pool;
    
public:
    static void init_thread_pool(int t) {
        n_threads = std::max(1, std::min(t, MAX_THREADS));
        if (pool) delete pool;
        pool = new ThreadPool(n_threads);
    }
    
    static void cleanup_thread_pool() {
        if (pool) {
            delete pool;
            pool = nullptr;
        }
    }

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
        
        // Use thread pool to avoid thread creation overhead
        pool->run_all(nt, [&](int t) {
            int rows_per = d / nt;
            int extra = d % nt;
            int start = t * rows_per + std::min(t, extra);
            int end = start + rows_per + (t < extra ? 1 : 0);
            
            for (int i = start; i < end; i++) {
                out[i] = vec_dot(wptr + (size_t)i * row_bytes, x, n, qtype);
            }
        });
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

    // RoPE for Qwen3 - SIMD optimized
    static void rope_qwen(float* q, float* k, int head_dim, int n_heads, int n_kv_heads,
                          const float* cos_pos, const float* sin_pos) {
        int half = head_dim / 2;

#ifdef PICOLM_SSE2
        // Process Q with SIMD
        for (int h = 0; h < n_heads; h++) {
            float* qh = q + h * head_dim;
            int i = 0;
            
            // Process 4 elements at a time (2 pairs)
            for (; i + 3 < half; i += 4) {
                __m128 q0 = _mm_loadu_ps(qh + i);
                __m128 q1 = _mm_loadu_ps(qh + i + half);
                __m128 cos = _mm_loadu_ps(cos_pos + i);
                __m128 sin = _mm_loadu_ps(sin_pos + i);
                
                // qh[i] = q0 * cos - q1 * sin
                __m128 out0 = _mm_sub_ps(_mm_mul_ps(q0, cos), _mm_mul_ps(q1, sin));
                // qh[i+half] = q0 * sin + q1 * cos
                __m128 out1 = _mm_add_ps(_mm_mul_ps(q0, sin), _mm_mul_ps(q1, cos));
                
                _mm_storeu_ps(qh + i, out0);
                _mm_storeu_ps(qh + i + half, out1);
            }
            
            // Scalar fallback
            for (; i < half; i++) {
                float q0 = qh[i];
                float q1 = qh[i + half];
                qh[i] = q0 * cos_pos[i] - q1 * sin_pos[i];
                qh[i + half] = q0 * sin_pos[i] + q1 * cos_pos[i];
            }
        }

        // Process K with SIMD
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = k + h * head_dim;
            int i = 0;
            
            for (; i + 3 < half; i += 4) {
                __m128 k0 = _mm_loadu_ps(kh + i);
                __m128 k1 = _mm_loadu_ps(kh + i + half);
                __m128 cos = _mm_loadu_ps(cos_pos + i);
                __m128 sin = _mm_loadu_ps(sin_pos + i);
                
                __m128 out0 = _mm_sub_ps(_mm_mul_ps(k0, cos), _mm_mul_ps(k1, sin));
                __m128 out1 = _mm_add_ps(_mm_mul_ps(k0, sin), _mm_mul_ps(k1, cos));
                
                _mm_storeu_ps(kh + i, out0);
                _mm_storeu_ps(kh + i + half, out1);
            }
            
            for (; i < half; i++) {
                float k0 = kh[i];
                float k1 = kh[i + half];
                kh[i] = k0 * cos_pos[i] - k1 * sin_pos[i];
                kh[i + half] = k0 * sin_pos[i] + k1 * cos_pos[i];
            }
        }
#else
        // Scalar version
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
#endif
    }

    // RoPE for LLaMA - SIMD optimized
    static void rope_llama(float* q, float* k, int head_dim, int n_heads, int n_kv_heads,
                           const float* cos_pos, const float* sin_pos) {
        int half = head_dim / 2;

#ifdef PICOLM_SSE2
        // Process Q with SIMD
        for (int h = 0; h < n_heads; h++) {
            float* qh = q + h * head_dim;
            int i = 0;
            
            // Process 4 pairs (8 elements) at a time
            for (; i + 3 < half; i += 4) {
                // Load q[2*i:2*i+7] - interleaved format
                __m128 q_even = _mm_loadu_ps(qh + i * 2);      // q0[0], q0[1], q0[2], q0[3]
                __m128 q_odd = _mm_loadu_ps(qh + i * 2 + 4);   // q1[0], q1[1], q1[2], q1[3]
                __m128 cos = _mm_loadu_ps(cos_pos + i);
                __m128 sin = _mm_loadu_ps(sin_pos + i);
                
                // Compute: out_even = q_even * cos - q_odd * sin
                __m128 out_even = _mm_sub_ps(_mm_mul_ps(q_even, cos), _mm_mul_ps(q_odd, sin));
                // Compute: out_odd = q_even * sin + q_odd * cos
                __m128 out_odd = _mm_add_ps(_mm_mul_ps(q_even, sin), _mm_mul_ps(q_odd, cos));
                
                _mm_storeu_ps(qh + i * 2, out_even);
                _mm_storeu_ps(qh + i * 2 + 4, out_odd);
            }
            
            // Scalar fallback
            for (; i < half; i++) {
                float q0 = qh[i * 2];
                float q1 = qh[i * 2 + 1];
                qh[i * 2] = q0 * cos_pos[i] - q1 * sin_pos[i];
                qh[i * 2 + 1] = q0 * sin_pos[i] + q1 * cos_pos[i];
            }
        }

        // Process K with SIMD
        for (int h = 0; h < n_kv_heads; h++) {
            float* kh = k + h * head_dim;
            int i = 0;
            
            for (; i + 3 < half; i += 4) {
                __m128 k_even = _mm_loadu_ps(kh + i * 2);
                __m128 k_odd = _mm_loadu_ps(kh + i * 2 + 4);
                __m128 cos = _mm_loadu_ps(cos_pos + i);
                __m128 sin = _mm_loadu_ps(sin_pos + i);
                
                __m128 out_even = _mm_sub_ps(_mm_mul_ps(k_even, cos), _mm_mul_ps(k_odd, sin));
                __m128 out_odd = _mm_add_ps(_mm_mul_ps(k_even, sin), _mm_mul_ps(k_odd, cos));
                
                _mm_storeu_ps(kh + i * 2, out_even);
                _mm_storeu_ps(kh + i * 2 + 4, out_odd);
            }
            
            for (; i < half; i++) {
                float k0 = kh[i * 2];
                float k1 = kh[i * 2 + 1];
                kh[i * 2] = k0 * cos_pos[i] - k1 * sin_pos[i];
                kh[i * 2 + 1] = k0 * sin_pos[i] + k1 * cos_pos[i];
            }
        }
#else
        // Scalar version
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
#endif
    }

    static void silu(float* x, int size) {
#ifdef PICOLM_SSE2
        int i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 v = _mm_loadu_ps(x + i);
            // Compute -x
            __m128 neg_v = _mm_sub_ps(_mm_setzero_ps(), v);
            // Compute exp(-x) using approximation or library call
            // For simplicity, use scalar fallback for exp
            float arr[4];
            _mm_storeu_ps(arr, neg_v);
            for (int j = 0; j < 4; j++) {
                arr[j] = 1.0f / (1.0f + std::exp(arr[j]));
            }
            __m128 result = _mm_mul_ps(v, _mm_loadu_ps(arr));
            _mm_storeu_ps(x + i, result);
        }
        for (; i < size; i++) {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
#else
        for (int i = 0; i < size; i++) {
            x[i] = x[i] / (1.0f + std::exp(-x[i]));
        }
#endif
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
};

// Static member definitions
int TensorOps::n_threads = 1;
TensorOps::ThreadPool* TensorOps::pool = nullptr;

#endif // TENSOR_HPP
