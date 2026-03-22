#ifndef QUANT_HPP
#define QUANT_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <immintrin.h>

// SIMD detection for x86
#if defined(__SSE2__) || (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_AMD64)))
#define PICOLM_SSE2
#endif

enum class GGUFType : uint32_t {
    NONE   = 0,
    F32    = 1,
    F16    = 2,
    Q4_0   = 3,
    Q4_1   = 4,
    Q5_0   = 7,
    Q5_1   = 8,
    Q8_0   = 9,
    Q8_1   = 10,
    Q2_K   = 11,
    Q3_K   = 12,
    Q4_K   = 13,
    Q5_K   = 14,
    Q6_K   = 15,
};

#pragma pack(push, 1)
struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};

struct block_q3_K {
    uint16_t d;
    uint8_t  qs[64];
    uint8_t  hmask[32];
    uint8_t  scales[12];
};

struct block_q2_K {
    uint8_t  scales[16];
    uint8_t  qs[64];
    uint16_t d;
    uint16_t dmin;
};

struct block_q8_0 {
    uint16_t d;
    int8_t   qs[32];
};

struct block_q6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};

struct block_q4_0 {
    uint16_t d;
    uint8_t  qs[16];
};
#pragma pack(pop)

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;

    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400)) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }

    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

inline uint16_t fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    uint32_t sign = (bits >> 16) & 0x8000;
    int      exp  = (int)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (((bits >> 23) & 0xFF) == 0) {
        return (uint16_t)sign;
    }
    if (((bits >> 23) & 0xFF) == 0xFF) {
        return (uint16_t)(sign | 0x7C00 | (mant ? 0x0200 : 0));
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00);
    }
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t round_bit = 1U << (shift - 1);
        mant = (mant + round_bit) >> shift;
        return (uint16_t)(sign | mant);
    }

    mant += 0x00001000;
    if (mant & 0x00800000) {
        mant = 0;
        exp++;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

inline int gguf_type_block_size(GGUFType type) {
    switch (type) {
        case GGUFType::F32: case GGUFType::F16: return 1;
        case GGUFType::Q4_0: case GGUFType::Q4_1:
        case GGUFType::Q5_0: case GGUFType::Q5_1:
        case GGUFType::Q8_0: case GGUFType::Q8_1: return 32;
        case GGUFType::Q2_K: case GGUFType::Q3_K:
        case GGUFType::Q4_K: case GGUFType::Q5_K:
        case GGUFType::Q6_K: return 256;
        default: return 0;
    }
}

inline int gguf_type_quant_size(GGUFType type) {
    switch (type) {
        case GGUFType::F32: return 4;
        case GGUFType::F16: return 2;
        case GGUFType::Q4_0: return 18;
        case GGUFType::Q4_1: return 20;
        case GGUFType::Q5_0: return 22;
        case GGUFType::Q5_1: return 24;
        case GGUFType::Q8_0: return 34;
        case GGUFType::Q8_1: return 40;
        case GGUFType::Q2_K: return 84;
        case GGUFType::Q3_K: return 110;
        case GGUFType::Q4_K: return 144;
        case GGUFType::Q5_K: return 176;
        case GGUFType::Q6_K: return 210;
        default: return 0;
    }
}

inline size_t gguf_type_row_size(GGUFType type, int n) {
    int bs = gguf_type_block_size(type);
    int qs = gguf_type_quant_size(type);
    if (bs == 0 || qs == 0) return 0;
    return (size_t)(n / bs) * qs;
}

inline void dequantize_row_f32(const void* src, float* dst, int n) {
    std::memcpy(dst, src, n * sizeof(float));
}

inline void dequantize_row_f16(const void* src, float* dst, int n) {
    const uint16_t* fp16 = (const uint16_t*)src;
    for (int i = 0; i < n; i++) {
        dst[i] = fp16_to_fp32(fp16[i]);
    }
}

inline void dequantize_row_q8_0(const void* src, float* dst, int n) {
    const block_q8_0* blocks = (const block_q8_0*)src;
    int nb = n / 32;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            dst[i * 32 + j] = d * (float)blocks[i].qs[j];
        }
    }
}

inline void dequantize_row_q4_0(const void* src, float* dst, int n) {
    const block_q4_0* blocks = (const block_q4_0*)src;
    int nb = n / 32;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        for (int j = 0; j < 32; j++) {
            uint8_t nibble = (j < 16) ? (blocks[i].qs[j] & 0xF) : (blocks[i].qs[j - 16] >> 4);
            dst[i * 32 + j] = d * ((float)nibble - 8.0f);
        }
    }
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* sc, uint8_t* mn) {
    if (j < 4) {
        *sc = q[j] & 63;
        *mn = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

inline void dequantize_row_q4_K(const void* src, float* dst, int n) {
    const block_q4_K* blocks = (const block_q4_K*)src;
    int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        const block_q4_K* b = &blocks[i];
        float d = fp16_to_fp32(b->d);
        float dmin = fp16_to_fp32(b->dmin);
        const uint8_t* q = b->qs;
        float* y = dst + i * 256;

        int is = 0;
        for (int j = 0; j < 4; j++) {
            uint8_t sc, mn;
            get_scale_min_k4(is, b->scales, &sc, &mn);
            float d1 = d * (float)sc;
            float m1 = dmin * (float)mn;
            get_scale_min_k4(is + 1, b->scales, &sc, &mn);
            float d2 = d * (float)sc;
            float m2 = dmin * (float)mn;

            for (int l = 0; l < 32; l++) {
                y[l] = d1 * (float)(q[l] & 0xF) - m1;
            }
            for (int l = 0; l < 32; l++) {
                y[l + 32] = d2 * (float)(q[l] >> 4) - m2;
            }
            y += 64;
            q += 32;
            is += 2;
        }
    }
}

inline void dequantize_row_q6_K(const void* src, float* dst, int n) {
    const block_q6_K* blocks = (const block_q6_K*)src;
    int nb = n / 256;
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        const uint8_t* ql = blocks[i].ql;
        const uint8_t* qh = blocks[i].qh;
        const int8_t* sc = blocks[i].scales;
        float* y = dst + i * 256;

        for (int chunk = 0; chunk < 256; chunk += 128) {
            int is = chunk / 16;
            for (int l = 0; l < 32; l++) {
                int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                int is_l = is + (l / 16);
                y[l]      = d * (float)sc[is_l + 0] * (float)q1;
                y[l + 32] = d * (float)sc[is_l + 2] * (float)q2;
                y[l + 64] = d * (float)sc[is_l + 4] * (float)q3;
                y[l + 96] = d * (float)sc[is_l + 6] * (float)q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
        }
    }
}

inline void dequantize_row(const void* src, float* dst, int n, GGUFType type) {
    switch (type) {
        case GGUFType::F32:  dequantize_row_f32(src, dst, n); break;
        case GGUFType::F16:  dequantize_row_f16(src, dst, n); break;
        case GGUFType::Q4_0: dequantize_row_q4_0(src, dst, n); break;
        case GGUFType::Q8_0: dequantize_row_q8_0(src, dst, n); break;
        case GGUFType::Q4_K: dequantize_row_q4_K(src, dst, n); break;
        case GGUFType::Q6_K: dequantize_row_q6_K(src, dst, n); break;
        default:
            throw std::runtime_error("Unsupported quantization type");
    }
}

inline float hsum_sse(__m128 v) {
    __m128 shuf = _mm_movehl_ps(v, v);
    __m128 sum  = _mm_add_ps(v, shuf);
    shuf = _mm_shuffle_ps(sum, sum, 1);
    sum  = _mm_add_ss(sum, shuf);
    return _mm_cvtss_f32(sum);
}

inline float vec_dot_f32_f32(const void* src, const float* x, int n) {
    const float* w = (const float*)src;
    
#ifdef PICOLM_SSE2
    __m128 acc0 = _mm_setzero_ps();
    __m128 acc1 = _mm_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc0 = _mm_add_ps(acc0, _mm_mul_ps(_mm_loadu_ps(w + i), _mm_loadu_ps(x + i)));
        acc1 = _mm_add_ps(acc1, _mm_mul_ps(_mm_loadu_ps(w + i + 4), _mm_loadu_ps(x + i + 4)));
    }
    float sum = hsum_sse(_mm_add_ps(acc0, acc1));
    for (; i < n; i++) sum += w[i] * x[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += w[i] * x[i];
    return sum;
#endif
}

inline float vec_dot_q8_0_f32(const void* src, const float* x, int n) {
    const block_q8_0* blocks = (const block_q8_0*)src;
    int nb = n / 32;
    float sumf = 0.0f;
    
    for (int i = 0; i < nb; i++) {
        float d = fp16_to_fp32(blocks[i].d);
        const int8_t* q = blocks[i].qs;
        const float* xp = x + i * 32;
        
        // Scalar fallback (same as original C code)
        float sum = 0.0f;
        for (int j = 0; j < 32; j++) {
            sum += (float)q[j] * xp[j];
        }
        sumf += d * sum;
    }
    return sumf;
}

inline float vec_dot(const void* src, const float* x, int n, GGUFType type) {
    switch (type) {
        case GGUFType::F32:  return vec_dot_f32_f32(src, x, n);
        case GGUFType::Q8_0: return vec_dot_q8_0_f32(src, x, n);
        default: {
            float tmp[8192];
            float* buf = (n <= 8192) ? tmp : new float[(size_t)n];
            dequantize_row(src, buf, n, type);
            float sum = vec_dot_f32_f32(buf, x, n);
            if (buf != tmp) delete[] buf;
            return sum;
        }
    }
}

#endif // QUANT_HPP
