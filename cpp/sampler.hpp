#ifndef SAMPLER_HPP
#define SAMPLER_HPP

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include "tensor.hpp"

struct Sampler {
    float temperature = 0.8f;
    float top_p = 0.9f;
    uint64_t rng_state = 42;

    void init(float temp, float top_p_val, uint64_t seed) {
        temperature = temp;
        top_p = top_p_val;
        rng_state = seed ? seed : 42;
    }

    int sample(float* logits, int vocab_size) {
        if (temperature <= 0.0f) {
            int best = 0;
            for (int i = 1; i < vocab_size; i++) {
                if (logits[i] > logits[best]) best = i;
            }
            return best;
        }

        float inv_temp = 1.0f / temperature;
        for (int i = 0; i < vocab_size; i++) {
            logits[i] *= inv_temp;
        }

        TensorOps::softmax(logits, vocab_size);

        if (top_p >= 1.0f) {
            float r = rand_float();
            float cum = 0.0f;
            for (int i = 0; i < vocab_size; i++) {
                cum += logits[i];
                if (cum > r) return i;
            }
            return vocab_size - 1;
        }

        // Top-p sampling
        std::vector<std::pair<float, int>> sorted(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            sorted[i] = {logits[i], i};
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;
        });

        float cum = 0.0f;
        int cutoff = 0;
        for (int i = 0; i < vocab_size; i++) {
            cum += sorted[i].first;
            cutoff = i + 1;
            if (cum >= top_p) break;
        }

        float r = rand_float() * cum;
        float acc = 0.0f;
        int result = sorted[0].second;
        for (int i = 0; i < cutoff; i++) {
            acc += sorted[i].first;
            if (acc > r) {
                result = sorted[i].second;
                break;
            }
        }
        return result;
    }

private:
    uint64_t xorshift64() {
        uint64_t x = rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        rng_state = x;
        return x;
    }

    float rand_float() {
        return (float)(xorshift64() >> 11) / (float)(1ULL << 53);
    }
};

#endif // SAMPLER_HPP
