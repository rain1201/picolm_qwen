#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <cstdio>

struct Tokenizer {
    std::vector<std::string> vocab;
    std::vector<float> scores;
    std::vector<int> sorted_idx;
    uint32_t bos_id = 1;
    uint32_t eos_id = 2;
    int vocab_size = 0;

    int load(const void* tokens_data, uint64_t n_tokens,
             const void* scores_data, uint64_t n_scores,
             uint32_t bos, uint32_t eos, int vs) {
        vocab_size = vs;
        bos_id = bos;
        eos_id = eos;

        vocab.resize((size_t)vs);
        scores.resize((size_t)vs);
        sorted_idx.resize((size_t)vs);

        if (tokens_data && n_tokens > 0) {
            const uint8_t* p = (const uint8_t*)tokens_data;
            uint64_t n = std::min((uint64_t)vs, n_tokens);
            for (uint64_t i = 0; i < n; i++) {
                uint64_t slen;
                std::memcpy(&slen, p, 8);
                p += 8;
                if (slen > 0) {
                    vocab[i].assign((const char*)p, (size_t)slen);
                }
                p += slen;
            }
        }
        for (int i = 0; i < vs; i++) {
            if (vocab[i].empty()) vocab[i] = "";
        }

        if (scores_data && n_scores > 0) {
            uint64_t n = std::min((uint64_t)vs, n_scores);
            std::memcpy(scores.data(), scores_data, (size_t)n * sizeof(float));
        }

        for (int i = 0; i < vs; i++) sorted_idx[i] = i;
        std::sort(sorted_idx.begin(), sorted_idx.end(), [this](int a, int b) {
            return vocab[a] < vocab[b];
        });

        return 0;
    }

    int vocab_lookup(const char* str, int len) const {
        int lo = 0, hi = vocab_size - 1;
        while (lo <= hi) {
            int mid = (lo + hi) / 2;
            int idx = sorted_idx[mid];
            const std::string& v = vocab[idx];
            int cmp = strncmp(v.c_str(), str, (size_t)len);
            if (cmp == 0) {
                if ((int)v.size() == len) return idx;
                if (v[(size_t)len] > '\0') { hi = mid - 1; }
                else { lo = mid + 1; }
            } else if (cmp < 0) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }

    // BPE encode for Qwen - matches original C implementation
    int encode_qwen(const char* text, std::vector<int>& tokens, bool add_bos) const {
        tokens.clear();
        if (add_bos) tokens.push_back((int)bos_id);
        if (!text || !*text) return (int)tokens.size();

        int text_len = (int)strlen(text);
        
        // Step 1: Convert text to initial token sequence (char by char)
        std::vector<int> merge_buf;
        merge_buf.reserve((size_t)text_len * 3 + 1);

        for (int i = 0; i < text_len; ) {
            int clen = 1;
            unsigned char c = (unsigned char)text[i];
            if (c >= 0xF0) clen = 4;
            else if (c >= 0xE0) clen = 3;
            else if (c >= 0xC0) clen = 2;
            if (i + clen > text_len) clen = text_len - i;

            int tok = vocab_lookup(text + i, clen);
            if (tok >= 0) {
                merge_buf.push_back(tok);
                i += clen;
            } else {
                // Fall back to byte token
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)text[i]);
                tok = vocab_lookup(byte_tok, (int)strlen(byte_tok));
                if (tok >= 0) merge_buf.push_back(tok);
                i++;
            }
        }

        // Step 2: BPE merge loop
        while ((int)merge_buf.size() >= 2) {
            float best_score = -1e30f;
            int best_idx = -1;
            int best_tok = -1;

            for (int i = 0; i < (int)merge_buf.size() - 1; i++) {
                const std::string& s1 = vocab[merge_buf[i]];
                const std::string& s2 = vocab[merge_buf[i + 1]];
                
                // Check if concatenation fits
                if (s1.size() + s2.size() > 256) continue;
                
                std::string merged = s1 + s2;
                int tok = vocab_lookup(merged.c_str(), (int)merged.size());
                if (tok >= 0 && scores[tok] > best_score) {
                    best_score = scores[tok];
                    best_idx = i;
                    best_tok = tok;
                }
            }

            if (best_idx < 0) break;
            
            // Apply merge
            merge_buf[best_idx] = best_tok;
            merge_buf.erase(merge_buf.begin() + best_idx + 1);
        }

        // Copy to output
        for (int tok : merge_buf) {
            tokens.push_back(tok);
        }

        return (int)tokens.size();
    }

    const char* decode_qwen(int token) const {
        if (token < 0 || token >= vocab_size) return "";
        const std::string& str = vocab[token];
        static thread_local std::string clean_buf;
        clean_buf.clear();

        for (size_t i = 0; i < str.size(); i++) {
            unsigned char c = (unsigned char)str[i];
            // Handle Qwen byte-level BPE mapping
            // Ġ (U+0100 = C4 A0 in UTF-8) -> space
            if (c == 0xC4 && i + 1 < str.size()) {
                if ((unsigned char)str[i+1] == 0xA0) {
                    clean_buf += ' ';
                    i++;
                    continue;
                }
                // Ċ (U+010A = C4 8A in UTF-8) -> newline  
                if ((unsigned char)str[i+1] == 0x8A) {
                    clean_buf += '\n';
                    i++;
                    continue;
                }
            }
            // Handle byte tokens <0xHH>
            if (c == '<' && i + 5 < str.size() && str[i+1] == '0' && str[i+2] == 'x') {
                unsigned int val = 0;
                if (sscanf(str.c_str() + i, "<0x%02X>", &val) == 1) {
                    clean_buf += (char)val;
                    i += 5;
                    continue;
                }
            }
            // Regular character - check if it's a multi-byte UTF-8
            clean_buf += str[i];
        }
        return clean_buf.c_str();
    }
};

#endif // TOKENIZER_HPP
