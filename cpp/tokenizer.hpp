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

    // Byte-level BPE: Unicode code point to byte mapping (GPT-2/Qwen style)
    // This reverses the byte_to_unicode mapping used in tokenizer training
    static int unicode_to_byte(int cp) {
        // GPT-2 byte-to-unicode mapping:
        // - Printable ASCII (33-126) stay as is
        // - Latin-1 supplement (161-172, 174-255) stay as is  
        // - Other bytes (0-32, 127-160, 173) are shifted to 256+
        
        // Direct mappings (these code points represent themselves)
        if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255)) {
            return cp;
        }
        
        // Shifted mappings: code points 256+ represent bytes that were shifted
        // We need to find which byte maps to this code point
        int n = 0;
        for (int b = 0; b < 256; b++) {
            bool skip = false;
            if (b >= 33 && b <= 126) skip = true;
            if (b >= 161 && b <= 172) skip = true;
            if (b >= 174 && b <= 255) skip = true;
            
            if (!skip) {
                if (cp == 256 + n) return b;
                n++;
            }
        }
        
        // Fallback
        return cp & 0xFF;
    }

    // Recover raw bytes from vocab string (reverse of byte_to_unicode encoding)
    // This converts the "mangled" UTF-8 back to original byte sequence
    static std::string recover_raw_bytes(const std::string& s) {
        std::string res;
        res.reserve(s.size());
        
        for (size_t i = 0; i < s.size(); ) {
            unsigned char c = (unsigned char)s[i];
            uint32_t cp = 0;
            int len = 0;
            
            // UTF-8 decode to get Unicode code point
            if (c < 0x80) {
                // Single byte (ASCII)
                cp = c;
                len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                // 2-byte UTF-8
                if (i + 1 >= s.size()) { i++; continue; }
                cp = c & 0x1F;
                len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                // 3-byte UTF-8
                if (i + 2 >= s.size()) { i++; continue; }
                cp = c & 0x0F;
                len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                // 4-byte UTF-8
                if (i + 3 >= s.size()) { i++; continue; }
                cp = c & 0x07;
                len = 4;
            } else {
                // Invalid UTF-8 lead byte, treat as raw byte
                res += (char)c;
                i++;
                continue;
            }
            
            // Complete UTF-8 decode
            for (int j = 1; j < len && i + j < s.size(); j++) {
                cp = (cp << 6) | ((unsigned char)s[i + j] & 0x3F);
            }
            i += len;
            
            // Map Unicode code point back to original byte
            res += (char)unicode_to_byte((int)cp);
        }
        
        return res;
    }
    
    // Encode UTF-8 string to double-encoded form (for matching against fixed vocab)
    static std::string to_double_utf8(const std::string& s) {
        std::string result;
        result.reserve(s.size() * 2);
        
        for (size_t i = 0; i < s.size(); i++) {
            unsigned char c = (unsigned char)s[i];
            if (c < 0x80) {
                // ASCII stays as is
                result += (char)c;
            } else {
                // High byte gets UTF-8 encoded as if it were Latin-1
                // 0x80-0xFF -> U+0080-U+00FF -> UTF-8: C2 80-C3 BF
                unsigned char c1 = 0xC0 | (c >> 6);
                unsigned char c2 = 0x80 | (c & 0x3F);
                // Fix: 0x80-0xBF should use C2, 0xC0-0xFF should use C3
                if (c < 0xC0) {
                    c1 = 0xC2;
                } else {
                    c1 = 0xC3;
                }
                result += (char)c1;
                result += (char)c2;
            }
        }
        
        return result;
    }

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
                    std::string raw((const char*)p, (size_t)slen);
                    // Recover raw bytes from byte-level BPE encoded vocab string
                    vocab[i] = recover_raw_bytes(raw);
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

        // Step 1: Greedy longest match tokenization
        std::vector<int> merge_buf;
        merge_buf.reserve((size_t)text_len + 1);

        int i = 0;
        while (i < text_len) {
            // Try to find the longest matching token starting at position i
            int best_len = 0;
            int best_tok = -1;

            // Try lengths from 1 up to remaining text (or max token length)
            int max_try = text_len - i;
            if (max_try > 64) max_try = 64;  // Limit max token length to check

            for (int try_len = max_try; try_len >= 1; --try_len) {
                int tok = vocab_lookup(text + i, try_len);
                if (tok >= 0) {
                    best_len = try_len;
                    best_tok = tok;
                    break;  // Found longest match
                }
            }

            if (best_tok >= 0) {
                merge_buf.push_back(best_tok);
                i += best_len;
            } else {
                // Fall back to byte token for unknown characters
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)text[i]);
                int tok = vocab_lookup(byte_tok, (int)strlen(byte_tok));
                if (tok >= 0) {
                    merge_buf.push_back(tok);
                }
                i++;
            }
        }

        // Copy to output (no BPE merge needed since we already did longest match)
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

            // 处理字节级 BPE 转义符 <0xXX>
            if (c == '<' && i + 5 < str.size() && str[i+1] == '0' && str[i+2] == 'x') {
                unsigned int val = 0;
                if (sscanf(str.c_str() + i, "<0x%02X>", &val) == 1) {
                    clean_buf += (char)val;
                    i += 5;
                    continue;
                }
            }

            // 直接透传原始字节
            // vocab 已经通过 recover_raw_bytes 还原为原始字节序列
            // 这些字节就是正确的 UTF-8 编码（中文等多字节字符会正确组合）
            clean_buf += str[i];
        }
        return clean_buf.c_str();
    }
};

#endif // TOKENIZER_HPP
