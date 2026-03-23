#ifndef TOKENIZER_HPP
#define TOKENIZER_HPP

#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <cstdio>
#include <unordered_map>

struct Tokenizer {
    std::vector<std::string> vocab;
    std::vector<float> scores;
    std::vector<int> sorted_idx;
    uint32_t bos_id = 1;
    uint32_t eos_id = 2;
    int vocab_size = 0;
    
    // Byte-to-Unicode mapping (GPT-2/Qwen style)
    std::unordered_map<unsigned char, uint32_t> byte_to_unicode_map;
    std::unordered_map<uint32_t, unsigned char> unicode_to_byte_map;
    bool mappings_initialized = false;
    
    // Initialize byte-to-unicode mapping tables
    void init_byte_mappings() {
        if (mappings_initialized) return;
        
        auto insert_range = [&](int start, int end) {
            for (int b = start; b <= end; b++) {
                byte_to_unicode_map[b] = b;
                unicode_to_byte_map[b] = b;
            }
        };
        
        // GPT-2/Qwen standard mapping: visible chars map to themselves
        insert_range('!', '~');      // 33-126
        insert_range(0xA1, 0xAC);    // 161-172
        insert_range(0xAE, 0xFF);    // 174-255
        
        // Other bytes map to 256+
        int n = 256;
        for (int b = 0; b < 256; b++) {
            if (byte_to_unicode_map.find(b) == byte_to_unicode_map.end()) {
                byte_to_unicode_map[b] = n;
                unicode_to_byte_map[n] = b;
                n++;
            }
        }
        
        mappings_initialized = true;
    }
    
    // Convert a byte to its mapped Unicode code point
    uint32_t byte_to_unicode(unsigned char b) const {
        if (!mappings_initialized) const_cast<Tokenizer*>(this)->init_byte_mappings();
        auto it = byte_to_unicode_map.find(b);
        if (it != byte_to_unicode_map.end()) return it->second;
        return b; // fallback
    }
    
    // Convert a mapped Unicode code point back to byte
    unsigned char unicode_to_byte(uint32_t cp) const {
        if (!mappings_initialized) const_cast<Tokenizer*>(this)->init_byte_mappings();
        auto it = unicode_to_byte_map.find(cp);
        if (it != unicode_to_byte_map.end()) return it->second;
        return (unsigned char)cp; // fallback
    }
    
    // Encode a Unicode code point to UTF-8 string
    std::string codepoint_to_utf8(uint32_t cp) const {
        std::string res;
        if (cp < 0x80) {
            res += (char)cp;
        } else if (cp < 0x800) {
            res += (char)(0xC0 | (cp >> 6));
            res += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            res += (char)(0xE0 | (cp >> 12));
            res += (char)(0x80 | ((cp >> 6) & 0x3F));
            res += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x110000) {
            res += (char)(0xF0 | (cp >> 18));
            res += (char)(0x80 | ((cp >> 12) & 0x3F));
            res += (char)(0x80 | ((cp >> 6) & 0x3F));
            res += (char)(0x80 | (cp & 0x3F));
        }
        return res;
    }
    
    // Decode UTF-8 to Unicode code point, returns number of bytes consumed
    int utf8_to_codepoint(const char* s, int len, uint32_t& cp) const {
        unsigned char c = (unsigned char)s[0];
        if (c < 0x80) {
            cp = c;
            return 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (len < 2) return -1;
            cp = (c & 0x1F);
            cp = (cp << 6) | ((unsigned char)s[1] & 0x3F);
            return 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (len < 3) return -1;
            cp = (c & 0x0F);
            cp = (cp << 6) | ((unsigned char)s[1] & 0x3F);
            cp = (cp << 6) | ((unsigned char)s[2] & 0x3F);
            return 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (len < 4) return -1;
            cp = (c & 0x07);
            cp = (cp << 6) | ((unsigned char)s[1] & 0x3F);
            cp = (cp << 6) | ((unsigned char)s[2] & 0x3F);
            cp = (cp << 6) | ((unsigned char)s[3] & 0x3F);
            return 4;
        }
        return -1;
    }
    
    // Convert mangled (byte-mapped) vocab string back to raw bytes
    std::string mangled_to_raw_bytes(const std::string& s) const {
        std::string res;
        res.reserve(s.size());
        
        for (size_t i = 0; i < s.size(); ) {
            uint32_t cp = 0;
            int len = utf8_to_codepoint(s.c_str() + i, (int)(s.size() - i), cp);
            if (len <= 0) {
                i++;
                continue;
            }
            res += (char)unicode_to_byte(cp);
            i += len;
        }
        
        return res;
    }
    
    // Convert raw bytes to mangled (byte-mapped) string for vocab lookup
    std::string raw_to_mangled(const char* bytes, size_t len) const {
        std::string res;
        res.reserve(len * 3); // worst case: each byte becomes 3-byte UTF-8
        
        for (size_t i = 0; i < len; i++) {
            unsigned char b = (unsigned char)bytes[i];
            uint32_t cp = byte_to_unicode(b);
            res += codepoint_to_utf8(cp);
        }
        
        return res;
    }

    int load(const void* tokens_data, uint64_t n_tokens,
             const void* scores_data, uint64_t n_scores,
             uint32_t bos, uint32_t eos, int vs) {
        vocab_size = vs;
        bos_id = bos;
        eos_id = eos;
        mappings_initialized = false;
        byte_to_unicode_map.clear();
        unicode_to_byte_map.clear();

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
                    // Vocab stores mangled (byte-mapped) strings directly
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
                // Prefix matches, check if lengths match exactly
                if ((int)v.size() == len) return idx;
                // If v is longer than str, v starts with str, search left for exact match
                if ((int)v.size() > len) { hi = mid - 1; }
                else { lo = mid + 1; }
            } else if (cmp < 0) {
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        return -1;
    }

    // BPE encode for Qwen - converts input to mangled form for vocab matching
    int encode_qwen(const char* text, std::vector<int>& tokens, bool add_bos) const {
        tokens.clear();
        if (add_bos) tokens.push_back((int)bos_id);
        if (!text || !*text) return (int)tokens.size();
        
        // Initialize byte mappings
        if (!mappings_initialized) const_cast<Tokenizer*>(this)->init_byte_mappings();

        // Step 1: Convert raw UTF-8 input to mangled (byte-mapped) form
        size_t text_len = strlen(text);
        std::string mangled = raw_to_mangled(text, text_len);
        const char* m_ptr = mangled.c_str();
        int m_len = (int)mangled.size();

        // Step 2: Greedy longest match on mangled string
        std::vector<int> merge_buf;
        merge_buf.reserve((size_t)m_len + 1);

        int i = 0;
        while (i < m_len) {
            int best_len = 0;
            int best_tok = -1;

            // Try lengths from 1 up to remaining (limit to reasonable max)
            int max_try = m_len - i;
            if (max_try > 48) max_try = 48;

            for (int try_len = max_try; try_len >= 1; --try_len) {
                int tok = vocab_lookup(m_ptr + i, try_len);
                if (tok >= 0) {
                    best_len = try_len;
                    best_tok = tok;
                    break;
                }
            }

            if (best_tok >= 0) {
                merge_buf.push_back(best_tok);
                i += best_len;
            } else {
                // Should not happen with proper byte mappings, but fallback to byte token
                char byte_tok[8];
                snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)text[i]);
                int tok = vocab_lookup(byte_tok, (int)strlen(byte_tok));
                if (tok >= 0) {
                    merge_buf.push_back(tok);
                }
                i++;
            }
        }

        // Copy to output
        for (int tok : merge_buf) {
            tokens.push_back(tok);
        }

        return (int)tokens.size();
    }

    const char* decode_qwen(int token) const {
        if (token < 0 || token >= vocab_size) return "";
        const std::string& mangled_str = vocab[token];
        
        static thread_local std::string raw_output;
        raw_output.clear();
        
        if (!mappings_initialized) const_cast<Tokenizer*>(this)->init_byte_mappings();

        // Convert mangled string back to raw bytes
        for (size_t i = 0; i < mangled_str.size(); ) {
            uint32_t cp = 0;
            int len = utf8_to_codepoint(mangled_str.c_str() + i, (int)(mangled_str.size() - i), cp);
            if (len <= 0) {
                i++;
                continue;
            }
            
            // Handle <0xXX> escape sequences (should not appear in Qwen vocab, but just in case)
            if (cp == '<' && i + 5 < mangled_str.size() && 
                mangled_str[i+1] == '0' && mangled_str[i+2] == 'x') {
                unsigned int val = 0;
                if (sscanf(mangled_str.c_str() + i, "<0x%02X>", &val) == 1) {
                    raw_output += (char)val;
                    i += 6;
                    continue;
                }
            }
            
            raw_output += (char)unicode_to_byte(cp);
            i += len;
        }
        
        return raw_output.c_str();
    }
};

#endif // TOKENIZER_HPP
