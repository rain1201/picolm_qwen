#ifndef GGUF_METADATA_HPP
#define GGUF_METADATA_HPP

#include <string>
#include <unordered_map>
#include <cstdint>
#include <any>
#include <variant>
#include <optional>
#include <vector>

// ============================================================================
// GGUF 元数据键名映射表（替代硬编码常量）
// ============================================================================

namespace gguf {

// 模型配置键名映射
inline const std::unordered_map<std::string, std::string> CONFIG_KEYS = {
    {"embedding_length", "n_embd"},
    {"feed_forward_length", "n_ffn"},
    {"attention.head_count", "n_heads"},
    {"attention.head_count_kv", "n_kv_heads"},
    {"block_count", "n_layers"},
    {"context_length", "max_seq_len"},
    {"attention.key_length", "head_dim"},
    {"rope.freq_base", "rope_freq_base"},
    {"general.alignment", "alignment"},
    {"tokenizer.ggml.bos_token_id", "bos_token_id"},
    {"tokenizer.ggml.eos_token_id", "eos_token_id"},
    {"tokenizer.ggml.tokens", "tokens"},
    {"tokenizer.ggml.scores", "scores"},
    {"tokenizer.ggml.token_type", "token_type"},
    {"tokenizer.ggml.pre", "tokenizer_pre"},
    {"general.architecture", "architecture"},
    {"rope.dimension_count", "rope_dim"},
};

// 层权重键名映射
inline const std::unordered_map<std::string, std::string> LAYER_KEYS = {
    {"attn_norm", "attn_norm"},
    {"ffn_norm", "ffn_norm"},
    {"attn_q", "attn_q"},
    {"attn_k", "attn_k"},
    {"attn_v", "attn_v"},
    {"attn_output", "attn_output"},
    {"attn_q.bias", "attn_q_b"},
    {"attn_k.bias", "attn_k_b"},
    {"attn_v.bias", "attn_v_b"},
    {"attn_output.bias", "attn_output_b"},
    {"attn_q_norm", "attn_q_norm"},
    {"attn_k_norm", "attn_k_norm"},
    {"ffn_gate", "ffn_gate"},
    {"ffn_down", "ffn_down"},
    {"ffn_up", "ffn_up"},
    {"ffn_gate.bias", "ffn_gate_b"},
    {"ffn_down.bias", "ffn_down_b"},
    {"ffn_up.bias", "ffn_up_b"},
};

// 输出层权重键名映射
inline const std::unordered_map<std::string, std::string> OUTPUT_KEYS = {
    {"token_embd", "token_embd"},
    {"output_norm", "output_norm"},
    {"output", "output"},
};

} // namespace gguf

// ============================================================================
// 元数据类型定义
// ============================================================================

namespace metadata {

// 权重条目：指针 + 量化类型
struct WeightEntry {
    const void* ptr = nullptr;
    uint32_t type = 0;
    
    WeightEntry() = default;
    WeightEntry(const void* p, uint32_t t) : ptr(p), type(t) {}
    
    explicit operator bool() const { return ptr != nullptr; }
};

// 层权重映射表
using LayerWeightMap = std::unordered_map<std::string, WeightEntry>;

// 模型权重映射表
using ModelWeightMap = std::unordered_map<std::string, WeightEntry>;

// 层权重集合
struct LayerWeights {
    LayerWeightMap weights;
    
    WeightEntry& operator[](const std::string& key) {
        return weights[key];
    }
    
    const WeightEntry& operator[](const std::string& key) const {
        static WeightEntry empty;
        auto it = weights.find(key);
        return (it != weights.end()) ? it->second : empty;
    }
    
    WeightEntry get(const std::string& key) const {
        auto it = weights.find(key);
        return (it != weights.end()) ? it->second : WeightEntry{};
    }
    
    void set(const std::string& key, const void* ptr, uint32_t type) {
        if (ptr) {
            weights[key] = WeightEntry{ptr, type};
        }
    }
    
    bool has(const std::string& key) const {
        return weights.find(key) != weights.end();
    }
};

// 支持的元数据类型
using MetaValue = std::variant<
    int32_t,
    uint32_t,
    int64_t,
    uint64_t,
    float,
    double,
    bool,
    std::string,
    std::vector<int32_t>,
    std::vector<float>,
    std::vector<std::string>
>;

// 元数据映射表
using MetaMap = std::unordered_map<std::string, MetaValue>;

// 张量信息结构
struct TensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dims[4] = {0, 0, 0, 0};
    uint32_t type;
    uint64_t offset;
    const void* data_ptr = nullptr;
};

// 模型配置结构（完全基于 map，直接读写）
struct ModelConfig {
    // 统一的配置存储
    std::unordered_map<std::string, int32_t> ints;
    std::unordered_map<std::string, float> floats;
    std::unordered_map<std::string, uint32_t> uints;
    metadata::MetaMap metadata;
    
    // 直接访问方法
    int32_t getInt(const std::string& key, int32_t def = 0) const {
        auto it = ints.find(key);
        return (it != ints.end()) ? it->second : def;
    }
    
    float getFloat(const std::string& key, float def = 0.0f) const {
        auto it = floats.find(key);
        return (it != floats.end()) ? it->second : def;
    }
    
    uint32_t getUint(const std::string& key, uint32_t def = 0) const {
        auto it = uints.find(key);
        return (it != uints.end()) ? it->second : def;
    }
    
    void setInt(const std::string& key, int32_t val) { ints[key] = val; }
    void setFloat(const std::string& key, float val) { floats[key] = val; }
    void setUint(const std::string& key, uint32_t val) { uints[key] = val; }
};

} // namespace metadata

// ============================================================================
// 名称解析工具类
// ============================================================================

namespace parser {

// 解析层权重名称，返回 {layer_id, suffix}
inline std::pair<int, std::string> parse_layer_name(const std::string& full_name) {
    if (full_name.size() < 6 || full_name.substr(0, 4) != "blk.") {
        return {-1, ""};
    }
    
    size_t dot_after_layer = full_name.find('.', 4);
    if (dot_after_layer == std::string::npos) {
        return {-1, ""};
    }
    
    try {
        int layer_id = std::stoi(full_name.substr(4, dot_after_layer - 4));
        std::string suffix = full_name.substr(dot_after_layer + 1);
        return {layer_id, suffix};
    } catch (...) {
        return {-1, ""};
    }
}

// 判断是否为层权重名称
inline bool is_layer_weight(const std::string& name) {
    return name.size() >= 6 && name.substr(0, 4) == "blk.";
}

// 判断是否为输出层权重
inline bool is_output_weight(const std::string& name) {
    std::string base = name;
    const std::string suffix = ".weight";
    if (name.size() >= suffix.size() && 
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        base = name.substr(0, name.size() - suffix.size());
    }
    return base == "token_embd" || base == "output_norm" || base == "output";
}

// 获取输出权重类型
inline std::string get_output_type(const std::string& name) {
    std::string base = name;
    const std::string suffix = ".weight";
    if (name.size() >= suffix.size() && 
        name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
        base = name.substr(0, name.size() - suffix.size());
    }
    return base;
}

// 元数据键名匹配
inline bool match_meta_key(const std::string& key, const std::string& target) {
    return key.find(target) != std::string::npos;
}

// 从 GGUF 元数据中解析配置值
inline void parse_config_value(const std::string& key, int32_t value, metadata::ModelConfig& config) {
    // 直接存储到对应的 map
    if (key.find("embedding_length") != std::string::npos) {
        config.ints["n_embd"] = value;
    } else if (key.find("feed_forward_length") != std::string::npos) {
        config.ints["n_ffn"] = value;
    } else if (key.find("attention.head_count") != std::string::npos &&
               key.find("_kv") == std::string::npos) {
        config.ints["n_heads"] = value;
    } else if (key.find("attention.head_count_kv") != std::string::npos) {
        config.ints["n_kv_heads"] = value;
    } else if (key.find("block_count") != std::string::npos) {
        config.ints["n_layers"] = value;
    } else if (key.find("context_length") != std::string::npos) {
        config.ints["max_seq_len"] = value;
    } else if (key.find("general.alignment") != std::string::npos) {
        config.ints["alignment"] = value;
    } else if (key.find("attention.key_length") != std::string::npos) {
        config.ints["head_dim"] = value;
    } else if (key.find("tokenizer.ggml.bos_token_id") != std::string::npos) {
        config.uints["tok_bos_id"] = static_cast<uint32_t>(value);
    } else if (key.find("tokenizer.ggml.eos_token_id") != std::string::npos) {
        config.uints["tok_eos_id"] = static_cast<uint32_t>(value);
    }

    // 存储到通用 metadata
    config.metadata[key] = value;
}

inline void parse_config_value(const std::string& key, float value, metadata::ModelConfig& config) {
    if (key.find("rope.freq_base") != std::string::npos) {
        config.floats["rope_freq_base"] = value;
    }
    config.metadata[key] = value;
}

// 辅助函数：从 map 获取值（带默认值）
inline int getInt(const std::unordered_map<std::string, int32_t>& m, const char* key, int def = 0) {
    auto it = m.find(key);
    return (it != m.end()) ? (int)it->second : def;
}
inline float getFloat(const std::unordered_map<std::string, float>& m, const char* key, float def = 0.0f) {
    auto it = m.find(key);
    return (it != m.end()) ? it->second : def;
}
inline uint32_t getUint(const std::unordered_map<std::string, uint32_t>& m, const char* key, uint32_t def = 0) {
    auto it = m.find(key);
    return (it != m.end()) ? it->second : def;
}
inline bool getBool(const std::unordered_map<std::string, int32_t>& m, const char* key) {
    auto it = m.find(key);
    return (it != m.end()) ? (it->second != 0) : false;
}

} // namespace parser

#endif // GGUF_METADATA_HPP
