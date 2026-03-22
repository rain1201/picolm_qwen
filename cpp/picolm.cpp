#include "model.hpp"
#include "tokenizer.hpp"
#include "sampler.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <clocale>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart * 1000.0;
}
#else
#include <sys/time.h>
static double get_time_ms() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
}
#endif

void usage(const char* prog) {
    std::cerr << "PicoLM C++ — ultra-lightweight LLM inference engine\n\n";
    std::cerr << "Usage: " << prog << " <model.gguf> [options]\n";
    std::cerr << "\nGeneration options:\n";
    std::cerr << "  -p <prompt>    Input prompt (or pipe via stdin)\n";
    std::cerr << "  -n <int>       Max tokens to generate (default: 256)\n";
    std::cerr << "  -t <float>     Temperature (default: 0.8, 0=greedy)\n";
    std::cerr << "  -k <float>     Top-p / nucleus sampling (default: 0.9)\n";
    std::cerr << "  -s <int>       RNG seed (default: 42)\n";
    std::cerr << "  -c <int>       Context length override\n";
    std::cerr << "  -j <int>       Number of threads (default: 4)\n";
    std::cerr << "\nOptimization options:\n";
    std::cerr << "  --prefetch     Enable layer prefetching (may improve performance on some systems)\n";
    std::cerr << "\nNote: UTF-8/Chinese input is fully supported.\n";
    std::cerr << "      On Windows, use: chcp 65001 before running for best compatibility.\n";
}

int main(int argc, char** argv) {
    // UTF-8 支持：设置 C 运行时本地化为 UTF-8
    std::setlocale(LC_ALL, ".UTF-8");

#ifdef _WIN32
    // Windows: 强制控制台代码页为 UTF-8
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);
    // powershell -Command "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; 'Hello' | picolm.exe model.gguf -n 50"
    // 启用虚拟终端处理
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD dwMode = 0;
    if (GetConsoleMode(hOut, &dwMode)) {
        SetConsoleMode(hOut, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING);
    }
#endif

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char* model_path = argv[1];
    const char* prompt = nullptr;
    int max_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.9f;
    uint64_t seed = 42;
    int context_override = 0;
    int num_threads = 4;
    bool enable_prefetch = false;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            context_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-j") == 0 && i + 1 < argc) {
            num_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--prefetch") == 0) {
            enable_prefetch = true;
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (!prompt || !*prompt) {
        // 尝试从 stdin 读取 prompt
        static std::string stdin_prompt;
        char buffer[256];
        while (fgets(buffer, sizeof(buffer), stdin)) {
            stdin_prompt += buffer;
        }
        if (!stdin_prompt.empty()) {
            // 移除 UTF-8 BOM (如果存在)
            size_t start = 0;
            if (stdin_prompt.size() >= 3 && 
                (unsigned char)stdin_prompt[0] == 0xEF && 
                (unsigned char)stdin_prompt[1] == 0xBB && 
                (unsigned char)stdin_prompt[2] == 0xBF) {
                start = 3;
            }
            stdin_prompt = stdin_prompt.substr(start);
            
            // 移除末尾空白字符
            while (!stdin_prompt.empty() && (stdin_prompt.back() == '\n' || stdin_prompt.back() == '\r' || stdin_prompt.back() == ' ')) {
                stdin_prompt.pop_back();
            }
            prompt = stdin_prompt.c_str();
        }
    }

    if (!prompt || !*prompt) {
        std::cerr << "No prompt provided. Use -p or pipe via stdin.\n";
        usage(argv[0]);
        return 1;
    }

    // Load model
    std::cerr << "Loading model: " << model_path << "\n";
    Model model;
    if (model.load(model_path, context_override, enable_prefetch) != 0) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    TensorOps::init_thread_pool(num_threads);

    // Load tokenizer
    Tokenizer tokenizer;
    if (tokenizer.load(model.tok_tokens_data, model.tok_n_tokens,
                       model.tok_scores_data, model.tok_n_scores,
                       model.tok_bos_id, model.tok_eos_id,
                       model.config.vocab_size) != 0) {
        std::cerr << "Failed to load tokenizer\n";
        return 1;
    }

    // Select RoPE function based on model type
    RopeFn rope_fn = nullptr;
    if (strstr(model_path, "wen") != nullptr) {
        rope_fn = TensorOps::rope_qwen;
    } else {
        rope_fn = TensorOps::rope_llama;
    }
    model.rope_fn = rope_fn;

    // Init sampler
    Sampler sampler;
    sampler.init(temperature, top_p, seed);

    // Encode prompt
    std::vector<int> prompt_tokens;
    tokenizer.encode_qwen(prompt, prompt_tokens, true);
    int n_prompt = (int)prompt_tokens.size();

    std::cerr << "Prompt: " << n_prompt << " tokens, generating up to " << max_tokens
              << " (temp=" << temperature << ", top_p=" << top_p << ", threads=" << num_threads
              << ", prefetch=" << (enable_prefetch ? "on" : "off") << ")\n";
    std::cerr << "---\n";

    // Generation loop
    int total_gen = 0;
    double t_start = get_time_ms();
    double t_first_token = 0;

    int token = prompt_tokens[0];
    int pos = 0;
    int total_steps = std::min(n_prompt + max_tokens, model.config.max_seq_len);

    for (; pos < total_steps; pos++) {
        float* logits = model.forward(token, pos);

        int next;
        if (pos < n_prompt - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            if (pos == n_prompt - 1) {
                t_first_token = get_time_ms();
            }

            next = sampler.sample(logits, model.config.vocab_size);

            const char* piece = tokenizer.decode_qwen(next);
            // 使用 fwrite 直接输出字节，避免 UTF-8 字节碎片问题
            fwrite(piece, 1, strlen(piece), stdout);
            fflush(stdout);

            total_gen++;

            if (next == (int)tokenizer.eos_id) break;
        }

        token = next;
    }

    printf("\n");
    double t_end = get_time_ms();

    // Stats
    double total_time = (t_end - t_start) / 1000.0;
    double gen_time = (t_end - t_first_token) / 1000.0;
    double prefill_time = (t_first_token - t_start) / 1000.0;

    std::cerr << "---\n";
    std::cerr << "Prefill: " << n_prompt << " tokens in " << prefill_time
              << "s (" << (prefill_time > 0 ? n_prompt / prefill_time : 0) << " tok/s)\n";
    std::cerr << "Generation: " << total_gen << " tokens in " << gen_time
              << "s (" << (gen_time > 0 ? total_gen / gen_time : 0) << " tok/s)\n";
    std::cerr << "Total: " << total_time << "s\n";
    std::cerr << "Memory: " << (model.state.mem_size / (1024.0 * 1024.0)) << " MB runtime state\n";

    TensorOps::cleanup_thread_pool();

    return 0;
}
