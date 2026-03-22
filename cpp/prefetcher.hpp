#ifndef PREFETCHER_HPP
#define PREFETCHER_HPP

#include "quant.hpp"
#include <cstdint>
#include <cstddef>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#endif

// 轻量级内存预取器
// 使用操作系统 API 提示内核即将访问的内存区域
struct MemoryPrefetcher {
    // 预取下一层的权重数据
    // 在计算当前层时调用，提示内核加载下一层的数据到缓存
    static inline void prefetch_layer(const void* ptr, size_t size) {
        if (!ptr || size == 0) return;
        
#ifdef _WIN32
        // Windows: 使用 PrefetchVirtualMemory API (Windows 8+)
        WIN32_MEMORY_RANGE_ENTRY entry;
        entry.VirtualAddress = (PVOID)ptr;
        entry.NumberOfBytes = (SIZE_T)size;
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#else
        // Linux/Unix: 使用 madvise
        madvise((void*)ptr, size, MADV_WILLNEED);
#endif
    }
};

#endif // PREFETCHER_HPP
