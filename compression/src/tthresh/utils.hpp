#pragma once

#include "memtrace.h"
#include <vector>

namespace tthresh
{
    inline void cumulative_products(const std::vector<uint32_t>& in, std::vector<size_t>& out) {
        uint8_t n = in.size();
        out = std::vector<size_t>(n + 1); // Cumulative size products. The i-th element contains s[0]*...*s[i-1]
        out[0] = 1;
        for (uint8_t i = 0; i < n; ++i)
            out[i + 1] = out[i] * in[i];
    }


    inline int64_t min(int64_t a, int64_t b) {
        return (a < b) ? a : b;
    }

    inline int64_t max(int64_t a, int64_t b) {
        return (a > b) ? a : b;
    }
}