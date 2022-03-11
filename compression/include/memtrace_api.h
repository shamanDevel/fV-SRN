#pragma once
#include <cstddef>


namespace compression
{
    namespace memory
    {
        enum class Device
        {
            CPU,
            GPU
        };
        ptrdiff_t totalMemAllocated(Device device);
        ptrdiff_t totalMemFreed(Device device);
        ptrdiff_t currentMemAllocated(Device device);
        ptrdiff_t peakMemAllocated(Device device);
        void resetPeakMemory(Device device);
    }
}
