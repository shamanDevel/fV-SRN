//No include of memtrace.h !!

#include <cuda_runtime.h>
#include <atomic>
#include <unordered_map>
#include <iostream>
#include <mutex>

namespace compression
{
    class MemoryStatistics
    {
        bool initialized_ = false;
        bool withinAlloc_ = false;
        bool withinFree_ = false;
        std::recursive_mutex mutex_;
        std::unordered_map<uintptr_t, size_t> allocations_;

    public:
        ptrdiff_t totalMemAllocated;
        ptrdiff_t totalMemFreed;
        ptrdiff_t currentMemAllocated;
        ptrdiff_t peakMemAllocated;

        MemoryStatistics()
            : totalMemAllocated(0), totalMemFreed(0)
            , currentMemAllocated(0), peakMemAllocated(0)
        {
            initialized_ = true;
        }
        ~MemoryStatistics()
        {
            initialized_ = false;
        }

        void alloc(void* ptr, size_t Size)
        {
            if (!initialized_) return; //catch static initializer for this instance
            std::lock_guard lock(mutex_);

            if (withinAlloc_) return; //catch infinite recursion
            withinAlloc_ = true;

            uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
            auto it = allocations_.find(p);
            if (it != allocations_.end())
            {
                //std::cerr << "Double registration of memory with pointer " << ptr << ", size " << Size << std::endl;
            }
            else
            {
                totalMemAllocated += Size;
                currentMemAllocated += Size;
                //weak update, not critical that this is atomic.
                peakMemAllocated = std::max(peakMemAllocated, currentMemAllocated);
                allocations_[p] = Size;
            }

            withinAlloc_ = false;
        }
        void free(void* ptr)
        {
            if (!initialized_) return; //catch static deconstructor for this instance
            std::lock_guard lock(mutex_);

            if (withinFree_) return; //catch infinite recursion
            withinFree_ = true;

            uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
            auto it = allocations_.find(p);
            if (it == allocations_.end()) {
                //std::cerr << "Freeing memory that was not tracked. Pointer:" << ptr << std::endl;
            }
            else
            {
                size_t size = it->second;
                allocations_.erase(it);
                totalMemFreed += size;
                currentMemAllocated -= size;
            }

            withinFree_ = false;
        }
        void resetPeak() {
            peakMemAllocated = currentMemAllocated;
        }
    };
    MemoryStatistics g_GPUStats;
    MemoryStatistics g_CPUStats;
}

namespace compression
{
    namespace detail {
        void* my_malloc(size_t _Size)
        {
            void* ptr = ::malloc(_Size);
            if (ptr) g_CPUStats.alloc(ptr, _Size);
            return ptr;
        }
        void my_free(void* ptr)
        {
            if (ptr) g_CPUStats.free(ptr);
            ::free(ptr);
        }

        cudaError_t my_cudaMalloc(void** ptr, size_t size)
        {
            auto err = ::cudaMalloc(ptr, size);
            if (err==cudaSuccess) g_GPUStats.alloc(*ptr, size);
            return err;
        }
        cudaError_t my_cudaMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height)
        {
            auto err = ::cudaMallocPitch(ptr, pitch, width, height);
            if (err == cudaSuccess) g_GPUStats.alloc(*ptr, width*height);
            return err;
        }
        cudaError_t my_cudaMallocHost(void** ptr, size_t size, unsigned int flags = 0)
        {
            auto err = ::cudaMallocHost(ptr, size, flags);
            if (err == cudaSuccess) g_CPUStats.alloc(*ptr, size);
            return err;
        }
        cudaError_t my_cudaFree(void* ptr)
        {
            if (ptr) g_GPUStats.free(ptr);
            return ::cudaFree(ptr);
        }
        cudaError_t my_cudaFreeHost(void* ptr)
        {
            if (ptr) g_CPUStats.free(ptr);
            return ::cudaFreeHost(ptr);
        }
    }
}
//namespace std {
//    namespace compression {
//        namespace detail
//        {
//            void* __cdecl my_malloc(size_t _Size) { return ::compression::detail::my_malloc(_Size); }
//            void __cdecl my_free(void* ptr) { ::compression::detail::my_free(ptr); }
//        }
//    }
//}



namespace compression
{
    namespace memory
    {
        enum class Device
        {
            CPU,
            GPU
        };
        ptrdiff_t totalMemAllocated(Device device)
        {
            return (device == Device::CPU) ? g_CPUStats.totalMemAllocated : g_GPUStats.totalMemAllocated;
        }
        ptrdiff_t totalMemFreed(Device device)
        {
            return (device == Device::CPU) ? g_CPUStats.totalMemFreed : g_GPUStats.totalMemFreed;
        }
        ptrdiff_t currentMemAllocated(Device device)
        {
            return (device == Device::CPU) ? g_CPUStats.currentMemAllocated : g_GPUStats.currentMemAllocated;
        }
        ptrdiff_t peakMemAllocated(Device device)
        {
            return (device == Device::CPU) ? g_CPUStats.peakMemAllocated : g_GPUStats.peakMemAllocated;
        }
        void resetPeakMemory(Device device)
        {
            if (device == Device::CPU)
                g_CPUStats.resetPeak();
            else
                g_GPUStats.resetPeak();
        }
    }
}


void* operator new (size_t size)
{
    if (size == 0) size = 1;
    void* p = ::compression::detail::my_malloc(size);
    if (p == 0) // did malloc succeed?
        throw std::bad_alloc(); // ANSI/ISO compliant behavior

    return p;
}
void operator delete(void* ptr) throw()
{
    ::compression::detail::my_free(ptr);
}
