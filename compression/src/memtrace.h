#pragma once

/**
 * Tools to log memory usage on GPU and CPU
 */

#include <cstdlib>
#include <cuda_runtime.h>
#include <new>

namespace compression
{
    namespace detail {
        void* my_malloc(size_t _Size);
        void my_free(void* ptr);

        cudaError_t my_cudaMalloc(void** ptr, size_t size);
        cudaError_t my_cudaMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);
        cudaError_t my_cudaMallocHost(void** ptr, size_t size, unsigned int flags = 0);
        cudaError_t my_cudaFree(void* ptr);
        cudaError_t my_cudaFreeHost(void* ptr);

        template<class T>
        static __inline__ __host__ cudaError_t my_cudaMalloc(
            T** devPtr,
            size_t   size
        )
        {
            return ::compression::detail::my_cudaMalloc((void**)(void*)devPtr, size);
        }

        template<class T>
        static __inline__ __host__ cudaError_t my_cudaMallocHost(
            T** ptr,
            size_t         size,
            unsigned int   flags = 0
        )
        {
            return ::compression::detail::my_cudaMallocHost((void**)(void*)ptr, size, flags);
        }

        template<class T>
        static __inline__ __host__ cudaError_t my_cudaMallocPitch(
            T** devPtr,
            size_t* pitch,
            size_t   width,
            size_t   height
        )
        {
            return ::compression::detail::my_cudaMallocPitch((void**)(void*)devPtr, pitch, width, height);
        }
    }
}
//namespace std{namespace compression{namespace detail
//{
//    void* __cdecl my_malloc(size_t size);
//    void __cdecl my_free(void* ptr);
//}}}

#define malloc(...) compression::detail::my_malloc(__VA_ARGS__)
#define free(...) compression::detail::my_free(__VA_ARGS__)

#define cudaMalloc(...) ::compression::detail::my_cudaMalloc(__VA_ARGS__)
#define cudaMallocPitch(...) ::compression::detail::my_cudaMallocPitch(__VA_ARGS__)
#define cudaMallocHost(...) ::compression::detail::my_cudaMallocHost(__VA_ARGS__)
#define cudaFree(...) ::compression::detail::my_cudaFree(__VA_ARGS__)
#define cudaFreeHost(...) ::compression::detail::my_cudaFreeHost(__VA_ARGS__)
