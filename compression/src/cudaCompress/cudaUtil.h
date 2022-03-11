#ifndef __TUM3D_CUDACOMPRESS__CUDA_UTIL_H__
#define __TUM3D_CUDACOMPRESS__CUDA_UTIL_H__


#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>


constexpr auto LOG2_WARP_SIZE = 5U;
constexpr auto WARP_SIZE = (1U << LOG2_WARP_SIZE);


#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckMsg(msg) __cudaCheckMsg(msg, __FILE__, __LINE__)


#ifdef _DEBUG
#define CHECK_ERROR(err) (cudaSuccess != err || cudaSuccess != (err = cudaDeviceSynchronize()))
#else
#define CHECK_ERROR(err) (cudaSuccess != err)
#endif

inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
    if(CHECK_ERROR(err)) {
        fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n", file, line, cudaGetErrorString(err));
#ifdef _DEBUG
        __debugbreak();
#endif
    }
}

inline void __cudaCheckMsg(const char* errorMessage, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if(CHECK_ERROR(err)) {
        fprintf(stderr, "%s(%i) : cudaCheckMsg() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString(err));
#ifdef _DEBUG
        __debugbreak();
#endif
    }
}

#undef CHECK_ERROR


// Utility template struct for generating small vector types from scalar types
template<typename T, int N>
struct typeToVector
{
    typedef T Result;
};

template<>
struct typeToVector<short, 2>
{
    typedef short2 Result;
};
template<>
struct typeToVector<unsigned short, 2>
{
    typedef ushort2 Result;
};
template<>
struct typeToVector<int, 2>
{
    typedef int2 Result;
};
template<>
struct typeToVector<unsigned int, 2>
{
    typedef uint2 Result;
};
template<>
struct typeToVector<float, 2>
{
    typedef float2 Result;
};
template<>
struct typeToVector<short, 3>
{
    typedef short3 Result;
};
template<>
struct typeToVector<unsigned short, 3>
{
    typedef ushort3 Result;
};
template<>
struct typeToVector<int, 3>
{
    typedef int3 Result;
};
template<>
struct typeToVector<unsigned int, 3>
{
    typedef uint3 Result;
};
template<>
struct typeToVector<float, 3>
{
    typedef float3 Result;
};
template<>
struct typeToVector<short, 4>
{
    typedef short4 Result;
};
template<>
struct typeToVector<unsigned short, 4>
{
    typedef ushort4 Result;
};
template<>
struct typeToVector<int, 4>
{
    typedef int4 Result;
};
template<>
struct typeToVector<unsigned int, 4>
{
    typedef uint4 Result;
};
template<>
struct typeToVector<float, 4>
{
    typedef float4 Result;
};


// Helper class for dynamic shared memory declaration with templatized types.
// (Straightforward declaration results in linker errors (multiple definition)).
template<typename T>
struct SharedMemory
{
    __device__ inline T* getPointer() 
    {
        extern __device__ void Error_UnsupportedType(); // Ensure that we won't compile any un-specialized types
        Error_UnsupportedType();
        return (T*)0;
    }
};

template<>
struct SharedMemory <char>
{
    __device__ inline char* getPointer() { extern __shared__ char s_char[]; return s_char; }    
};
template<>
struct SharedMemory <unsigned char>
{
    __device__ inline unsigned char* getPointer() { extern __shared__ unsigned char s_uchar[]; return s_uchar; }    
};
template<>
struct SharedMemory <short>
{
    __device__ inline short* getPointer() { extern __shared__ short s_short[]; return s_short; }    
};
template<>
struct SharedMemory <unsigned short>
{
    __device__ inline unsigned short* getPointer() { extern __shared__ unsigned short s_ushort[]; return s_ushort; }    
};
template<>
struct SharedMemory <int>
{
    __device__ inline int* getPointer() { extern __shared__ int s_int[]; return s_int; }      
};
template<>
struct SharedMemory <unsigned int>
{
    __device__ inline unsigned int* getPointer() { extern __shared__ unsigned int s_uint[]; return s_uint; }    
};
template<>
struct SharedMemory <float>
{
    __device__ inline float* getPointer() { extern __shared__ float s_float[]; return s_float; }    
};


#endif
