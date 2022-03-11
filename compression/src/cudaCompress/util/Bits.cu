#include "memtrace.h"
#include <cudaCompress/util/Bits.h>

#include <cudaCompress/cudaUtil.h>


namespace cudaCompress {

namespace util {


#define ELEMS_PER_THREAD 32

// negative -> -1   non-negative -> 0
static __device__ inline int getNegativeSign(short val)
{
    return (val >> 15);
}

template<int blockSize>
__global__ void getLSBkernel(const short* __restrict__ pData, uint* __restrict__ pBits, uint elemCount)
{
    static_assert(blockSize % ELEMS_PER_THREAD == 0, "blockSize must be a multiple of ELEMS_PER_THREAD");

    __shared__ short s_Data[ELEMS_PER_THREAD][blockSize + 1];

    pData += blockIdx.x * blockDim.x * ELEMS_PER_THREAD + threadIdx.x;

    uint index0 = threadIdx.x % ELEMS_PER_THREAD;
    uint index1 = threadIdx.x / ELEMS_PER_THREAD;

    // LOAD
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        s_Data[index0][index1 + i * blockSize / ELEMS_PER_THREAD] = pData[i * blockSize];
    }

    __syncthreads();

    // COMPUTE
    uint bits = 0;
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        short val = s_Data[i][threadIdx.x];
        bits |= (val & 1) << i;
    }

    __syncthreads();

    // STORE
    pBits[blockIdx.x * blockDim.x + threadIdx.x] = bits;
}

template<int blockSize>
__global__ void removeLSBKernel(short* __restrict__ pData, uint elemCount, short shift)
{
    static_assert(blockSize % ELEMS_PER_THREAD == 0, "blockSize must be a multiple of ELEMS_PER_THREAD");

    __shared__ short s_Data[ELEMS_PER_THREAD][blockSize + 1];

    pData += blockIdx.x * blockDim.x * ELEMS_PER_THREAD + threadIdx.x;

    uint index0 = threadIdx.x % ELEMS_PER_THREAD;
    uint index1 = threadIdx.x / ELEMS_PER_THREAD;

    // LOAD
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        s_Data[index0][index1 + i * blockSize / ELEMS_PER_THREAD] = pData[i * blockSize];
    }

    __syncthreads();

    // COMPUTE
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        short val = s_Data[i][threadIdx.x] + shift;
        s_Data[i][threadIdx.x] = (val + getNegativeSign(val)) / 2;
    }

    __syncthreads();

    // STORE
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        pData[i * blockSize] = s_Data[index0][index1 + i * blockSize / ELEMS_PER_THREAD];
    }
}

template<int blockSize>
__global__ void getAndRemoveLSBKernel(short* __restrict__ pData, uint* __restrict__ pBits, uint elemCount, short shift)
{
    static_assert(blockSize % ELEMS_PER_THREAD == 0, "blockSize must be a multiple of ELEMS_PER_THREAD");

    __shared__ short s_Data[ELEMS_PER_THREAD][blockSize + 1];

    pData += blockIdx.x * blockDim.x * ELEMS_PER_THREAD + threadIdx.x;

    uint index0 = threadIdx.x % ELEMS_PER_THREAD;
    uint index1 = threadIdx.x / ELEMS_PER_THREAD;

    // LOAD
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        s_Data[index0][index1 + i * blockSize / ELEMS_PER_THREAD] = pData[i * blockSize];
    }

    __syncthreads();

    // COMPUTE
    uint bits = 0;
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        short val = s_Data[i][threadIdx.x] + shift;
        bits |= (val & 1) << i;
        s_Data[i][threadIdx.x] = (val + getNegativeSign(val)) / 2;
    }

    __syncthreads();

    // STORE
    pBits[blockIdx.x * blockDim.x + threadIdx.x] = bits;
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        pData[i * blockSize] = s_Data[index0][index1 + i * blockSize / ELEMS_PER_THREAD];
    }
}

template<int blockSize>
__global__ void appendLSBKernel(short* __restrict__ pData, const uint* __restrict__ pBits, uint elemCount, short shift)
{
    static_assert(blockSize % ELEMS_PER_THREAD == 0, "blockSize must be a multiple of ELEMS_PER_THREAD");

    __shared__ short s_Data[ELEMS_PER_THREAD][blockSize + 1];

    pData += blockIdx.x * blockDim.x * ELEMS_PER_THREAD + threadIdx.x;

    uint index0 = threadIdx.x % ELEMS_PER_THREAD;
    uint index1 = threadIdx.x / ELEMS_PER_THREAD;

    // LOAD
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        s_Data[index0][index1 + i * blockSize / ELEMS_PER_THREAD] = pData[i * blockSize];
    }
    uint bits = pBits[blockIdx.x * blockDim.x + threadIdx.x];

    __syncthreads();

    // COMPUTE
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        short val = s_Data[i][threadIdx.x];
        short bit = (bits >> i) & 1;
        val = (val * 2) | bit;
        s_Data[i][threadIdx.x] = val + shift;
    }

    __syncthreads();

    // STORE
    #pragma unroll
    for(uint i = 0; i < ELEMS_PER_THREAD; i++) {
        pData[i * blockSize] = s_Data[index0][index1 + i * blockSize / ELEMS_PER_THREAD];
    }
}

__global__ void shiftKernel(short* __restrict__ pData, uint elemCount, short shift)
{
    uint index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < elemCount) {
        pData[index] += shift;
    }
}


bool getLSB(const short* dpData, uint* dpBits, uint elemCount)
{
    const uint blockSize = 128;
    uint blockCount = (elemCount + blockSize * ELEMS_PER_THREAD - 1) / (blockSize * ELEMS_PER_THREAD);

    getLSBkernel<blockSize><<<blockCount, blockSize>>>(dpData, dpBits, elemCount);
    cudaCheckMsg("getLSBkernel execution failed");

    return true;
}

bool removeLSB(short* dpData, uint* dpBits, uint elemCount, short shift)
{
    const uint blockSize = 128;
    uint blockCount = (elemCount + blockSize * ELEMS_PER_THREAD - 1) / (blockSize * ELEMS_PER_THREAD);

    if(dpBits != 0) {
        getAndRemoveLSBKernel<blockSize><<<blockCount, blockSize>>>(dpData, dpBits, elemCount, shift);
        cudaCheckMsg("getAndRemoveLSBKernel execution failed");
    } else {
        removeLSBKernel<blockSize><<<blockCount, blockSize>>>(dpData, elemCount, shift);
        cudaCheckMsg("removeLSBKernel execution failed");
    }

    return true;
}

bool appendLSB(short* dpData, const uint* dpBits, uint elemCount, short shift)
{
    const uint blockSize = 128;
    uint blockCount = (elemCount + blockSize * ELEMS_PER_THREAD - 1) / (blockSize * ELEMS_PER_THREAD);

    appendLSBKernel<blockSize><<<blockCount, blockSize>>>(dpData, dpBits, elemCount, shift);
    cudaCheckMsg("appendLSBKernel execution failed");

    return true;
}

bool shiftElems(short* dpData, uint elemCount, short shift)
{
    const uint blockSize = 192;
    uint blockCount = (elemCount + blockSize - 1) / blockSize;

    shiftKernel<<<blockCount, blockSize>>>(dpData, elemCount, shift);
    cudaCheckMsg("shiftKernel execution failed");

    return true;
}


}

}
