#include "memtrace.h"
#include <cudaCompress/util/YCoCg.h>

#include <cudaCompress/cudaUtil.h>


// RGB <-> YCoCg
//
// [Y]  = [ 1/4  1/2  1/4] [R]
// [Co] = [ 1/2    0 -1/2] [G]
// [Cg] = [-1/4  1/2 -1/4] [B]
//
// [R]  = [   1    1   -1] [Y]
// [G]  = [   1    0    1] [Co]
// [B]  = [   1   -1   -1] [Cg]


namespace cudaCompress {

namespace util {


//TODO read coalesced uints into shared memory?

__global__ void convertRGBtoYCoCgKernel(uchar3* pTarget, const uchar3* pData, int pixelCount)
{
    for(uint index = blockIdx.x * blockDim.x + threadIdx.x; index < pixelCount; index += gridDim.x * blockDim.x) {
        // This is the non-reversible version: it preserves the dynamic range, but loses 1 bit in the chroma channels
        uchar3 rgb = pData[index];
        uchar3 ycocg;
        ycocg.x = ( rgb.x   + rgb.y*2 + rgb.z + 2) / 4;
        ycocg.y = ( rgb.x             - rgb.z + 1) / 2 + 127;
        ycocg.z = (-rgb.x   + rgb.y*2 - rgb.z + 2) / 4 + 127;
        pTarget[index] = ycocg;
    }
}

__global__ void convertYCoCgtoRGBKernel(uchar3* pTarget, const uchar3* pData, int pixelCount)
{
    for(uint index = blockIdx.x * blockDim.x + threadIdx.x; index < pixelCount; index += gridDim.x * blockDim.x) {
        uchar3 ycocg = pData[index];
        uchar3 rgb;
        rgb.x = min(max(ycocg.x + ycocg.y - ycocg.z,       0), 255);
        rgb.y = min(max(ycocg.x           + ycocg.z - 127, 0), 255);
        rgb.z = min(max(ycocg.x - ycocg.y - ycocg.z + 254, 0), 255);
        pTarget[index] = rgb;
    }
}


__global__ void convertRGBtoYCoCgKernel(uchar4* pTarget, const uchar4* pData, int pixelCount)
{
    for(uint index = blockIdx.x * blockDim.x + threadIdx.x; index < pixelCount; index += gridDim.x * blockDim.x) {
        // This is the non-reversible version: it preserves the dynamic range, but loses 1 bit in the chroma channels
        uchar4 rgb = pData[index];
        uchar4 ycocg;
        ycocg.x = ( rgb.x   + rgb.y*2 + rgb.z + 2) / 4;
        ycocg.y = ( rgb.x             - rgb.z + 1) / 2 + 127;
        ycocg.z = (-rgb.x   + rgb.y*2 - rgb.z + 2) / 4 + 127;
        ycocg.w = rgb.w;
        pTarget[index] = ycocg;
    }
}

__global__ void convertYCoCgtoRGBKernel(uchar4* pTarget, const uchar4* pData, int pixelCount)
{
    for(uint index = blockIdx.x * blockDim.x + threadIdx.x; index < pixelCount; index += gridDim.x * blockDim.x) {
        uchar4 ycocg = pData[index];
        uchar4 rgb;
        rgb.x = min(max(ycocg.x + ycocg.y - ycocg.z,       0), 255);
        rgb.y = min(max(ycocg.x           + ycocg.z - 127, 0), 255);
        rgb.z = min(max(ycocg.x - ycocg.y - ycocg.z + 254, 0), 255);
        rgb.w = ycocg.w;
        pTarget[index] = rgb;
    }
}


void convertRGBtoYCoCg(uchar3* dpTarget, const uchar3* dpData, int pixelCount)
{
    uint blockSize = 512;
    uint blockCount = min((pixelCount + blockSize - 1) / blockSize, 512);

    convertRGBtoYCoCgKernel<<<blockCount, blockSize>>>(dpTarget, dpData, pixelCount);
    cudaCheckMsg("convertRGBtoYCoCgKernel execution failed");
}

void convertYCoCgtoRGB(uchar3* dpTarget, const uchar3* dpData, int pixelCount)
{
    uint blockSize = 512;
    uint blockCount = min((pixelCount + blockSize - 1) / blockSize, 512);

    convertYCoCgtoRGBKernel<<<blockCount, blockSize>>>(dpTarget, dpData, pixelCount);
    cudaCheckMsg("convertYCoCgtoRGBKernel execution failed");
}


void convertRGBtoYCoCg(uchar4* dpTarget, const uchar4* dpData, int pixelCount)
{
    uint blockSize = 512;
    uint blockCount = min((pixelCount + blockSize - 1) / blockSize, 512);

    convertRGBtoYCoCgKernel<<<blockCount, blockSize>>>(dpTarget, dpData, pixelCount);
    cudaCheckMsg("convertRGBtoYCoCgKernel execution failed");
}

void convertYCoCgtoRGB(uchar4* dpTarget, const uchar4* dpData, int pixelCount)
{
    uint blockSize = 512;
    uint blockCount = min((pixelCount + blockSize - 1) / blockSize, 512);

    convertYCoCgtoRGBKernel<<<blockCount, blockSize>>>(dpTarget, dpData, pixelCount);
    cudaCheckMsg("convertYCoCgtoRGBKernel execution failed");
}


}

}
