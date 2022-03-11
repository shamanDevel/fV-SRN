#include "memtrace.h"
#include <cudaCompress/util/DWT.h>

#include <assert.h>

#include <cudaCompress/cudaUtil.h>

#include "DWTFloatKernels.cui"
#include "DWTFloatFromSymbolsKernels.cui"
#include "DWTFloat2DLowpassKernels.cui"


namespace cudaCompress {

namespace util {


template<typename TIn, int channelCountIn>
static void dwtFloatForwardLowpassOnly2D(
    float* dpDest, float* dpBuffer, const TIn* dpSource,
    int sizeX, int sizeY,
    int dstRowPitch, int srcRowPitch,
    cudaStream_t stream)
{
    const int xBlockSizeX = 32;
    const int xBlockSizeY = 4;
    const dim3 xBlockSize(xBlockSizeX, xBlockSizeY);
    const int xResultBlockCount = 8;

    const int yBlockSizeX = 32;
    const int yBlockSizeY = 4;
    const dim3 yBlockSize(yBlockSizeX, yBlockSizeY);
    const int yResultBlockCount = 8;


    dim3 xBlockCount(sizeX / (xResultBlockCount * xBlockSizeX), (sizeY + xBlockSizeY - 1) / xBlockSizeY);

    if(xBlockCount.x > 0) {
        forwardDWT9XLowpassKernel2D
            <TIn, channelCountIn, xBlockSizeX, xBlockSizeY, xResultBlockCount>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpBuffer, dpSource, sizeX, sizeY, dstRowPitch, srcRowPitch);
        cudaCheckMsg("forwardDWT9XLowpassKernel2D execution failed");
    }

    int sizeXdone = xBlockCount.x * xResultBlockCount * xBlockSizeX;
    int sizeXrest = sizeX - sizeXdone;
    if(sizeXrest > 0) {
        dim3 xBlockCountRest(1, xBlockCount.y);
        int xResultBlockCountRest = (sizeXrest + xBlockSizeX - 1) / xBlockSizeX;
        uint sharedSize = (xBlockSizeY * (xResultBlockCountRest * xBlockSizeX + (FILTER_LENGTH-1))) * sizeof(float);
        forwardDWT9XLowpassRestKernel2D
            <TIn, channelCountIn, xBlockSizeX, xBlockSizeY>
            <<<xBlockCountRest, xBlockSize, sharedSize, stream>>>
            (dpBuffer, dpSource, sizeXdone, sizeX, sizeY, xResultBlockCountRest, dstRowPitch, srcRowPitch);
        cudaCheckMsg("forwardDWT9XLowpassRestKernel2D execution failed");
    }


    dim3 yBlockCount((sizeX/2 + yBlockSizeX - 1) / yBlockSizeX, sizeY / (yResultBlockCount * yBlockSizeY));

    if(yBlockCount.y > 0) {
        forwardDWT9YLowpassKernel2D
            <yBlockSizeX, yBlockSizeY, yResultBlockCount>
            <<<yBlockCount, yBlockSize, 0, stream>>>
            (dpDest, dpBuffer, sizeX/2, sizeY, dstRowPitch);
        cudaCheckMsg("forwardDWT9YLowpassKernel2D execution failed");
    }

    int sizeYdone = yBlockCount.y * yResultBlockCount * yBlockSizeY;
    int sizeYrest = sizeY - sizeYdone;
    if(sizeYrest > 0) {
        dim3 yBlockCountRest(yBlockCount.x, 1);
        int yResultBlockCountRest = (sizeYrest + yBlockSizeY - 1) / yBlockSizeY;
        uint sharedSize = (yBlockSizeX * (yResultBlockCountRest * yBlockSizeY + (FILTER_LENGTH-1) + 1)) * sizeof(float);
        forwardDWT9YLowpassRestKernel2D
            <yBlockSizeX, yBlockSizeY>
            <<<yBlockCountRest, yBlockSize, sharedSize, stream>>>
            (dpDest, dpBuffer, sizeYdone, sizeX/2, sizeY, yResultBlockCountRest, dstRowPitch);
        cudaCheckMsg("forwardDWT9YLowpassRestKernel2D execution failed");
    }
}


template<typename TIn, int channelCountIn>
static void dwtFloatForward(
    float* dpDest, float* dpBuffer2, float* dpBuffer1, const TIn* dpSource,
    int sizeX, int sizeY, int sizeZ,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    const int xBlockSizeX = 32;
    const int xBlockSizeY = 4;
    const dim3 xBlockSize(xBlockSizeX, xBlockSizeY);
    const int xResultBlockCount  = 8;
    const int xResultBlockCount2 = 4;
    const int xResultBlockCount3 = 2;

    const int yBlockSizeX = 32;
    const int yBlockSizeY = 4;
    const dim3 yBlockSize(yBlockSizeX, yBlockSizeY);
    const int yResultBlockCount = 8;

    const int zBlockSizeX = 32;
    const int zBlockSizeY = 4;
    const dim3 zBlockSize(zBlockSizeX, zBlockSizeY);
    const int zResultBlockCount = 8;


    bool do3D = (sizeZ > 1);


    dim3 xBlockCount(sizeX / (xResultBlockCount * xBlockSizeX), (sizeY + xBlockSizeY - 1) / xBlockSizeY, sizeZ);

    if(xBlockCount.x > 0) {
        forwardDWT9XKernel
            <TIn, channelCountIn, xBlockSizeX, xBlockSizeY, xResultBlockCount>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpBuffer1, dpSource, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("forwardDWT9XKernel execution failed");
    } else if(sizeX == (xResultBlockCount2 * xBlockSizeX)) {
        // special case for sizeX == 128
        xBlockCount.x = 1;
        forwardDWT9XKernel
            <TIn, channelCountIn, xBlockSizeX, xBlockSizeY, xResultBlockCount2>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpBuffer1, dpSource, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("forwardDWT9XKernel execution failed");
    } else if(sizeX == (xResultBlockCount3 * xBlockSizeX)) {
        // special case for sizeX == 64
        xBlockCount.x = 1;
        forwardDWT9XKernel
            <TIn, channelCountIn, xBlockSizeX, xBlockSizeY, xResultBlockCount3>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpBuffer1, dpSource, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("forwardDWT9XKernel execution failed");
    }

    int sizeXdone = xBlockCount.x * xResultBlockCount * xBlockSizeX;
    int sizeXrest = sizeX - sizeXdone;
    if(sizeXrest > 0) {
        dim3 xBlockCountRest(1, xBlockCount.y, xBlockCount.z);
        int xResultBlockCountRest = (sizeXrest + xBlockSizeX - 1) / xBlockSizeX;
        uint sharedSize = (xBlockSizeY * (xResultBlockCountRest * xBlockSizeX + (FILTER_LENGTH-1))) * sizeof(float);
        forwardDWT9XRestKernel
            <TIn, channelCountIn, xBlockSizeX, xBlockSizeY>
            <<<xBlockCountRest, xBlockSize, sharedSize, stream>>>
            (dpBuffer1, dpSource, sizeXdone, sizeX, sizeY, xResultBlockCountRest, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("forwardDWT9XRestKernel execution failed");
    }


    float* dpDestY = (do3D ? dpBuffer2 : dpDest);

    dim3 yBlockCount((sizeX + yBlockSizeX - 1) / yBlockSizeX, sizeY / (yResultBlockCount * yBlockSizeY), sizeZ);

    if(yBlockCount.y > 0) {
        forwardDWT9YKernel
            <yBlockSizeX, yBlockSizeY, yResultBlockCount>
            <<<yBlockCount, yBlockSize, 0, stream>>>
            (dpDestY, dpBuffer1, sizeX, sizeY, dstRowPitch, dstSlicePitch);
        cudaCheckMsg("forwardDWT9YKernel execution failed");
    }

    int sizeYdone = yBlockCount.y * yResultBlockCount * yBlockSizeY;
    int sizeYrest = sizeY - sizeYdone;
    if(sizeYrest > 0) {
        dim3 yBlockCountRest(yBlockCount.x, 1, yBlockCount.z);
        int yResultBlockCountRest = (sizeYrest + yBlockSizeY - 1) / yBlockSizeY;
        uint sharedSize = (yBlockSizeX * (yResultBlockCountRest * yBlockSizeY + (FILTER_LENGTH-1) + 1)) * sizeof(float);
        forwardDWT9YRestKernel
            <yBlockSizeX, yBlockSizeY>
            <<<yBlockCountRest, yBlockSize, sharedSize, stream>>>
            (dpDestY, dpBuffer1, sizeYdone, sizeX, sizeY, yResultBlockCountRest, dstRowPitch, dstSlicePitch);
        cudaCheckMsg("forwardDWT9YRestKernel execution failed");
    }


    if(do3D) {
        dim3 zBlockCount((sizeX + zBlockSizeX - 1) / zBlockSizeX, sizeY, sizeZ / (zResultBlockCount * zBlockSizeY));
        if(zBlockCount.z > 0) {
            forwardDWT9ZKernel
                <zBlockSizeX, zBlockSizeY, zResultBlockCount>
                <<<zBlockCount, zBlockSize, 0, stream>>>
                (dpDest, dpBuffer2, sizeX, sizeZ, dstRowPitch, dstSlicePitch);
            cudaCheckMsg("forwardDWT9ZKernel execution failed");
        }

        int sizeZdone = zBlockCount.z * zResultBlockCount * zBlockSizeY;
        int sizeZrest = sizeZ - sizeZdone;
        if(sizeZrest > 0) {
            dim3 zBlockCountRest(zBlockCount.x, zBlockCount.y, 1);
            int zResultBlockCountRest = (sizeZrest + zBlockSizeY - 1) / zBlockSizeY;
            uint sharedSize = (zBlockSizeX * (zResultBlockCountRest * zBlockSizeY + (FILTER_LENGTH-1) + 1)) * sizeof(float);
            forwardDWT9ZRestKernel
                <zBlockSizeX, zBlockSizeY>
                <<<zBlockCountRest, zBlockSize, sharedSize, stream>>>
                (dpDest, dpBuffer2, sizeZdone, sizeX, sizeZ, zResultBlockCountRest, dstRowPitch, dstSlicePitch);
            cudaCheckMsg("forwardDWT9ZRestKernel execution failed");
        }
    }
}

template<typename TOut, int channelCountOut>
static void dwtFloatInverse(
    TOut* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpSource,
    int sizeX, int sizeY, int sizeZ,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    const int xBlockSizeX = 32;
    const int xBlockSizeY = 4;
    const dim3 xBlockSize(xBlockSizeX, xBlockSizeY);
    const int xResultBlockCount  = 8;
    const int xResultBlockCount2 = 4;
    const int xResultBlockCount3 = 2;

    const int yBlockSizeX = 32;
    const int yBlockSizeY = 4;
    const dim3 yBlockSize(yBlockSizeX, yBlockSizeY);
    const int yResultBlockCount = 8;

    const int zBlockSizeX = 32;
    const int zBlockSizeY = 4;
    const dim3 zBlockSize(zBlockSizeX, zBlockSizeY);
    const int zResultBlockCount = 8;


    bool do3D = (sizeZ > 1);


    if(do3D) {
        dim3 zBlockCount(((sizeX + zBlockSizeX - 1) / zBlockSizeX), sizeY, sizeZ / (zResultBlockCount * zBlockSizeY));

        if(zBlockCount.z > 0) {
            inverseDWT9ZKernel
                <zBlockSizeX, zBlockSizeY, zResultBlockCount>
                <<<zBlockCount, zBlockSize, 0, stream>>>
                (dpBuffer2, dpSource, sizeX, sizeZ, srcRowPitch, srcSlicePitch);
            cudaCheckMsg("inverseDWT9ZKernel execution failed");
        }

        int sizeZdone = zBlockCount.z * zResultBlockCount * zBlockSizeY;
        int sizeZrest = sizeZ - sizeZdone;
        if(sizeZrest > 0) {
            dim3 zBlockCountRest(zBlockCount.x, zBlockCount.y, 1);
            int zResultBlockCountRest = (sizeZrest + zBlockSizeY - 1) / zBlockSizeY;
            uint sharedSize = (zBlockSizeX * (zResultBlockCountRest * zBlockSizeY + (FILTER_LENGTH-1) + 1)) * sizeof(float);
            inverseDWT9ZRestKernel
                <zBlockSizeX, zBlockSizeY>
                <<<zBlockCountRest, zBlockSize, sharedSize, stream>>>
                (dpBuffer2, dpSource, sizeZdone, sizeX, sizeZ, zResultBlockCountRest, srcRowPitch, srcSlicePitch);
            cudaCheckMsg("inverseDWT9ZRestKernel execution failed");
        }
    }


    const float* dpSourceY = (do3D ? dpBuffer2 : dpSource);

    dim3 yBlockCount((sizeX + yBlockSizeX - 1) / yBlockSizeX, sizeY / (yResultBlockCount * yBlockSizeY), sizeZ);

    if(yBlockCount.y > 0) {
        inverseDWT9YKernel
            <yBlockSizeX, yBlockSizeY, yResultBlockCount>
            <<<yBlockCount, yBlockSize, 0, stream>>>
            (dpBuffer1, dpSourceY, sizeX, sizeY, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWT9YKernel execution failed");
    }

    int sizeYdone = yBlockCount.y * yResultBlockCount * yBlockSizeY;
    int sizeYrest = sizeY - sizeYdone;
    if(sizeYrest > 0) {
        dim3 yBlockCountRest(yBlockCount.x, 1, yBlockCount.z);
        int yResultBlockCountRest = (sizeYrest + yBlockSizeY - 1) / yBlockSizeY;
        uint sharedSize = (yBlockSizeX * (yResultBlockCountRest * yBlockSizeY + (FILTER_LENGTH-1) + 1)) * sizeof(float);
        inverseDWT9YRestKernel
            <yBlockSizeX, yBlockSizeY>
            <<<yBlockCountRest, yBlockSize, sharedSize, stream>>>
            (dpBuffer1, dpSourceY, sizeYdone, sizeX, sizeY, yResultBlockCountRest, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWT9YRestKernel execution failed");
    }


    dim3 xBlockCount(sizeX / (xResultBlockCount * xBlockSizeX), ((sizeY + xBlockSizeY - 1) / xBlockSizeY), sizeZ);

    if(xBlockCount.x > 0) {
        inverseDWT9XKernel
            <TOut, channelCountOut, xBlockSizeX, xBlockSizeY, xResultBlockCount>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDest, dpBuffer1, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWT9XKernel execution failed");
    } else if(sizeX == (xResultBlockCount2 * xBlockSizeX)) {
        // special case for sizeX == 128
        xBlockCount.x = 1;
        inverseDWT9XKernel
            <TOut, channelCountOut, xBlockSizeX, xBlockSizeY, xResultBlockCount2>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDest, dpBuffer1, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWT9XKernel execution failed");
    } else if(sizeX == (xResultBlockCount3 * xBlockSizeX)) {
        // special case for sizeX == 64
        xBlockCount.x = 1;
        inverseDWT9XKernel
            <TOut, channelCountOut, xBlockSizeX, xBlockSizeY, xResultBlockCount3>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDest, dpBuffer1, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWT9XKernel execution failed");
    }

    int sizeXdone = xBlockCount.x * xResultBlockCount * xBlockSizeX;
    int sizeXrest = sizeX - sizeXdone;
    if(sizeXrest > 0) {
        dim3 xBlockCountRest(1, xBlockCount.y, xBlockCount.z);
        int xResultBlockCountRest = (sizeXrest + xBlockSizeX - 1) / xBlockSizeX;
        uint sharedSize = (xBlockSizeY * (xResultBlockCountRest * xBlockSizeX + (FILTER_LENGTH-1))) * sizeof(float);
        inverseDWT9XRestKernel
            <TOut, channelCountOut, xBlockSizeX, xBlockSizeY>
            <<<xBlockCountRest, xBlockSize, sharedSize, stream>>>
            (dpDest, dpBuffer1, sizeXdone, sizeX, sizeY, xResultBlockCountRest, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWT9XRestKernel execution failed");
    }
}

template<typename TOut, typename THigh, int channelCountOut>
static void dwtFloatInverseFromSymbols(
    TOut* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpLowpass,
    const THigh*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int sizeZ,
    int dstRowPitch, int dstSlicePitch,
    int lowpassRowPitch, int lowpassSlicePitch,
    cudaStream_t stream)
{
    const int xBlockSizeX = 32;
    const int xBlockSizeY = 4;
    const dim3 xBlockSize(xBlockSizeX, xBlockSizeY);
    const int xResultBlockCount  = 8;
    const int xResultBlockCount2 = 4;
    const int xResultBlockCount3 = 2;

    const int yBlockSizeX = 32;
    const int yBlockSizeY = 4;
    const dim3 yBlockSize(yBlockSizeX, yBlockSizeY);
    const int yResultBlockCount = 8;

    const int zBlockSizeX = 32;
    const int zBlockSizeY = 4;
    const dim3 zBlockSize(zBlockSizeX, zBlockSizeY);
    const int zResultBlockCount = 8;


    bool do3D = (sizeZ > 1);


    int bufferRowPitch = sizeX;
    int bufferSlicePitch = bufferRowPitch * sizeY;


    if(do3D) {
        dim3 zBlockCount(((sizeX + zBlockSizeX - 1) / zBlockSizeX), sizeY, sizeZ / (zResultBlockCount * zBlockSizeY));

        if(zBlockCount.z > 0) {
            inverseDWT9ZFromSymbolsKernel
                <THigh, zBlockSizeX, zBlockSizeY, zResultBlockCount>
                <<<zBlockCount, zBlockSize, 0, stream>>>
                (dpBuffer2, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, sizeZ, bufferRowPitch, bufferSlicePitch, lowpassRowPitch, lowpassSlicePitch);
            cudaCheckMsg("inverseDWT9ZFromSymbolsKernel execution failed");
        }

        int sizeZdone = zBlockCount.z * zResultBlockCount * zBlockSizeY;
        int sizeZrest = sizeZ - sizeZdone;
        if(sizeZrest > 0) {
            dim3 zBlockCountRest(zBlockCount.x, zBlockCount.y, 1);
            int zResultBlockCountRest = (sizeZrest + zBlockSizeY - 1) / zBlockSizeY;
            uint sharedSize = (zBlockSizeX * (zResultBlockCountRest * zBlockSizeY + (FILTER_LENGTH-1) + 1)) * sizeof(float);
            inverseDWT9ZFromSymbolsRestKernel
                <THigh, zBlockSizeX, zBlockSizeY>
                <<<zBlockCountRest, zBlockSize, sharedSize, stream>>>
                (dpBuffer2, dpLowpass, dppHighpass, quantStep, sizeZdone, sizeX, sizeY, sizeZ, zResultBlockCountRest, bufferRowPitch, bufferSlicePitch, lowpassRowPitch, lowpassSlicePitch);
            cudaCheckMsg("inverseDWT9ZFromSymbolsRestKernel execution failed");
        }
    }


    dim3 yBlockCount((sizeX + yBlockSizeX - 1) / yBlockSizeX, sizeY / (yResultBlockCount * yBlockSizeY), sizeZ);

    if(yBlockCount.y > 0) {
        if(do3D) {
            inverseDWT9YKernel
                <yBlockSizeX, yBlockSizeY, yResultBlockCount>
                <<<yBlockCount, yBlockSize, 0, stream>>>
                (dpBuffer1, dpBuffer2, sizeX, sizeY, bufferRowPitch, bufferSlicePitch);
            cudaCheckMsg("inverseDWT9YKernel execution failed");
        } else {
            inverseDWT9YFromSymbolsKernel
                <THigh, yBlockSizeX, yBlockSizeY, yResultBlockCount>
                <<<yBlockCount, yBlockSize, 0, stream>>>
                (dpBuffer1, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, bufferRowPitch, lowpassRowPitch);
            cudaCheckMsg("inverseDWT9YFromSymbolsKernel execution failed");
        }
    }

    int sizeYdone = yBlockCount.y * yResultBlockCount * yBlockSizeY;
    int sizeYrest = sizeY - sizeYdone;
    if(sizeYrest > 0) {
        dim3 yBlockCountRest(yBlockCount.x, 1, yBlockCount.z);
        int yResultBlockCountRest = (sizeYrest + yBlockSizeY - 1) / yBlockSizeY;
        uint sharedSize = (yBlockSizeX * (yResultBlockCountRest * yBlockSizeY + (FILTER_LENGTH-1) + 1)) * sizeof(float);
        if(do3D) {
            inverseDWT9YRestKernel
                <yBlockSizeX, yBlockSizeY>
                <<<yBlockCountRest, yBlockSize, sharedSize, stream>>>
                (dpBuffer1, dpBuffer2, sizeYdone, sizeX, sizeY, yResultBlockCountRest, bufferRowPitch, bufferSlicePitch);
            cudaCheckMsg("inverseDWT9YRestKernel execution failed");
        } else {
            inverseDWT9YFromSymbolsRestKernel
                <THigh, yBlockSizeX, yBlockSizeY>
                <<<yBlockCountRest, yBlockSize, sharedSize, stream>>>
                (dpBuffer1, dpLowpass, dppHighpass, quantStep, sizeYdone, sizeX, sizeY, yResultBlockCountRest, bufferRowPitch, lowpassRowPitch);
            cudaCheckMsg("inverseDWT9YFromSymbolsRestKernel execution failed");
        }
    }


    dim3 xBlockCount(sizeX / (xResultBlockCount * xBlockSizeX), ((sizeY + xBlockSizeY - 1) / xBlockSizeY), sizeZ);

    if(xBlockCount.x > 0) {
        inverseDWT9XKernel
            <TOut, channelCountOut, xBlockSizeX, xBlockSizeY, xResultBlockCount>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDest, dpBuffer1, sizeX, sizeY, dstRowPitch, dstSlicePitch, bufferRowPitch, bufferSlicePitch);
        cudaCheckMsg("inverseDWT9XKernel execution failed");
    } else if(sizeX == (xResultBlockCount2 * xBlockSizeX)) {
        // special case for sizeX == 128
        xBlockCount.x = 1;
        inverseDWT9XKernel
            <TOut, channelCountOut, xBlockSizeX, xBlockSizeY, xResultBlockCount2>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDest, dpBuffer1, sizeX, sizeY, dstRowPitch, dstSlicePitch, bufferRowPitch, bufferSlicePitch);
        cudaCheckMsg("inverseDWT9XKernel execution failed");
    } else if(sizeX == (xResultBlockCount3 * xBlockSizeX)) {
        // special case for sizeX == 64
        xBlockCount.x = 1;
        inverseDWT9XKernel
            <TOut, channelCountOut, xBlockSizeX, xBlockSizeY, xResultBlockCount3>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDest, dpBuffer1, sizeX, sizeY, dstRowPitch, dstSlicePitch, bufferRowPitch, bufferSlicePitch);
        cudaCheckMsg("inverseDWT9XKernel execution failed");
    }

    int sizeXdone = xBlockCount.x * xResultBlockCount * xBlockSizeX;
    int sizeXrest = sizeX - sizeXdone;
    if(sizeXrest > 0) {
        dim3 xBlockCountRest(1, xBlockCount.y, xBlockCount.z);
        int xResultBlockCountRest = (sizeXrest + xBlockSizeX - 1) / xBlockSizeX;
        uint sharedSize = (xBlockSizeY * (xResultBlockCountRest * xBlockSizeX + (FILTER_LENGTH-1))) * sizeof(float);
        inverseDWT9XRestKernel
            <TOut, channelCountOut, xBlockSizeX, xBlockSizeY>
            <<<xBlockCountRest, xBlockSize, sharedSize, stream>>>
            (dpDest, dpBuffer1, sizeXdone, sizeX, sizeY, xResultBlockCountRest, dstRowPitch, dstSlicePitch, bufferRowPitch, bufferSlicePitch);
        cudaCheckMsg("inverseDWT9XRestKernel execution failed");
    }
}



template<typename TIn>
static void dwtFloatForward(
    float* dpDest, float* dpBuffer2, float* dpBuffer1, const TIn* dpSource,
    int sizeX, int sizeY, int sizeZ, int srcChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    if(dstRowPitch <= 0)   dstRowPitch   = sizeX;
    if(dstSlicePitch <= 0) dstSlicePitch = sizeY * dstRowPitch;
    if(srcRowPitch <= 0)   srcRowPitch   = sizeX * srcChannelCount;
    if(srcSlicePitch <= 0) srcSlicePitch = sizeY * srcRowPitch;

    switch(srcChannelCount) {
        case 1: dwtFloatForward<TIn, 1>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream); break;
        case 2: dwtFloatForward<TIn, 2>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream); break;
        case 3: dwtFloatForward<TIn, 3>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream); break;
        case 4: dwtFloatForward<TIn, 4>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream); break;
        default: assert(false);
    }
}

template<typename TOut>
static void dwtFloatInverse(
    TOut* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpSource,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    if(dstRowPitch <= 0)   dstRowPitch   = sizeX * dstChannelCount;
    if(dstSlicePitch <= 0) dstSlicePitch = sizeY * dstRowPitch;
    if(srcRowPitch <= 0)   srcRowPitch   = sizeX;
    if(srcSlicePitch <= 0) srcSlicePitch = sizeY * srcRowPitch;

    switch(dstChannelCount) {
        case 1: dwtFloatInverse<TOut, 1>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream); break;
        case 2: dwtFloatInverse<TOut, 2>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream); break;
        case 3: dwtFloatInverse<TOut, 3>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream); break;
        case 4: dwtFloatInverse<TOut, 4>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream); break;
        default: assert(false);
    }
}

template<typename TOut, typename THigh>
void dwtFloatInverseFromSymbols(
    TOut* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpLowpass, const THigh*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int lowpassRowPitch, int lowpassSlicePitch,
    cudaStream_t stream)
{
    if(dstRowPitch <= 0)   dstRowPitch   = sizeX * dstChannelCount;
    if(dstSlicePitch <= 0) dstSlicePitch = sizeY * dstRowPitch;
    if(lowpassRowPitch <= 0)   lowpassRowPitch   = sizeX;
    if(lowpassSlicePitch <= 0) lowpassSlicePitch = sizeY * lowpassRowPitch;

    switch(dstChannelCount) {
        case 1: dwtFloatInverseFromSymbols<TOut, THigh, 1>(dpDest, dpBuffer2, dpBuffer1, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, lowpassRowPitch, lowpassSlicePitch, stream); break;
        case 2: dwtFloatInverseFromSymbols<TOut, THigh, 2>(dpDest, dpBuffer2, dpBuffer1, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, lowpassRowPitch, lowpassSlicePitch, stream); break;
        case 3: dwtFloatInverseFromSymbols<TOut, THigh, 3>(dpDest, dpBuffer2, dpBuffer1, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, lowpassRowPitch, lowpassSlicePitch, stream); break;
        case 4: dwtFloatInverseFromSymbols<TOut, THigh, 4>(dpDest, dpBuffer2, dpBuffer1, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, sizeZ, dstRowPitch, dstSlicePitch, lowpassRowPitch, lowpassSlicePitch, stream); break;
        default: assert(false);
    }
}


template<typename T>
static void dwtFloatForwardLowpassOnly(
    float* dpDest, float* dpBuffer, const T* dpSource,
    int sizeX, int sizeY, int srcChannelCount,
    int dstRowPitch,
    int srcRowPitch,
    cudaStream_t stream)
{
    if(dstRowPitch <= 0) dstRowPitch = sizeX / 2;
    if(srcRowPitch <= 0) srcRowPitch = sizeX * srcChannelCount;

    switch(srcChannelCount) {
        case 1: dwtFloatForwardLowpassOnly2D<T, 1>(dpDest, dpBuffer, dpSource, sizeX, sizeY, dstRowPitch, srcRowPitch, stream); break;
        case 2: dwtFloatForwardLowpassOnly2D<T, 2>(dpDest, dpBuffer, dpSource, sizeX, sizeY, dstRowPitch, srcRowPitch, stream); break;
        case 3: dwtFloatForwardLowpassOnly2D<T, 3>(dpDest, dpBuffer, dpSource, sizeX, sizeY, dstRowPitch, srcRowPitch, stream); break;
        case 4: dwtFloatForwardLowpassOnly2D<T, 4>(dpDest, dpBuffer, dpSource, sizeX, sizeY, dstRowPitch, srcRowPitch, stream); break;
        default: assert(false);
    }
}







void dwtFloat2DForward(
    float* dpDest, float* dpBuffer, const float* dpSource,
    int sizeX, int sizeY, int srcChannelCount,
    int dstRowPitch,
    int srcRowPitch,
    cudaStream_t stream)
{
    dwtFloatForward<float>(dpDest, nullptr, dpBuffer, dpSource, sizeX, sizeY, 1, srcChannelCount, dstRowPitch, 0, srcRowPitch, 0, stream);
}

void dwtFloat3DForward(
    float* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpSource,
    int sizeX, int sizeY, int sizeZ, int srcChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    dwtFloatForward<float>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, srcChannelCount, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream);
}


void dwtFloat2DInverse(
    float* dpDest, float* dpBuffer, const float* dpSource,
    int sizeX, int sizeY, int dstChannelCount,
    int dstRowPitch,
    int srcRowPitch,
    cudaStream_t stream)
{
    dwtFloatInverse<float>(dpDest, nullptr, dpBuffer, dpSource, sizeX, sizeY, 1, dstChannelCount, dstRowPitch, 0, srcRowPitch, 0, stream);
}

void dwtFloat3DInverse(
    float* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpSource,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    dwtFloatInverse<float>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstChannelCount, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream);
}


void dwtFloat2DInverseFromSymbols(
    float* dpDest, float* dpBuffer,
    const float* dpLowpass, const ushort*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int dstChannelCount,
    int dstRowPitch,
    int lowpassRowPitch,
    cudaStream_t stream)
{
    dwtFloatInverseFromSymbols<float, ushort>(dpDest, nullptr, dpBuffer, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, 1, dstChannelCount, dstRowPitch, 0, lowpassRowPitch, 0, stream);
}

void dwtFloat3DInverseFromSymbols(
    float* dpDest, float* dpBuffer2, float* dpBuffer1,
    const float* dpLowpass, const ushort*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int lowpassRowPitch, int lowpassSlicePitch,
    cudaStream_t stream)
{
    dwtFloatInverseFromSymbols<float, ushort>(dpDest, dpBuffer2, dpBuffer1, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, sizeZ, dstChannelCount, dstRowPitch, dstSlicePitch, lowpassRowPitch, lowpassSlicePitch, stream);
}


void dwtFloat2DInverseFromSymbols(
    float* dpDest, float* dpBuffer,
    const float* dpLowpass, const uint*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int dstChannelCount,
    int dstRowPitch,
    int lowpassRowPitch,
    cudaStream_t stream)
{
    dwtFloatInverseFromSymbols<float, uint>(dpDest, nullptr, dpBuffer, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, 1, dstChannelCount, dstRowPitch, 0, lowpassRowPitch, 0, stream);
}

void dwtFloat3DInverseFromSymbols(
    float* dpDest, float* dpBuffer2, float* dpBuffer1,
    const float* dpLowpass, const uint*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int lowpassRowPitch, int lowpassSlicePitch,
    cudaStream_t stream)
{
    dwtFloatInverseFromSymbols<float, uint>(dpDest, dpBuffer2, dpBuffer1, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, sizeZ, dstChannelCount, dstRowPitch, dstSlicePitch, lowpassRowPitch, lowpassSlicePitch, stream);
}



void dwtFloat2DForwardFromByte(
    float* dpDest, float* dpBuffer, const byte* dpSource,
    int sizeX, int sizeY, int srcChannelCount,
    int dstRowPitch,
    int srcRowPitch,
    cudaStream_t stream)
{
    dwtFloatForward<byte>(dpDest, nullptr, dpBuffer, dpSource, sizeX, sizeY, 1, srcChannelCount, dstRowPitch, 0, srcRowPitch, 0, stream);
}

void dwtFloat3DForwardFromByte(
    float* dpDest, float* dpBuffer2, float* dpBuffer1, const byte* dpSource,
    int sizeX, int sizeY, int sizeZ, int srcChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    dwtFloatForward<byte>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, srcChannelCount, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream);
}


void dwtFloat2DInverseToByte(
    byte* dpDest, float* dpBuffer, const float* dpSource,
    int sizeX, int sizeY, int dstChannelCount,
    int dstRowPitch,
    int srcRowPitch,
    cudaStream_t stream)
{
    dwtFloatInverse<byte>(dpDest, nullptr, dpBuffer, dpSource, sizeX, sizeY, 1, dstChannelCount, dstRowPitch, 0, srcRowPitch, 0, stream);
}

void dwtFloat3DInverseToByte(
    byte* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpSource,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    dwtFloatInverse<byte>(dpDest, dpBuffer2, dpBuffer1, dpSource, sizeX, sizeY, sizeZ, dstChannelCount, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch, stream);
}


void dwtFloat2DInverseFromSymbolsToByte(
    byte* dpDest, float* dpBuffer,
    const float* dpLowpass, const ushort*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int dstChannelCount,
    int dstRowPitch,
    int lowpassRowPitch,
    cudaStream_t stream)
{
    dwtFloatInverseFromSymbols<byte>(dpDest, nullptr, dpBuffer, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, 1, dstChannelCount, dstRowPitch, 0, lowpassRowPitch, 0, stream);
}

void dwtFloat3DInverseFromSymbolsToByte(
    byte* dpDest, float* dpBuffer2, float* dpBuffer1,
    const float* dpLowpass, const ushort*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount,
    int dstRowPitch, int dstSlicePitch,
    int lowpassRowPitch, int lowpassSlicePitch,
    cudaStream_t stream)
{
    dwtFloatInverseFromSymbols<byte>(dpDest, dpBuffer2, dpBuffer1, dpLowpass, dppHighpass, quantStep, sizeX, sizeY, sizeZ, dstChannelCount, dstRowPitch, dstSlicePitch, lowpassRowPitch, lowpassSlicePitch, stream);
}


void dwtFloat2DForwardLowpassOnlyFromByte(
    float* dpDest, float* dpBuffer, const byte* dpSource,
    int sizeX, int sizeY, int srcChannelCount,
    int dstRowPitch,
    int srcRowPitch,
    cudaStream_t stream)
{
    dwtFloatForwardLowpassOnly<byte>(dpDest, dpBuffer, dpSource, sizeX, sizeY, srcChannelCount, dstRowPitch, srcRowPitch, stream);
}


}

}
