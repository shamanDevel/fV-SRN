#include "memtrace.h"
#include <cudaCompress/util/DWT.h>

#include <assert.h>

#include <cudaCompress/cudaUtil.h>

#include "DWTIntKernels.cui"
#include "DWTInt2DLowpassKernels.cui"


namespace cudaCompress {

namespace util {


void dwtIntForwardLowpassOnly(
    short* dpDest, short* dpBuffer, const short* dpSource,
    int sizeX, int sizeY,
    int dstRowPitch,
    int srcRowPitch,
    cudaStream_t stream)
{
    if(dstRowPitch <= 0) dstRowPitch = sizeX / 2;
    if(srcRowPitch <= 0) srcRowPitch = sizeX;


    const int xBlockSizeX = 32;
    const int xBlockSizeY = 4;
    const dim3 xBlockSize(xBlockSizeX, xBlockSizeY);
    const int xResultBlockCount = 16;

    const int yBlockSizeX = 32;
    const int yBlockSizeY = 4;
    const dim3 yBlockSize(yBlockSizeX, yBlockSizeY);
    const int yResultBlockCount = 16;


    dim3 xBlockCount(sizeX / (xResultBlockCount * xBlockSizeX), (sizeY + xBlockSizeY - 1) / xBlockSizeY);

    if(xBlockCount.x > 0) {
        forwardDWTIntXLowpassKernel2D
            <xBlockSizeX, xBlockSizeY, xResultBlockCount>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpBuffer, dpSource, sizeX, sizeY, dstRowPitch, srcRowPitch);
        cudaCheckMsg("forwardDWTIntXLowpassKernel2D execution failed");
    }

    //int sizeXdone = xBlockCount.x * xResultBlockCount * xBlockSizeX;
    //int sizeXrest = sizeX - sizeXdone;
    //if(sizeXrest > 0) {
    //    dim3 xBlockCountRest(1, xBlockCount.y);
    //    int xResultBlockCountRest = (sizeXrest + xBlockSizeX - 1) / xBlockSizeX;
    //        case 8: forwardDWTIntXRestKernel2D<TIn, channelCountIn, xBlockSizeX, xBlockSizeY, 8><<<xBlockCountRest, xBlockSize, 0, stream>>>(dpBuffer, dpSource, sizeXdone, sizeX, sizeY, dstRowPitch, srcRowPitch); break;
    //    cudaCheckMsg("forwardDWTIntXRestKernel2D execution failed");
    //}


    dim3 yBlockCount((sizeX/2 + yBlockSizeX - 1) / yBlockSizeX, sizeY / (yResultBlockCount * yBlockSizeY));

    if(yBlockCount.y > 0) {
        forwardDWTIntYLowpassKernel2D
            <yBlockSizeX, yBlockSizeY, yResultBlockCount>
            <<<yBlockCount, yBlockSize, 0, stream>>>
            (dpDest, dpBuffer, sizeX/2, sizeY, dstRowPitch);
        cudaCheckMsg("forwardDWTIntYLowpassKernel2D execution failed");
    }

    //int sizeYdone = yBlockCount.y * yResultBlockCount * yBlockSizeY;
    //int sizeYrest = sizeY - sizeYdone;
    //if(sizeYrest > 0) {
    //    dim3 yBlockCountRest(yBlockCount.x, 1);
    //    int yResultBlockCountRest = (sizeYrest + yBlockSizeY - 1) / yBlockSizeY;
    //        case 8: forwardDWTIntYRestKernel2D<yBlockSizeX, yBlockSizeY, 8><<<yBlockCountRest, yBlockSize, 0, stream>>>(dpDest, dpBuffer, sizeYdone, sizeX, sizeY, dstRowPitch); break;
    //    cudaCheckMsg("forwardDWTIntYRestKernel2D execution failed");
    //}
}


void dwtIntForward(
    short* dpDest, short* dpBuffer, const short* dpSource,
    int sizeX, int sizeY, int sizeZ,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    if(dstRowPitch <= 0)   dstRowPitch   = sizeX;
    if(dstSlicePitch <= 0) dstSlicePitch = sizeY * dstRowPitch;
    if(srcRowPitch <= 0)   srcRowPitch   = sizeX;
    if(srcSlicePitch <= 0) srcSlicePitch = sizeY * srcRowPitch;


    const int xBlockSizeX = 32;
    const int xBlockSizeY = 4;
    const dim3 xBlockSize(xBlockSizeX, xBlockSizeY);
    const int xResultBlockCount = 8;

    const int yBlockSizeX = 32;
    const int yBlockSizeY = 4;
    const dim3 yBlockSize(yBlockSizeX, yBlockSizeY);
    const int yResultBlockCount = 8;

    const int zBlockSizeX = 32;
    const int zBlockSizeY = 4;
    const dim3 zBlockSize(zBlockSizeX, zBlockSizeY);
    const int zResultBlockCount = 8;


    bool do3D = (sizeZ > 1);


    short* dpDestX = (do3D ? dpDest : dpBuffer);

    dim3 xBlockCount(sizeX / (xResultBlockCount * xBlockSizeX), (sizeY + xBlockSizeY - 1) / xBlockSizeY, sizeZ);

    if(xBlockCount.x > 0) {
        forwardDWTIntXKernel
            <xBlockSizeX, xBlockSizeY, xResultBlockCount>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDestX, dpSource, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("forwardDWTIntXKernel execution failed");
    }

    int sizeXdone = xBlockCount.x * xResultBlockCount * xBlockSizeX;
    int sizeXrest = sizeX - sizeXdone;
    if(sizeXrest > 0) {
        dim3 xBlockCountRest(1, xBlockCount.y, xBlockCount.z);
        int xResultBlockCountRest = (sizeXrest + xBlockSizeX - 1) / xBlockSizeX;
        uint sharedSize = (xBlockSizeY * (xResultBlockCountRest * xBlockSizeX + OVERLAP_TOTAL)) * sizeof(short);
        forwardDWTIntXRestKernel
            <xBlockSizeX, xBlockSizeY>
            <<<xBlockCountRest, xBlockSize, sharedSize, stream>>>
            (dpDestX, dpSource, sizeXdone, sizeX, sizeY, xResultBlockCountRest, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("forwardDWTIntXRestKernel2D execution failed");
    }


    const short* dpSourceY = dpDestX;
    short* dpDestY = (do3D ? dpBuffer : dpDest);

    dim3 yBlockCount((sizeX + yBlockSizeX - 1) / yBlockSizeX, sizeY / (yResultBlockCount * yBlockSizeY), sizeZ);

    if(yBlockCount.y > 0) {
        forwardDWTIntYKernel
            <yBlockSizeX, yBlockSizeY, yResultBlockCount>
            <<<yBlockCount, yBlockSize, 0, stream>>>
            (dpDestY, dpSourceY, sizeX, sizeY, dstRowPitch, dstSlicePitch);
        cudaCheckMsg("forwardDWTIntYKernel execution failed");
    }

    int sizeYdone = yBlockCount.y * yResultBlockCount * yBlockSizeY;
    int sizeYrest = sizeY - sizeYdone;
    if(sizeYrest > 0) {
        dim3 yBlockCountRest(yBlockCount.x, 1, yBlockCount.z);
        int yResultBlockCountRest = (sizeYrest + yBlockSizeY - 1) / yBlockSizeY;
        uint sharedSize = ((yResultBlockCountRest * yBlockSizeY + OVERLAP_TOTAL) * yBlockSizeX) * sizeof(short);
        forwardDWTIntYRestKernel
            <yBlockSizeX, yBlockSizeY>
            <<<yBlockCountRest, yBlockSize, sharedSize, stream>>>
            (dpDestY, dpSourceY, sizeYdone, sizeX, sizeY, yResultBlockCountRest, dstRowPitch, dstSlicePitch);
        cudaCheckMsg("forwardDWTIntYRestKernel execution failed");
    }


    if(do3D) {
        dim3 zBlockCount((sizeX + zBlockSizeX - 1) / zBlockSizeX, sizeY, sizeZ / (zResultBlockCount * zBlockSizeY));
        if(zBlockCount.z > 0) {
            forwardDWTIntZKernel
                <zBlockSizeX, zBlockSizeY, zResultBlockCount>
                <<<zBlockCount, zBlockSize, 0, stream>>>
                (dpDest, dpBuffer, sizeX, sizeZ, dstRowPitch, dstSlicePitch);
            cudaCheckMsg("forwardDWTIntZKernel execution failed");
        }

        int sizeZdone = zBlockCount.z * zResultBlockCount * zBlockSizeY;
        int sizeZrest = sizeZ - sizeZdone;
        if(sizeZrest > 0) {
            dim3 zBlockCountRest(zBlockCount.x, zBlockCount.y, 1);
            int zResultBlockCountRest = (sizeZrest + zBlockSizeY - 1) / zBlockSizeY;
            uint sharedSize = ((zResultBlockCountRest * zBlockSizeY + OVERLAP_TOTAL) * zBlockSizeX) * sizeof(short);
            forwardDWTIntZRestKernel
                <zBlockSizeX, zBlockSizeY>
                <<<zBlockCountRest, zBlockSize, sharedSize, stream>>>
                (dpDest, dpBuffer, sizeZdone, sizeX, sizeZ, zResultBlockCountRest, dstRowPitch, dstSlicePitch);
            cudaCheckMsg("forwardDWTIntZRestKernel execution failed");
        }
    }
}

void dwtIntInverse(short* dpDest, short* dpBuffer, const short* dpSource,
    int sizeX, int sizeY, int sizeZ,
    int dstRowPitch, int dstSlicePitch,
    int srcRowPitch, int srcSlicePitch,
    cudaStream_t stream)
{
    if(dstRowPitch <= 0)   dstRowPitch   = sizeX;
    if(dstSlicePitch <= 0) dstSlicePitch = sizeY * dstRowPitch;
    if(srcRowPitch <= 0)   srcRowPitch   = sizeX;
    if(srcSlicePitch <= 0) srcSlicePitch = sizeY * srcRowPitch;


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
        dim3 zBlockCount((sizeX + zBlockSizeX - 1) / zBlockSizeX, sizeY, sizeZ / (zResultBlockCount * zBlockSizeY));

        if(zBlockCount.z > 0) {
            inverseDWTIntZKernel
                <zBlockSizeX, zBlockSizeY, zResultBlockCount>
                <<<zBlockCount, zBlockSize, 0, stream>>>
                (dpDest, dpSource, sizeX, sizeZ, srcRowPitch, srcSlicePitch);
            cudaCheckMsg("inverseDWTIntZKernel execution failed");
        }

        int sizeZdone = zBlockCount.z * zResultBlockCount * zBlockSizeY;
        int sizeZrest = sizeZ - sizeZdone;
        if(sizeZrest > 0) {
            dim3 zBlockCountRest(zBlockCount.x, zBlockCount.y, 1);
            int zResultBlockCountRest = (sizeZrest + zBlockSizeY - 1) / zBlockSizeY;
            uint sharedSize = ((zResultBlockCountRest * zBlockSizeY + 4) * zBlockSizeX) * sizeof(short);
            inverseDWTIntZRestKernel
                <zBlockSizeX, zBlockSizeY>
                <<<zBlockCountRest, zBlockSize, sharedSize, stream>>>
                (dpDest, dpSource, sizeZdone, sizeX, sizeZ, zResultBlockCountRest, srcRowPitch, srcSlicePitch);
            cudaCheckMsg("inverseDWTIntZRestKernel execution failed");
        }
    }


    const short* dpSourceY = (do3D ? dpDest : dpSource);

    dim3 yBlockCount((sizeX + yBlockSizeX - 1) / yBlockSizeX, sizeY / (yResultBlockCount * yBlockSizeY), sizeZ);

    if(yBlockCount.y > 0) {
        inverseDWTIntYKernel
            <yBlockSizeX, yBlockSizeY, yResultBlockCount>
            <<<yBlockCount, yBlockSize, 0, stream>>>
            (dpBuffer, dpSourceY, sizeX, sizeY, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWTIntYKernel execution failed");
    }

    int sizeYdone = yBlockCount.y * yResultBlockCount * yBlockSizeY;
    int sizeYrest = sizeY - sizeYdone;
    if(sizeYrest > 0) {
        dim3 yBlockCountRest(yBlockCount.x, 1, yBlockCount.z);
        int yResultBlockCountRest = (sizeYrest + yBlockSizeY - 1) / yBlockSizeY;
        uint sharedSize = ((yResultBlockCountRest * yBlockSizeY + 4) * yBlockSizeX) * sizeof(short);
        inverseDWTIntYRestKernel
            <yBlockSizeX, yBlockSizeY>
            <<<yBlockCountRest, yBlockSize, sharedSize, stream>>>
            (dpBuffer, dpSourceY, sizeYdone, sizeX, sizeY, yResultBlockCountRest, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWTIntYRestKernel execution failed");
    }


    dim3 xBlockCount(sizeX / (xResultBlockCount * xBlockSizeX), (sizeY + xBlockSizeY - 1) / xBlockSizeY, sizeZ);

    if(xBlockCount.x > 0) {
        inverseDWTIntXKernel
            <xBlockSizeX, xBlockSizeY, xResultBlockCount>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDest, dpBuffer, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWTIntXKernel execution failed");
    } else if(sizeX == (xResultBlockCount2 * xBlockSizeX)) {
        // special case for sizeX == 128
        xBlockCount.x = 1;
        inverseDWTIntXKernel
            <xBlockSizeX, xBlockSizeY, xResultBlockCount2>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDest, dpBuffer, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWTIntXKernel execution failed");
    } else if(sizeX == (xResultBlockCount3 * xBlockSizeX)) {
        // special case for sizeX == 64
        xBlockCount.x = 1;
        inverseDWTIntXKernel
            <xBlockSizeX, xBlockSizeY, xResultBlockCount3>
            <<<xBlockCount, xBlockSize, 0, stream>>>
            (dpDest, dpBuffer, sizeX, sizeY, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWTIntXKernel execution failed");
    }

    int sizeXdone = xBlockCount.x * xResultBlockCount * xBlockSizeX;
    int sizeXrest = sizeX - sizeXdone;
    if(sizeXrest > 0) {
        dim3 xBlockCountRest(1, xBlockCount.y, xBlockCount.z);
        int xResultBlockCountRest = (sizeXrest + xBlockSizeX - 1) / xBlockSizeX;
        uint sharedSize = (xBlockSizeY * (xResultBlockCountRest * xBlockSizeX + 4)) * sizeof(short);
        inverseDWTIntXRestKernel
            <xBlockSizeX, xBlockSizeY>
            <<<xBlockCountRest, xBlockSize, sharedSize, stream>>>
            (dpDest, dpBuffer, sizeXdone, sizeX, sizeY, xResultBlockCountRest, dstRowPitch, dstSlicePitch, srcRowPitch, srcSlicePitch);
        cudaCheckMsg("inverseDWTIntXRestKernel execution failed");
    }
}


}

}
