#include "memtrace.h"
#include <cudaCompress/Histogram.h>

#include <cassert>

#include <cuda_runtime.h>

#include <cudaCompress/cudaUtil.h>
#include <cudaCompress/util.h>

#include <cudaCompress/InstanceImpl.h>

#include "HistogramKernels.cui"


namespace cudaCompress {

static const uint MAX_PARTIAL_HISTOGRAM_COUNT = 32768;


size_t histogramGetRequiredMemory(const Instance* pInstance)
{
    uint histogramCountMax = pInstance->m_streamCountMax;

    size_t size = 0;

    // dpPartialHistograms
    // this is tailored to the smallHistogram kernel, but largeHistogram needs less memory
    size += getAlignedSize(MAX_PARTIAL_HISTOGRAM_COUNT * SMALL_HISTOGRAM_MAX_BIN_COUNT * sizeof(uint), 128);

    // dppHistograms
    size += getAlignedSize(histogramCountMax * sizeof(uint*), 128);
    // dppData
    size += getAlignedSize(histogramCountMax * sizeof(ushort*), 128);
    // dpElemCount
    size += getAlignedSize(histogramCountMax * sizeof(uint), 128);

    return size;
}

bool histogramInit(Instance* pInstance)
{
    uint histogramCountMax = pInstance->m_streamCountMax;

    static_assert(SMALL_HISTOGRAM_THREADBLOCK_SIZE % (4 * SHARED_MEMORY_BANKS) == 0, "SMALL_HISTOGRAM_THREADBLOCK_SIZE % (4 * SHARED_MEMORY_BANKS) must equal 0");

    size_t maxUploadSize = getAlignedSize(histogramCountMax * sizeof(uint*), 128);
    maxUploadSize += getAlignedSize(histogramCountMax * sizeof(ushort*), 128);
    maxUploadSize += getAlignedSize(histogramCountMax * sizeof(uint), 128);
    cudaSafeCall(cudaMallocHost(&pInstance->Histogram.pUpload, maxUploadSize, cudaHostAllocWriteCombined));

    cudaSafeCall(cudaEventCreateWithFlags(&pInstance->Histogram.syncEvent, cudaEventDisableTiming));
    // immediately record to signal that buffers are ready to use (ie first cudaEventSynchronize works)
    cudaSafeCall(cudaEventRecord(pInstance->Histogram.syncEvent));

    return true;
}

bool histogramShutdown(Instance* pInstance)
{
    cudaSafeCall(cudaEventDestroy(pInstance->Histogram.syncEvent));
    pInstance->Histogram.syncEvent = 0;

    cudaSafeCall(cudaFreeHost(pInstance->Histogram.pUpload));
    pInstance->Histogram.pUpload = nullptr;

    return true;
}

//Round a / b to nearest higher integer value
inline uint iDivUp(uint a, uint b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Snap a to nearest lower multiple of b
inline uint iSnapDown(uint a, uint b) {
    return a - a % b;
}

uint histogramGetSizeIncrement()
{
    return sizeof(data_t) / sizeof(ushort);
}

uint histogramGetPaddedElemCount(uint elemCount)
{
    uint sizeIncrement = histogramGetSizeIncrement();
    return (elemCount + sizeIncrement - 1) / sizeIncrement * sizeIncrement;
}

void histogramPadData(Instance* pInstance, ushort* dpData, uint elemCount)
{
    uint elemCountPadded = histogramGetPaddedElemCount(elemCount);
    uint padBytes = (elemCountPadded - elemCount) * sizeof(ushort);
    if(padBytes > 0) {
        cudaSafeCall(cudaMemsetAsync(dpData + elemCount, 0xFF, padBytes, pInstance->m_stream));
    }
}

void histogramPadData(Instance* pInstance, uint* dpData, uint elemCount)
{
    uint elemCountPadded = histogramGetPaddedElemCount(elemCount);
    uint padBytes = (elemCountPadded - elemCount) * sizeof(uint);
    if(padBytes > 0) {
        cudaSafeCall(cudaMemsetAsync(dpData + elemCount, 0xFF, padBytes, pInstance->m_stream));
    }
}


template<typename T>
bool histogram(Instance* pInstance, uint* pdpHistograms[], uint histogramCount, const T* pdpData[], const uint* pElemCount, uint binCount)
{
    // find max number of elements per histogram
    uint elemCountMax = 0;
    for(uint i = 0; i < histogramCount; i++) {
        elemCountMax = max(elemCountMax, pElemCount[i]);
    }

    const uint partialHistogramCount = iDivUp(elemCountMax, SMALL_HISTOGRAM_THREADBLOCK_SIZE * iSnapDown(255, sizeof(data_t)));

    if(partialHistogramCount == 0)
        return true;


    uint* dpPartialHistograms = pInstance->getBuffer<uint>(MAX_PARTIAL_HISTOGRAM_COUNT * SMALL_HISTOGRAM_MAX_BIN_COUNT);
    uint** dppHistograms = pInstance->getBuffer<uint*>(histogramCount);
    T** dppData = pInstance->getBuffer<T*>(histogramCount);
    uint* dpElemCount = pInstance->getBuffer<uint>(histogramCount);


    // copy pointers and elemcount to pinned memory
    byte* pUploadHistogramPtrs = pInstance->Histogram.pUpload;
    uint size = histogramCount * sizeof(uint*);
    memcpy(pUploadHistogramPtrs, pdpHistograms, size);
    byte* pUploadDataPtrs = pUploadHistogramPtrs + ((size + 127) / 128 * 128);
    size = histogramCount * sizeof(T*);
    memcpy(pUploadDataPtrs, pdpData, size);
    byte* pUploadElemCounts = pUploadDataPtrs + ((size + 127) / 128 * 128);
    size = histogramCount * sizeof(uint);
    memcpy(pUploadElemCounts, pElemCount, size);

    // upload pointers and elemcount
    // first make sure cpu buffers are not in use anymore
    cudaSafeCall(cudaEventSynchronize(pInstance->Histogram.syncEvent));

    cudaSafeCall(cudaMemcpyAsync(dppHistograms, pUploadHistogramPtrs, histogramCount * sizeof(uint*), cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(dppData,       pUploadDataPtrs,      histogramCount * sizeof(T*),    cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(dpElemCount,   pUploadElemCounts,    histogramCount * sizeof(uint),  cudaMemcpyHostToDevice, pInstance->m_stream));

    cudaSafeCall(cudaEventRecord(pInstance->Histogram.syncEvent, pInstance->m_stream));


    assert( partialHistogramCount * histogramCount <= MAX_PARTIAL_HISTOGRAM_COUNT );


    uint smallHistogramPassCount = (binCount + SMALL_HISTOGRAM_MAX_BIN_COUNT - 1) / SMALL_HISTOGRAM_MAX_BIN_COUNT;
    // if we can finish with at most 2 smallHistogram passes, do it
    // otherwise, switch to largeHistogram after one pass
    if(smallHistogramPassCount > 2) smallHistogramPassCount = 1;

    uint binCountDone = 0;

    for(uint pass = 0; pass < smallHistogramPassCount; pass++) {
        const dim3 blockCount(partialHistogramCount, histogramCount);

        uint binCountThisPass = min(binCountDone + SMALL_HISTOGRAM_MAX_BIN_COUNT, binCount) - binCountDone;

        if(binCountThisPass > 64) {
            smallHistogramKernel<T, 128><<<blockCount, SMALL_HISTOGRAM_THREADBLOCK_SIZE>>>(
                dpPartialHistograms, (data_t**)dppData, dpElemCount, binCountDone, binCountThisPass
            );
        } else if(binCountThisPass > 32) {
            smallHistogramKernel<T, 64><<<blockCount, SMALL_HISTOGRAM_THREADBLOCK_SIZE>>>(
                dpPartialHistograms, (data_t**)dppData, dpElemCount, binCountDone, binCountThisPass
            );
        } else {
            smallHistogramKernel<T, 32><<<blockCount, SMALL_HISTOGRAM_THREADBLOCK_SIZE>>>(
                dpPartialHistograms, (data_t**)dppData, dpElemCount, binCountDone, binCountThisPass
            );
        }
        cudaCheckMsg("smallHistogramKernel() execution failed\n");

        const dim3 blockCountMerge(binCountThisPass, histogramCount);
        mergeHistogramKernel<<<blockCountMerge, MERGE_THREADBLOCK_SIZE>>>(
            dppHistograms, binCountDone, dpPartialHistograms, partialHistogramCount, binCountThisPass
        );
        cudaCheckMsg("mergeHistogramKernel() execution failed\n");

        binCountDone += binCountThisPass;
    }


    uint largeHistogramPassCount = (binCount - binCountDone + LARGE_HISTOGRAM_MAX_BIN_COUNT - 1) / LARGE_HISTOGRAM_MAX_BIN_COUNT;

    for(uint pass = 0; pass < largeHistogramPassCount; pass++) {
        uint partialHistogramCount = (128 + histogramCount - 1) / histogramCount; // arbitrary, but seems to give best performance
        if(partialHistogramCount < 16) {
            // not worth the extra merge pass...
            partialHistogramCount = 1;
        }
        const dim3 blockCount(partialHistogramCount, histogramCount);

        uint binCountThisPass = min(binCountDone + LARGE_HISTOGRAM_MAX_BIN_COUNT, binCount) - binCountDone;

        if(partialHistogramCount == 1) {
            if(binCountThisPass > 1536) {
                largeHistogramKernel2<T, 2048><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dppHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass, binCount
                );
            } else if(binCountThisPass > 1024) {
                largeHistogramKernel2<T, 1536><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dppHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass, binCount
                );
            } else if(binCountThisPass > 768) {
                largeHistogramKernel2<T, 1024><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dppHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass, binCount
                );
            } else if(binCountThisPass > 640) {
                largeHistogramKernel2<T, 768><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dppHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass, binCount
                );
            } else if(binCountThisPass > 512) {
                largeHistogramKernel2<T, 640><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dppHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass, binCount
                );
            } else if(binCountThisPass > 384) {
                largeHistogramKernel2<T, 512><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dppHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass, binCount
                );
            } else if(binCountThisPass > 256) {
                largeHistogramKernel2<T, 384><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dppHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass, binCount
                );
            } else {
                largeHistogramKernel2<T, 256><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dppHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass, binCount
                );
            }
            cudaCheckMsg("largeHistogramKernel2() execution failed\n");
        } else {
            if(binCountThisPass > 1536) {
                largeHistogramKernel1<T, 2048><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dpPartialHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass
                );
            } else if(binCountThisPass > 1024) {
                largeHistogramKernel1<T, 1536><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dpPartialHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass
                );
            } else if(binCountThisPass > 768) {
                largeHistogramKernel1<T, 1024><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dpPartialHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass
                );
            } else if(binCountThisPass > 640) {
                largeHistogramKernel1<T, 768><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dpPartialHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass
                );
            } else if(binCountThisPass > 512) {
                largeHistogramKernel1<T, 640><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dpPartialHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass
                );
            } else if(binCountThisPass > 384) {
                largeHistogramKernel1<T, 512><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dpPartialHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass
                );
            } else if(binCountThisPass > 256) {
                largeHistogramKernel1<T, 384><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dpPartialHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass
                );
            } else {
                largeHistogramKernel1<T, 256><<<blockCount, LARGE_HISTOGRAM_THREADBLOCK_SIZE>>>(
                    dpPartialHistograms, (uint**)dppData, dpElemCount, binCountDone, binCountThisPass
                );
            }
            cudaCheckMsg("largeHistogramKernel1() execution failed\n");

            //TODO mergeHistogramKernel is very inefficient for small partialHistogramCount
            const dim3 blockCountMerge(binCountThisPass, histogramCount);
            mergeHistogramKernel<<<blockCountMerge, MERGE_THREADBLOCK_SIZE>>>(
                dppHistograms, binCountDone, dpPartialHistograms, partialHistogramCount, binCountThisPass
            );
            cudaCheckMsg("mergeHistogramKernel() execution failed\n");
        }

        binCountDone += binCountThisPass;
    }

    pInstance->releaseBuffers(4);

    return true;
}

bool histogram(Instance* pInstance, uint* pdpHistograms[], uint histogramCount, const ushort* pdpData[], const uint* pElemCount, uint binCount)
{
    return histogram<ushort>(pInstance, pdpHistograms, histogramCount, pdpData, pElemCount, binCount);
}

bool histogram(Instance* pInstance, uint* pdpHistograms[], uint histogramCount, const uint* pdpData[], const uint* pElemCount, uint binCount)
{
    return histogram<uint>(pInstance, pdpHistograms, histogramCount, pdpData, pElemCount, binCount);
}

}
