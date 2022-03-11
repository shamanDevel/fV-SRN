#include "memtrace.h"
#include <cudaCompress\RunLength.h>

#include <cassert>
#include <string>

#include <cuda_runtime.h>

#include <cudaCompress/cudaUtil.h>
#include <cudaCompress/util.h>

#include <cudaCompress/InstanceImpl.h>


#include <cudaCompress/scan/scan_app.cui>

#include "RunLengthKernels.cui"


namespace cudaCompress {


size_t runLengthGetRequiredMemory(const Instance* pInstance)
{
    uint streamCountMax = pInstance->m_streamCountMax;
    uint symbolCountMax = pInstance->m_elemCountPerStreamMax;

    size_t sizeDecode = 0;
    // dpValidSymbolIndices
    sizeDecode += getAlignedSize(streamCountMax * symbolCountMax * sizeof(uint), 128);
    // dpUploads
    sizeDecode += getAlignedSize(streamCountMax * (2 * sizeof(Symbol16*) + sizeof(uint)), 128);

    size_t sizeEncode = 0;
    // dpValidSymbolIndices
    sizeEncode += getAlignedSize(symbolCountMax * sizeof(uint), 128);
    // dpOutputIndices
    sizeEncode += getAlignedSize(streamCountMax * (symbolCountMax + 1) * sizeof(uint), 128);
    // dpScanTotal
    sizeEncode += getAlignedSize(streamCountMax * sizeof(uint), 128);

    size_t size = max(sizeEncode, sizeDecode);

    return size;
}

bool runLengthInit(Instance* pInstance)
{
    uint streamCountMax = pInstance->m_streamCountMax;

    cudaSafeCall(cudaMallocHost(&pInstance->RunLength.pReadback, streamCountMax * sizeof(uint)));
    pInstance->RunLength.syncEventsReadback.resize(streamCountMax);
    for(uint stream = 0; stream < streamCountMax; stream++) {
        cudaSafeCall(cudaEventCreateWithFlags(&pInstance->RunLength.syncEventsReadback[stream], cudaEventDisableTiming));
    }

    cudaSafeCall(cudaMallocHost(&pInstance->RunLength.pUpload, streamCountMax * (2 * sizeof(Symbol16*) + sizeof(uint))));
    cudaSafeCall(cudaEventCreateWithFlags(&pInstance->RunLength.syncEventUpload, cudaEventDisableTiming));
    cudaSafeCall(cudaEventRecord(pInstance->RunLength.syncEventUpload));

    return true;
}

bool runLengthShutdown(Instance* pInstance)
{
    cudaSafeCall(cudaEventDestroy(pInstance->RunLength.syncEventUpload));
    pInstance->RunLength.syncEventUpload = 0;

    cudaSafeCall(cudaFreeHost(pInstance->RunLength.pUpload));
    pInstance->RunLength.pUpload = nullptr;

    for(uint stream = 0; stream < pInstance->RunLength.syncEventsReadback.size(); stream++) {
        cudaSafeCall(cudaEventDestroy(pInstance->RunLength.syncEventsReadback[stream]));
    }
    pInstance->RunLength.syncEventsReadback.clear();

    cudaSafeCall(cudaFreeHost(pInstance->RunLength.pReadback));
    pInstance->RunLength.pReadback = nullptr;

    return true;
}

template<typename Symbol>
bool runLengthEncode(Instance* pInstance, Symbol** pdpSymbolsCompact, Symbol** pdpZeroCounts, const Symbol** pdpSymbols, const uint* pSymbolCount, uint streamCount, uint zeroCountMax, uint* pSymbolCountCompact)
{
    assert(streamCount <= pInstance->m_streamCountMax);

    uint streamCountMax = pInstance->m_streamCountMax;
    uint symbolCountMax = pInstance->m_elemCountPerStreamMax;

    size_t outputIndicesStride = getAlignedSize((symbolCountMax + 1) * sizeof(uint), 128) / sizeof(uint);
    uint* dpOutputIndicesAll   = pInstance->getBuffer<uint>(streamCountMax * outputIndicesStride);
    uint* dpValidSymbolIndices = pInstance->getBuffer<uint>(symbolCountMax);
    uint* dpScanTotal          = pInstance->getBuffer<uint>(streamCountMax);

    uint blockSize = 0;
    uint blockCount = 0;

    for(uint stream = 0; stream < streamCount; stream++) {
        assert(pSymbolCount[stream] <= symbolCountMax);

        uint* dpOutputIndices = dpOutputIndicesAll + stream * outputIndicesStride;

        util::CudaScopedTimer timer(pInstance->RunLength.timerEncode);

        timer("Scan Valid Flags");

        // run prefix sum on symbol non-zero flags to get output indices
        //TODO ballot scan!
        scanArray<Symbol, uint, true, FunctorFlagTrue<Symbol, uint>>(dpOutputIndices, pdpSymbols[stream], pSymbolCount[stream] + 1, pInstance->m_pScanPlan, pInstance->m_stream);
        cudaCheckMsg("runLengthEncode: Error in scanArray");

        // last element of outputindices == compact symbol count, start readback
        uint* dpCompactSymbolCount = dpOutputIndices + pSymbolCount[stream];
        cudaSafeCall(cudaMemcpyAsync(pInstance->RunLength.pReadback + stream, dpCompactSymbolCount, sizeof(uint), cudaMemcpyDeviceToHost, pInstance->m_stream));
        cudaSafeCall(cudaEventRecord(pInstance->RunLength.syncEventsReadback[stream], pInstance->m_stream));
    }

    for(uint stream = 0; stream < streamCount; stream++) {
        uint* dpOutputIndices = dpOutputIndicesAll + stream * outputIndicesStride;
        uint* dpCompactSymbolCount = dpOutputIndices + pSymbolCount[stream];

        util::CudaScopedTimer timer(pInstance->RunLength.timerEncode);

        timer("Get Valid Symbol Indices");

        // get indices of valid (non-zero) symbols
        blockSize = 256;
        blockCount = min((pSymbolCount[stream] + blockSize - 1) / blockSize, 256u);
        runLengthEncodeGetValidSymbolIndices<<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpOutputIndices, dpValidSymbolIndices, pSymbolCount[stream]);
        cudaCheckMsg("runLengthEncodeGetValidSymbolIndices execution failed");

        timer("Get # Extra Zeros");

        // compute number of extra zero symbols to insert in order to respect zeroCountMax
        // choose blockCount based on original (non-compact) symbol count, so we can wait a bit longer before syncing on the download
        blockSize = 256;
        blockCount = min((pSymbolCount/*Compact*/[stream] + blockSize - 1) / blockSize, 256u);

        runLengthEncodeExtraZeroSymbolCountsKernel<<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpValidSymbolIndices, dpOutputIndices, dpCompactSymbolCount, zeroCountMax);
        cudaCheckMsg("runLengthEncodeExtraZeroSymbolCountsKernel execution failed");

        timer("Sync Readback");

        // wait for download of compacted symbol count - need it for the next scan
        cudaSafeCall(cudaEventSynchronize(pInstance->RunLength.syncEventsReadback[stream]));
        pSymbolCountCompact[stream] = pInstance->RunLength.pReadback[stream];

        timer("Scan # Extra Zeros");

        // run prefix sum on extra zero symbol counts to get output offsets
        scanArray<uint, uint, true>(dpOutputIndices, dpOutputIndices, pSymbolCountCompact[stream] + 1, pInstance->m_pScanPlan, pInstance->m_stream);
        cudaCheckMsg("runLengthEncode: Error in scanArray");

        timer("Download # Extra Zeros");

        // last write offset == total number of extra zeroes to be inserted
        cudaSafeCall(cudaMemcpyAsync(dpScanTotal + stream, dpOutputIndices + pSymbolCountCompact[stream], sizeof(uint), cudaMemcpyDeviceToDevice, pInstance->m_stream));
        // if this was the last stream, start readback to cpu
        if(stream == streamCount - 1) {
            cudaSafeCall(cudaMemcpyAsync(pInstance->RunLength.pReadback, dpScanTotal, streamCount * sizeof(uint), cudaMemcpyDeviceToHost, pInstance->m_stream));
            cudaSafeCall(cudaEventRecord(pInstance->RunLength.syncEventsReadback[0], pInstance->m_stream));
        }

        // if there are no non-zero symbols, we can bail out here
        if(pSymbolCountCompact[stream] == 0) {
            continue;
        }

        timer("Compact");

        // copy non-zero symbols to output, pad with extra zero symbols where necessary
        blockSize = 256;
        blockCount = min((pSymbolCountCompact[stream] + blockSize - 1) / blockSize, 256u);

        runLengthEncodeCompactKernel<Symbol><<<blockCount, blockSize, 0, pInstance->m_stream>>>(pdpSymbols[stream], dpValidSymbolIndices, dpOutputIndices, pdpSymbolsCompact[stream], pdpZeroCounts[stream], pSymbolCountCompact[stream], zeroCountMax);
        cudaCheckMsg("runLengthEncodeCompactKernel execution failed");
    }

    // add extra zeros to compacted symbol count
    cudaSafeCall(cudaEventSynchronize(pInstance->RunLength.syncEventsReadback[0]));
    for(uint stream = 0; stream < streamCount; stream++) {
        pSymbolCountCompact[stream] += pInstance->RunLength.pReadback[stream];
    }

    pInstance->releaseBuffers(3);

    return true;
}

template<typename Symbol>
bool runLengthDecode(Instance* pInstance, const Symbol** pdpSymbolsCompact, const Symbol** pdpZeroCounts, const uint* pSymbolCountCompact, Symbol** pdpSymbols, const uint* pSymbolCount, uint streamCount)
{
    assert(streamCount <= pInstance->m_streamCountMax);

    uint symbolCountMax = pInstance->m_elemCountPerStreamMax;

    uint* dpValidSymbolIndices = pInstance->getBuffer<uint>(streamCount * symbolCountMax);
    byte* dpUploads = pInstance->getBuffer<byte>(streamCount * (sizeof(Symbol*) + sizeof(uint)));

    util::CudaScopedTimer timer(pInstance->RunLength.timerDecode);

    {
        timer("Scan Zero Counts");

        for(uint i = 0; i < streamCount; i++) {
            // if there are no symbols, we're done here
            if(pSymbolCountCompact[i] == 0) {
                continue;
            }

            // run prefix sum on zero counts to get valid symbol indices
            assert(pSymbolCountCompact[i] < symbolCountMax);
            scanArray<Symbol, uint, false>(dpValidSymbolIndices + i * symbolCountMax, pdpZeroCounts[i], pSymbolCountCompact[i], pInstance->m_pScanPlan, pInstance->m_stream);
            cudaCheckMsg("runLengthDecode: Error in scanArray");
        }
    }


    uint symbolCountCompactMax = 0;
    for(uint i = 0; i < streamCount; i++) {
        symbolCountCompactMax = max(symbolCountCompactMax, pSymbolCountCompact[i]);
    }

    if(symbolCountCompactMax > 0)
    {
        timer("Scatter Symbols");

        // upload symbol stream pointers and compact symbol counts
        Symbol** ppSymbolsCompactUpload = (Symbol**)pInstance->RunLength.pUpload;
        Symbol** ppSymbolsUpload        = (Symbol**)(ppSymbolsCompactUpload + streamCount);
        uint* pSymbolCountCompactUpload = (uint*)   (ppSymbolsUpload        + streamCount);
        cudaSafeCall(cudaEventSynchronize(pInstance->RunLength.syncEventUpload));

        memcpy(ppSymbolsCompactUpload,    pdpSymbolsCompact,   streamCount * sizeof(Symbol*));
        memcpy(ppSymbolsUpload,           pdpSymbols,          streamCount * sizeof(Symbol*));
        memcpy(pSymbolCountCompactUpload, pSymbolCountCompact, streamCount * sizeof(uint));
        cudaSafeCall(cudaMemcpyAsync(dpUploads, pInstance->RunLength.pUpload, streamCount * (2 * sizeof(Symbol*) + sizeof(uint)), cudaMemcpyHostToDevice, pInstance->m_stream));
        cudaSafeCall(cudaEventRecord(pInstance->RunLength.syncEventUpload, pInstance->m_stream));

        // expand symbol stream - scattered write of non-zero symbols
        Symbol** dppSymbolsCompact = (Symbol**)dpUploads;
        Symbol** dppSymbols        = (Symbol**)(dppSymbolsCompact + streamCount);
        uint* dpSymbolCountCompact = (uint*)   (dppSymbols        + streamCount);

        uint blockSize = 256;
        dim3 blockCount(min((symbolCountCompactMax + blockSize - 1) / blockSize, 256u), streamCount);

        runLengthDecodeMultiScatterKernel<<<blockCount, blockSize, 0, pInstance->m_stream>>>((const Symbol**)dppSymbolsCompact, dpValidSymbolIndices, symbolCountMax, dpSymbolCountCompact, dppSymbols);
        cudaCheckMsg("runLengthDecodeMultiScatterKernel execution failed");
    }

    pInstance->releaseBuffers(2);

    return true;
}

template<typename Symbol>
bool runLengthDecode(Instance* pInstance, const Symbol* dpSymbolsCompact, const Symbol* dpZeroCounts, const uint* pSymbolCountCompact, uint stride, Symbol** pdpSymbols, uint symbolCount, uint streamCount)
{
    assert(streamCount <= pInstance->m_streamCountMax);
    assert(stride <= pInstance->m_elemCountPerStreamMax);

    //TODO make version of scanArray that takes separate input and output stride, and then alloc only streamCount * symbolCount here
    uint* dpValidSymbolIndices = pInstance->getBuffer<uint>(streamCount * stride);
    byte* dpUploads = pInstance->getBuffer<byte>(streamCount * (sizeof(Symbol*) + sizeof(uint)));

    util::CudaScopedTimer timer(pInstance->RunLength.timerDecode);

    uint symbolCountCompactMax = 0;
    for(uint i = 0; i < streamCount; i++) {
        symbolCountCompactMax = max(symbolCountCompactMax, pSymbolCountCompact[i]);
    }

    if(symbolCountCompactMax > 0) {
        timer("Scan Zero Counts");

        // run prefix sum on zero counts to get valid symbol indices
        // combine scans below cutoff into multi-row scans
        const uint cutoff = 64 * 1024; // chosen quite arbitrarily; TODO: benchmark scanArray...
        for(uint streamStart = 0; streamStart < streamCount; ) {
            uint elemCount = pSymbolCountCompact[streamStart];
            if(elemCount == 0) { streamStart++; continue; }

            uint streamEnd = streamStart + 1;
            if(elemCount <= cutoff) {
                while(streamEnd < streamCount && pSymbolCountCompact[streamEnd] <= cutoff) {
                    elemCount = max(elemCount, pSymbolCountCompact[streamEnd]);
                    streamEnd++;
                }
            }
            if(elemCount > 0) {
                uint offset = streamStart * stride;
                scanArray<Symbol, uint, false>(dpValidSymbolIndices + offset, dpZeroCounts + offset, elemCount, streamEnd - streamStart, stride, pInstance->m_pScanPlan, pInstance->m_stream);
                cudaCheckMsg("runLengthDecode: Error in scanArray");
            }

            streamStart = streamEnd;
        }
        //// simple version that just scans all streams at once
        //scanArray<Symbol, uint, false>(dpValidSymbolIndices, dpZeroCounts, symbolCountCompactMax, streamCount, stride, pInstance->m_pScanPlan, pInstance->m_stream);
        //cudaCheckMsg("runLengthDecode: Error in scanArray");

        timer("Scatter Symbols");

        // upload symbol stream pointers and compact symbol counts
        Symbol** ppSymbolsUpload        = (Symbol**)pInstance->RunLength.pUpload;
        uint* pSymbolCountCompactUpload = (uint*)(ppSymbolsUpload + streamCount);
        cudaSafeCall(cudaEventSynchronize(pInstance->RunLength.syncEventUpload));
        memcpy(ppSymbolsUpload,           pdpSymbols,          streamCount * sizeof(Symbol*));
        memcpy(pSymbolCountCompactUpload, pSymbolCountCompact, streamCount * sizeof(uint));
        cudaSafeCall(cudaMemcpyAsync(dpUploads, pInstance->RunLength.pUpload, streamCount * (sizeof(Symbol*) + sizeof(uint)), cudaMemcpyHostToDevice, pInstance->m_stream));
        cudaSafeCall(cudaEventRecord(pInstance->RunLength.syncEventUpload, pInstance->m_stream));

        // expand symbol stream - scattered write of non-zero symbols
        Symbol** dppSymbols = (Symbol**)dpUploads;
        uint* dpSymbolCountCompact = (uint*)(dppSymbols + streamCount);

        uint blockSize = 256;
        dim3 blockCount(min((symbolCountCompactMax + blockSize - 1) / blockSize, 256u), streamCount);

        runLengthDecodeMultiScatterKernel<<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpSymbolsCompact, stride, dpValidSymbolIndices, stride, dpSymbolCountCompact, dppSymbols);
        cudaCheckMsg("runLengthDecodeMultiScatterKernel execution failed");
    }

    pInstance->releaseBuffers(2);

    return true;
}



bool runLengthEncode(Instance* pInstance, Symbol16** pdpSymbolsCompact, Symbol16** pdpZeroCounts, const Symbol16** pdpSymbols, const uint* pSymbolCount, uint streamCount, uint zeroCountMax, uint* pSymbolCountCompact)
{
    return runLengthEncode<Symbol16>(pInstance, pdpSymbolsCompact, pdpZeroCounts, pdpSymbols, pSymbolCount, streamCount, zeroCountMax, pSymbolCountCompact);
}

bool runLengthDecode(Instance* pInstance, const Symbol16** pdpSymbolsCompact, const Symbol16** pdpZeroCounts, const uint* pSymbolCountCompact, Symbol16** pdpSymbols, const uint* pSymbolCount, uint streamCount)
{
    return runLengthDecode<Symbol16>(pInstance, pdpSymbolsCompact, pdpZeroCounts, pSymbolCountCompact, pdpSymbols, pSymbolCount, streamCount);
}

bool runLengthDecode(Instance* pInstance, const Symbol16* dpSymbolsCompact, const Symbol16* dpZeroCounts, const uint* pSymbolCountCompact, uint stride, Symbol16** pdpSymbols, uint symbolCount, uint streamCount)
{
    return runLengthDecode<Symbol16>(pInstance, dpSymbolsCompact, dpZeroCounts, pSymbolCountCompact, stride, pdpSymbols, symbolCount, streamCount);
}


bool runLengthEncode(Instance* pInstance, Symbol32** pdpSymbolsCompact, Symbol32** pdpZeroCounts, const Symbol32** pdpSymbols, const uint* pSymbolCount, uint streamCount, uint zeroCountMax, uint* pSymbolCountCompact)
{
    return runLengthEncode<Symbol32>(pInstance, pdpSymbolsCompact, pdpZeroCounts, pdpSymbols, pSymbolCount, streamCount, zeroCountMax, pSymbolCountCompact);
}

bool runLengthDecode(Instance* pInstance, const Symbol32** pdpSymbolsCompact, const Symbol32** pdpZeroCounts, const uint* pSymbolCountCompact, Symbol32** pdpSymbols, const uint* pSymbolCount, uint streamCount)
{
    return runLengthDecode<Symbol32>(pInstance, pdpSymbolsCompact, pdpZeroCounts, pSymbolCountCompact, pdpSymbols, pSymbolCount, streamCount);
}

bool runLengthDecode(Instance* pInstance, const Symbol32* dpSymbolsCompact, const Symbol32* dpZeroCounts, const uint* pSymbolCountCompact, uint stride, Symbol32** pdpSymbols, uint symbolCount, uint streamCount)
{
    return runLengthDecode<Symbol32>(pInstance, dpSymbolsCompact, dpZeroCounts, pSymbolCountCompact, stride, pdpSymbols, symbolCount, streamCount);
}


}
