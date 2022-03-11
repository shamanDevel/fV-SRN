#include "memtrace.h"
#include "Huffman.h"

#include <cassert>

#include <cuda_runtime.h>
//#include <thrust/device_ptr.h>
//#include <thrust/scan.h>

#include "cudaUtil.h"
#include "util.h"
#include "InstanceImpl.h"
#include "scan/scan_app.cui"

#include "HuffmanKernels.cui"


namespace cudaCompress {


size_t huffmanGetRequiredMemory(const Instance* pInstance)
{
    uint streamCountMax = pInstance->m_streamCountMax;
    uint symbolCountPerStreamMax = pInstance->m_elemCountPerStreamMax;

    size_t sizeEncode = 0;
    size_t sizeDecode = 0;

    // encode: dpStreamInfos
    sizeEncode += getAlignedSize(sizeof(HuffmanGPUStreamInfo) * streamCountMax, 128);

    // encode: dpScratch
    uint prefixCountMax = getPrefixCount(symbolCountPerStreamMax);
    uint scratchBytes = (uint)getAlignedSize((prefixCountMax + 1) * sizeof(uint), 128);
    sizeEncode += streamCountMax * getAlignedSize(scratchBytes, 128);
    // encode: dppScratch
    sizeEncode += getAlignedSize(streamCountMax * sizeof(uint*), 128);

    // encode: dpScanTotal
    sizeEncode += getAlignedSize(streamCountMax * sizeof(uint), 128);

    // decode: dpStreamInfos
    sizeDecode += getAlignedSize(sizeof(HuffmanGPUStreamInfo) * streamCountMax, 128);

    return max(sizeEncode, sizeDecode);
}

bool huffmanInit(Instance* pInstance)
{
    uint streamCountMax = pInstance->m_streamCountMax;

    cudaSafeCall(cudaMallocHost(&pInstance->Huffman.pReadback, streamCountMax * sizeof(uint)));

    cudaSafeCall(cudaEventCreateWithFlags(&pInstance->Huffman.syncEventReadback, cudaEventDisableTiming));

    return true;
}

bool huffmanShutdown(Instance* pInstance)
{
    cudaSafeCall(cudaEventDestroy(pInstance->Huffman.syncEventReadback));
    pInstance->Huffman.syncEventReadback = 0;

    cudaSafeCall(cudaFreeHost(pInstance->Huffman.pReadback));
    pInstance->Huffman.pReadback = nullptr;

    return true;
}


bool huffmanEncode(Instance* pInstance, const HuffmanGPUStreamInfo* pStreamInfos, uint streamCount, uint codingBlockSize, uint* pCompressedSizeBits)
{
    assert(streamCount <= pInstance->m_streamCountMax);

    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);

    uint prefixCountMax = 0;
    uint offsetCountMax = 0;
    for(uint i = 0; i < streamCount; i++) {
        const HuffmanGPUStreamInfo& streamInfo = pStreamInfos[i];

        assert(streamInfo.symbolCount <= pInstance->m_elemCountPerStreamMax);

        uint prefixCount = getPrefixCount(streamInfo.symbolCount);
        prefixCountMax = max(prefixCountMax, prefixCount);

        uint offsetCount = (streamInfo.symbolCount + codingBlockSize - 1) / codingBlockSize;
        offsetCountMax = max(offsetCountMax, offsetCount);
    }

    HuffmanGPUStreamInfo* dpStreamInfos = pInstance->getBuffer<HuffmanGPUStreamInfo>(streamCount);
    uint scratchElems = (uint)getAlignedSize(prefixCountMax + 1, 128 / sizeof(uint));
    uint* dpScratch = pInstance->getBuffer<uint>(streamCount * scratchElems);
    uint** dppScratch = pInstance->getBuffer<uint*>(streamCount);
    uint* dpScanTotal = pInstance->getBuffer<uint>(streamCount);

    std::vector<uint*> pdpScratch(streamCount);
    for(uint i = 0; i < streamCount; i++) {
        pdpScratch[i] = dpScratch + i * scratchElems;
    }


    util::CudaScopedTimer timer(pInstance->Huffman.timerEncode);

    timer("Upload Info");

    cudaSafeCall(cudaMemcpyAsync(dpStreamInfos, pStreamInfos, sizeof(HuffmanGPUStreamInfo) * streamCount, cudaMemcpyHostToDevice, pInstance->m_stream));
    // note: we don't sync on this upload - we trust that the caller won't overwrite/delete the array...

    cudaSafeCall(cudaMemcpyAsync(dppScratch, pdpScratch.data(), sizeof(uint*) * streamCount, cudaMemcpyHostToDevice, pInstance->m_stream)); //TODO upload buffer?
    // there's a sync in here later on, so this "should" be okay...

    timer("Words to Lengths");
    // get codeword lengths (of COMPACTIFY_ELEM_PER_THREAD consecutive codewords)
    if(prefixCountMax > 0) {
        uint blockSize = WORDS_TO_LENGTH_THREADS_PER_BLOCK;
        dim3 blockCount((prefixCountMax + blockSize - 1) / blockSize, streamCount);

        if(longSymbols) {
            huffmanEncodeWordsToLengthKernel<Symbol32><<<blockCount, blockSize, 0, pInstance->m_stream>>>(dppScratch, dpStreamInfos);
        } else {
            huffmanEncodeWordsToLengthKernel<Symbol16><<<blockCount, blockSize, 0, pInstance->m_stream>>>(dppScratch, dpStreamInfos);
        }
        cudaCheckMsg("huffmanEncodeWordsToLengthKernel execution failed");
    }

    timer("Scan Lengths");
    if(prefixCountMax > 0) {
        // scan codeword lengths to get output indices
        scanArray<uint, uint, true>(dpScratch, dpScratch, prefixCountMax + 1, streamCount, scratchElems, pInstance->m_pScanPlan, pInstance->m_stream);
        cudaCheckMsg("huffmanEncode: Error in scanArray");

        // copy scan totals (= compressed bit sizes) into contiguous buffer for common download
        uint blockSize = min(128u, streamCount);
        uint blockCount = (streamCount + blockSize - 1) / blockSize;
        huffmanEncodeCopyScanTotalsKernel<<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpStreamInfos, streamCount, (const uint**)dppScratch, dpScanTotal);
        cudaCheckMsg("huffmanEncodeCopyScanTotalsKernel execution failed");

        // start readback of compressed size
        cudaSafeCall(cudaMemcpyAsync(pInstance->Huffman.pReadback, dpScanTotal, streamCount * sizeof(uint), cudaMemcpyDeviceToHost, pInstance->m_stream));
        cudaSafeCall(cudaEventRecord(pInstance->Huffman.syncEventReadback, pInstance->m_stream));
    }

    timer("Collect Offsets");
    if(offsetCountMax > 0) {
        uint blockSize = min(128u, offsetCountMax);
        dim3 blockCount((offsetCountMax + blockSize - 1) / blockSize, streamCount);
        huffmanEncodeCollectOffsetsKernel<<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpStreamInfos, (const uint**)dppScratch, codingBlockSize);
        cudaCheckMsg("huffmanEncodeCollectOffsetsKernel execution failed");
    }

    timer("Compactify");
    if(prefixCountMax > 0) {
        uint blockSize = COMPACTIFY_THREADS_PER_BLOCK;
        dim3 blockCount((prefixCountMax + blockSize - 1) / blockSize, streamCount);

        if(longSymbols) {
            huffmanEncodeCompactifyKernel<Symbol32><<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpStreamInfos, (const uint**)dppScratch);
        } else {
            huffmanEncodeCompactifyKernel<Symbol16><<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpStreamInfos, (const uint**)dppScratch);
        }
        cudaCheckMsg("huffmanEncodeCompactifyKernel execution failed");
    }

    timer("Readback Sync");

    if(prefixCountMax > 0) {
        cudaSafeCall(cudaEventSynchronize(pInstance->Huffman.syncEventReadback));
    }
    for(uint i = 0; i < streamCount; i++) {
        const HuffmanGPUStreamInfo& streamInfo = pStreamInfos[i];

        if(streamInfo.symbolCount == 0) {
            pCompressedSizeBits[i] = 0;
        } else {
            pCompressedSizeBits[i] = pInstance->Huffman.pReadback[i];
        }
    }

    timer();

    pInstance->releaseBuffers(4);

    return true;
}

bool huffmanDecode(Instance* pInstance, const HuffmanGPUStreamInfo* pStreamInfos, uint streamCount, uint codingBlockSize)
{
    assert(streamCount <= pInstance->m_streamCountMax);

    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);

    HuffmanGPUStreamInfo* dpStreamInfos = pInstance->getBuffer<HuffmanGPUStreamInfo>(streamCount);

    util::CudaScopedTimer timer(pInstance->Huffman.timerDecode);

    timer("Upload Info");

    // upload stream infos
    cudaSafeCall(cudaMemcpyAsync(dpStreamInfos, pStreamInfos, sizeof(HuffmanGPUStreamInfo) * streamCount, cudaMemcpyHostToDevice, pInstance->m_stream));
    // note: we don't sync on this upload - we trust that the caller won't overwrite/delete the array...

    timer("Decode");

    // get max number of symbols
    uint symbolCountPerStreamMax = 0;
    for(uint i = 0; i < streamCount; i++)
        symbolCountPerStreamMax = max(symbolCountPerStreamMax, pStreamInfos[i].symbolCount);

    if(symbolCountPerStreamMax == 0) {
        pInstance->releaseBuffer();
        return true;
    }

    // launch decode kernel
    uint threadCountPerStream = (symbolCountPerStreamMax + codingBlockSize - 1) / codingBlockSize;
    uint blockSize = min(192u, threadCountPerStream);
    blockSize = max(blockSize, HUFFMAN_LOOKUP_SIZE);
    assert(blockSize >= HUFFMAN_LOOKUP_SIZE);
    dim3 blockCount((threadCountPerStream + blockSize - 1) / blockSize, streamCount);

    if(longSymbols) {
        huffmanDecodeKernel<Symbol32><<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpStreamInfos, codingBlockSize);
    } else {
        huffmanDecodeKernel<Symbol16><<<blockCount, blockSize, 0, pInstance->m_stream>>>(dpStreamInfos, codingBlockSize);
    }
    cudaCheckMsg("huffmanDecodeKernel execution failed");

    timer("Transpose");

    // launch transpose kernel
    dim3 blockSizeTranspose(TRANSPOSE_BLOCKDIM_X, TRANSPOSE_BLOCKDIM_Y);
    dim3 blockCountTranspose((symbolCountPerStreamMax + WARP_SIZE * codingBlockSize - 1) / (WARP_SIZE * codingBlockSize), streamCount);

    if(longSymbols) {
        switch(codingBlockSize) {
            case 32:
                huffmanDecodeTransposeKernel<Symbol32, 32><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                break;
            case 64:
                huffmanDecodeTransposeKernel<Symbol32, 64><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                break;
            case 128:
                huffmanDecodeTransposeKernel<Symbol32, 128><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                break;
            case 256:
                huffmanDecodeTransposeKernel<Symbol32, 256><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                break;
            default:
                assert(false);
        }
    } else {
        switch(codingBlockSize) {
            case 32:
                huffmanDecodeTransposeKernel<Symbol16, 32><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                break;
            case 64:
                huffmanDecodeTransposeKernel<Symbol16, 64><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                break;
            case 128:
                huffmanDecodeTransposeKernel<Symbol16, 128><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                break;
            case 256:
                huffmanDecodeTransposeKernel<Symbol16, 256><<<blockCountTranspose, blockSizeTranspose, 0, pInstance->m_stream>>>(dpStreamInfos);
                break;
            default:
                assert(false);
        }
    }
    cudaCheckMsg("huffmanDecodeTransposeKernel execution failed");

    timer();

    pInstance->releaseBuffer();

    return true;
}

}
