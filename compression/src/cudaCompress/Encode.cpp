#include "memtrace.h"
#include <cudaCompress/Encode.h>

#include <cassert>

#include <cuda_runtime.h>

#include <cudaCompress/cudaUtil.h>
#include <cudaCompress/util.h>

#include <cudaCompress/InstanceImpl.h>
#include <cudaCompress/Histogram.h>
#include <cudaCompress/Huffman.h>
#include <cudaCompress/HuffmanTable.h>
#include <cudaCompress/PackInc.h>
#include <cudaCompress/RunLength.h>


namespace cudaCompress {

//static uint g_bitsSymbolEncodeTables = 0;
//static uint g_bitsSymbolCodewords = 0;
//static uint g_bitsSymbolOffsets = 0;
//static uint g_bitsZeroCountEncodeTables = 0;
//static uint g_bitsZeroCountCodewords = 0;
//static uint g_bitsZeroCountOffsets = 0;
//static uint g_bitsTotal = 0;
//static uint g_totalEncodedCount = 0;


static uint getNumUintsForBits(uint bitsize)
{
    uint bitsPerUint = sizeof(uint) * 8;
    return (bitsize + bitsPerUint - 1) / bitsPerUint;
}

static uint getNumUintsForBytes(uint bytesize)
{
    uint bytesPerUint = sizeof(uint);
    return (bytesize + bytesPerUint - 1) / bytesPerUint;
}

static uint getNumOffsets(uint symbolCount, uint codingBlockSize)
{
    return (symbolCount + codingBlockSize - 1) / codingBlockSize;
}


size_t encodeGetRequiredMemory(const Instance* pInstance)
{
    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);
    uint symbolSize = longSymbols ? sizeof(Symbol32) : sizeof(Symbol16);

    uint streamCountMax = pInstance->m_streamCountMax;
    uint symbolCountPerStreamMax = pInstance->m_elemCountPerStreamMax;

    uint symbolStreamMaxBytes = symbolCountPerStreamMax * symbolSize;
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStreamMax, pInstance->m_codingBlockSize) * sizeof(uint);

    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    size_t size = 0;

    // streamInfo.dpSymbolStream - compacted symbols
    size += streamCountMax * getAlignedSize(symbolStreamMaxBytes, 128);
    // streamInfo.dpSymbolStream - zero counts
    size += streamCountMax * getAlignedSize(symbolStreamMaxBytes, 128);

    // streamInfo.dpCodewordStream
    size += streamCountMax * getAlignedSize(symbolStreamMaxBytes, 128);
    // streamInfo.dpOffsets
    size += streamCountMax * getAlignedSize(offsetStreamMaxBytes, 128);

    // streamInfo.dpEncodeCodeWords
    size += streamCountMax * getAlignedSize(distinctSymbolCountMax * sizeof(uint), 128);
    // streamInfo.dpEncodeCodeWordLengths
    size += streamCountMax * getAlignedSize(distinctSymbolCountMax * sizeof(uint), 128);

    // streamInfo.dpDecodeTable
    size += streamCountMax * getAlignedSize(HuffmanDecodeTable::computeMaxGPUSize(pInstance), 128);

    return size;
}

bool encodeInit(Instance* pInstance)
{
    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);
    uint symbolSize = longSymbols ? sizeof(Symbol32) : sizeof(Symbol16);

    uint streamCountMax = pInstance->m_streamCountMax;
    uint symbolCountPerStreamMax = pInstance->m_elemCountPerStreamMax;


    uint symbolStreamMaxBytes = symbolCountPerStreamMax * symbolSize;
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStreamMax, pInstance->m_codingBlockSize) * sizeof(uint);
    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    uint symbolStreamMaxBytesAligned = (uint)getAlignedSize(symbolStreamMaxBytes, 128);
    uint offsetStreamMaxBytesAligned = (uint)getAlignedSize(offsetStreamMaxBytes, 128);

    uint symbolStreamMaxElemsAligned = uint(symbolStreamMaxBytesAligned / sizeof(uint));
    uint offsetStreamMaxElemsAligned = uint(offsetStreamMaxBytesAligned / sizeof(uint));

    uint distinctSymbolCountMaxBytesAligned = (uint)getAlignedSize(distinctSymbolCountMax * sizeof(uint), 128);


    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pCodewordBuffer, 2 * streamCountMax * symbolStreamMaxBytesAligned));
    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pOffsetBuffer,   2 * streamCountMax * offsetStreamMaxBytesAligned));

    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pEncodeCodewords,       2 * streamCountMax * distinctSymbolCountMaxBytesAligned));
    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pEncodeCodewordLengths, 2 * streamCountMax * distinctSymbolCountMaxBytesAligned));

    cudaSafeCall(cudaMallocHost(&pInstance->Encode.pEncodeSymbolStreamInfos, 2 * streamCountMax * sizeof(HuffmanGPUStreamInfo)));


    pInstance->Encode.symbolEncodeTables.reserve(streamCountMax);
    pInstance->Encode.zeroCountEncodeTables.reserve(streamCountMax);
    for(uint i = 0; i < streamCountMax; i++) {
        pInstance->Encode.symbolEncodeTables.push_back(HuffmanEncodeTable(pInstance));
        pInstance->Encode.zeroCountEncodeTables.push_back(HuffmanEncodeTable(pInstance));
    }

    cudaSafeCall(cudaEventCreate(&pInstance->Encode.encodeFinishedEvent, cudaEventDisableTiming));
    cudaSafeCall(cudaEventRecord(pInstance->Encode.encodeFinishedEvent));


    for(int i = 0; i < pInstance->Encode.ms_decodeResourcesCount; i++) {
        Instance::EncodeResources::DecodeResources& res = pInstance->Encode.Decode[i];

        res.symbolDecodeTables.reserve(streamCountMax);
        res.zeroCountDecodeTables.reserve(streamCountMax);
        for(uint i = 0; i < streamCountMax; i++) {
            res.symbolDecodeTables.push_back(HuffmanDecodeTable(pInstance));
            res.zeroCountDecodeTables.push_back(HuffmanDecodeTable(pInstance));
        }

        size_t decodeTableSizeMax = getAlignedSize(HuffmanDecodeTable::computeMaxGPUSize(pInstance), 128);
        cudaSafeCall(cudaMallocHost(&res.pSymbolDecodeTablesBuffer, streamCountMax * decodeTableSizeMax));
        cudaSafeCall(cudaMallocHost(&res.pZeroCountDecodeTablesBuffer, streamCountMax * decodeTableSizeMax));

        // we read from these in huffmanDecode, so don't alloc as write combined
        cudaSafeCall(cudaMallocHost(&res.pSymbolStreamInfos, streamCountMax * sizeof(HuffmanGPUStreamInfo)));
        cudaSafeCall(cudaMallocHost(&res.pZeroCountStreamInfos, streamCountMax * sizeof(HuffmanGPUStreamInfo)));

        cudaSafeCall(cudaEventCreate(&res.syncEvent, cudaEventDisableTiming));
        cudaSafeCall(cudaEventRecord(res.syncEvent));

        cudaSafeCall(cudaMallocHost(&res.pCodewordStreams, streamCountMax * symbolStreamMaxBytes));
        cudaSafeCall(cudaMallocHost(&res.pSymbolOffsets, streamCountMax * offsetStreamMaxBytes));
        cudaSafeCall(cudaMallocHost(&res.pZeroCountOffsets, streamCountMax * offsetStreamMaxBytes));
    }

    return true;
}

bool encodeShutdown(Instance* pInstance)
{
    for(int i = 0; i < pInstance->Encode.ms_decodeResourcesCount; i++) {
        Instance::EncodeResources::DecodeResources& res = pInstance->Encode.Decode[i];

        cudaSafeCall(cudaFreeHost(res.pZeroCountOffsets));
        res.pZeroCountOffsets = nullptr;
        cudaSafeCall(cudaFreeHost(res.pSymbolOffsets));
        res.pSymbolOffsets = nullptr;
        cudaSafeCall(cudaFreeHost(res.pCodewordStreams));
        res.pCodewordStreams = nullptr;

        cudaSafeCall(cudaEventDestroy(res.syncEvent));
        res.syncEvent = 0;

        cudaSafeCall(cudaFreeHost(res.pZeroCountStreamInfos));
        res.pZeroCountStreamInfos = nullptr;
        cudaSafeCall(cudaFreeHost(res.pSymbolStreamInfos));
        res.pSymbolStreamInfos = nullptr;

        cudaSafeCall(cudaFreeHost(res.pZeroCountDecodeTablesBuffer));
        res.pZeroCountDecodeTablesBuffer = nullptr;
        cudaSafeCall(cudaFreeHost(res.pSymbolDecodeTablesBuffer));
        res.pSymbolDecodeTablesBuffer = nullptr;

        res.zeroCountDecodeTables.clear();
        res.symbolDecodeTables.clear();
    }


    cudaSafeCall(cudaEventDestroy(pInstance->Encode.encodeFinishedEvent));
    pInstance->Encode.encodeFinishedEvent = 0;

    pInstance->Encode.zeroCountEncodeTables.clear();
    pInstance->Encode.symbolEncodeTables.clear();

    cudaSafeCall(cudaFreeHost(pInstance->Encode.pEncodeSymbolStreamInfos));
    pInstance->Encode.pEncodeSymbolStreamInfos = nullptr;

    cudaSafeCall(cudaFreeHost(pInstance->Encode.pEncodeCodewords));
    pInstance->Encode.pEncodeCodewords = nullptr;
    cudaSafeCall(cudaFreeHost(pInstance->Encode.pEncodeCodewordLengths));
    pInstance->Encode.pEncodeCodewordLengths = nullptr;

    cudaSafeCall(cudaFreeHost(pInstance->Encode.pOffsetBuffer));
    pInstance->Encode.pOffsetBuffer = nullptr;
    cudaSafeCall(cudaFreeHost(pInstance->Encode.pCodewordBuffer));
    pInstance->Encode.pCodewordBuffer = nullptr;

    return true;
}

template<typename Symbol>
bool encodeRLHuff(Instance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, const Symbol* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    uint symbolStreamMaxBytes = symbolCountPerStream * sizeof(Symbol);
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize) * sizeof(uint);

    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    uint symbolStreamMaxBytesAligned = (uint)getAlignedSize(symbolStreamMaxBytes, 128);
    uint offsetStreamMaxBytesAligned = (uint)getAlignedSize(offsetStreamMaxBytes, 128);
    uint distinctSymbolCountMaxAligned = (uint)getAlignedSize(distinctSymbolCountMax, 128 / sizeof(uint));

    uint symbolStreamMaxElemsAligned = uint(symbolStreamMaxBytesAligned / sizeof(uint));
    uint offsetStreamMaxElemsAligned = uint(offsetStreamMaxBytesAligned / sizeof(uint));

    // get GPU buffers from pInstance
    std::vector<Symbol*> pdpSymbolStreamsCompacted(streamCount);
    std::vector<Symbol*> pdpZeroCounts(streamCount);
    std::vector<HuffmanGPUStreamInfo> pStreamInfos(streamCount);
    uint* dpCodewordStreams       = pInstance->getBuffer<uint>(streamCount * symbolStreamMaxElemsAligned);
    uint* dpOffsets               = pInstance->getBuffer<uint>(streamCount * offsetStreamMaxElemsAligned);
    uint* dpEncodeCodewords       = pInstance->getBuffer<uint>(streamCount * distinctSymbolCountMaxAligned);
    uint* dpEncodeCodewordLengths = pInstance->getBuffer<uint>(streamCount * distinctSymbolCountMaxAligned);
    for(uint block = 0; block < streamCount; block++) {
        pdpSymbolStreamsCompacted[block] = (Symbol*)pInstance->getBuffer<Symbol>(symbolCountPerStream);
        pdpZeroCounts[block]             = (Symbol*)pInstance->getBuffer<Symbol>(symbolCountPerStream);

        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpCodewordStream = dpCodewordStreams + block * symbolStreamMaxElemsAligned;
        streamInfo.dpOffsets        = dpOffsets         + block * offsetStreamMaxElemsAligned;

        // dpEncodeCodewords and dpEncodeCodewordLengths will be filled later
    }

    cudaSafeCall(cudaMemsetAsync(dpCodewordStreams, 0, streamCount * symbolStreamMaxBytesAligned, pInstance->m_stream));

    cudaSafeCall(cudaEventSynchronize(pInstance->Encode.encodeFinishedEvent));

    util::CudaScopedTimer timerLow(pInstance->Encode.timerEncodeLowDetail);
    util::CudaScopedTimer timerHigh(pInstance->Encode.timerEncodeHighDetail);

    timerLow("Run Length Encode");

    // run length encode
    std::vector<uint> symbolCountsPerBlock(streamCount, symbolCountPerStream);
    std::vector<uint> symbolCountsCompact(streamCount);
    runLengthEncode(pInstance, pdpSymbolStreamsCompacted.data(), pdpZeroCounts.data(), (const Symbol**)pdpSymbolStreams, symbolCountsPerBlock.data(), streamCount, ZERO_COUNT_MAX, symbolCountsCompact.data());

    timerLow("Huffman Encode Symbols");

    timerHigh("Symbols:    Design Huffman Tables");

    for(uint block = 0; block < streamCount; block++) {
        // padding for histogram (which wants the element count to be a multiple of 8)
        histogramPadData(pInstance, pdpSymbolStreamsCompacted[block], symbolCountsCompact[block]);
        histogramPadData(pInstance, pdpZeroCounts[block],             symbolCountsCompact[block]);
    }

    // 1. compacted symbols
    // build encode tables
    std::vector<HuffmanEncodeTable>& symbolEncodeTables = pInstance->Encode.symbolEncodeTables;
    if(!HuffmanEncodeTable::design(pInstance, symbolEncodeTables.data(), streamCount, (const Symbol**)pdpSymbolStreamsCompacted.data(), symbolCountsCompact.data())) {
        pInstance->releaseBuffers(4 + 2 * streamCount);
        return false;
    }

    timerHigh("Symbols:    Upload Huffman Tables");

    // fill stream infos
    uint* dpEncodeCodewordsNext = dpEncodeCodewords;
    uint* dpEncodeCodewordLengthsNext = dpEncodeCodewordLengths;
    uint* pEncodeCodewordsNext = pInstance->Encode.pEncodeCodewords;
    uint* pEncodeCodewordLengthsNext = pInstance->Encode.pEncodeCodewordLengths;
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpSymbolStream = (byte*)pdpSymbolStreamsCompacted[block];
        streamInfo.symbolCount = symbolCountsCompact[block];

        streamInfo.dpEncodeCodewords       = dpEncodeCodewordsNext;
        streamInfo.dpEncodeCodewordLengths = dpEncodeCodewordLengthsNext;

        symbolEncodeTables[block].copyToBuffer(pEncodeCodewordsNext, pEncodeCodewordLengthsNext);

        size_t elems = symbolEncodeTables[block].getTableSize();
        pEncodeCodewordsNext        += elems;
        pEncodeCodewordLengthsNext  += elems;
        dpEncodeCodewordsNext       += elems;
        dpEncodeCodewordLengthsNext += elems;
    }

    // upload encode tables
    size_t encodeCodewordElems = pEncodeCodewordsNext - pInstance->Encode.pEncodeCodewords;
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewords,       pInstance->Encode.pEncodeCodewords,       encodeCodewordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewordLengths, pInstance->Encode.pEncodeCodewordLengths, encodeCodewordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    timerHigh("Symbols:    Huffman Encode");

    // encode the symbols
    std::vector<uint> codewordBitsizeSymbols(streamCount);
    huffmanEncode(pInstance, pStreamInfos.data(), streamCount, pInstance->m_codingBlockSize, codewordBitsizeSymbols.data());

    timerHigh("Symbols:    Download");

    // download encoded symbols and offsets
    // (GPU buffers will be used again for the zero counts)
    // for small blocks: download everything in a single memcpy (but more memory traffic)
    //cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pCodewordBuffer, dpCodewordStreams, streamCount * symbolStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));
    // for large blocks: download only getNumUintsForBits(codewordBitsizeSymbols[block]) uints per block
    for(uint block = 0; block < streamCount; block++) {
        uint offset = block * symbolStreamMaxElemsAligned;
        uint numBytes = getNumUintsForBits(codewordBitsizeSymbols[block]) * sizeof(uint);
        cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pCodewordBuffer + offset, dpCodewordStreams + offset, numBytes, cudaMemcpyDeviceToHost, pInstance->m_stream));
    }
    // offsets are small, so always download everything in a single memcpy
    cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pOffsetBuffer, dpOffsets, streamCount * offsetStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));

    // clear codeword stream again for zero counts
    cudaSafeCall(cudaMemsetAsync(dpCodewordStreams, 0, streamCount * symbolStreamMaxBytesAligned, pInstance->m_stream));


    timerLow("Huffman Encode ZeroCounts");

    timerHigh("ZeroCounts: Design Huffman Tables");

    // 2. zero counts
    // build encode tables
    std::vector<HuffmanEncodeTable>& zeroCountEncodeTables = pInstance->Encode.zeroCountEncodeTables;
    if(!HuffmanEncodeTable::design(pInstance, zeroCountEncodeTables.data(), streamCount, (const Symbol**)pdpZeroCounts.data(), symbolCountsCompact.data())) {
        pInstance->releaseBuffers(4 + 2 * streamCount);
        return false;
    }

    timerHigh("ZeroCounts: Upload Huffman Tables");

    // fill stream infos
    cudaSafeCall(cudaDeviceSynchronize()); // sync before overwriting pStreamInfos
    dpEncodeCodewordsNext = dpEncodeCodewords;
    dpEncodeCodewordLengthsNext = dpEncodeCodewordLengths;
    pEncodeCodewordsNext       = pInstance->Encode.pEncodeCodewords       + streamCount * distinctSymbolCountMaxAligned;
    pEncodeCodewordLengthsNext = pInstance->Encode.pEncodeCodewordLengths + streamCount * distinctSymbolCountMaxAligned;
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpSymbolStream = (byte*)pdpZeroCounts[block];
        streamInfo.symbolCount = symbolCountsCompact[block];

        streamInfo.dpEncodeCodewords       = dpEncodeCodewordsNext;
        streamInfo.dpEncodeCodewordLengths = dpEncodeCodewordLengthsNext;

        zeroCountEncodeTables[block].copyToBuffer(pEncodeCodewordsNext, pEncodeCodewordLengthsNext);

        size_t elems = zeroCountEncodeTables[block].getTableSize();
        pEncodeCodewordsNext        += elems;
        pEncodeCodewordLengthsNext  += elems;
        dpEncodeCodewordsNext       += elems;
        dpEncodeCodewordLengthsNext += elems;
    }

    // upload encode tables
    encodeCodewordElems = pEncodeCodewordsNext - (pInstance->Encode.pEncodeCodewords + streamCount * distinctSymbolCountMaxAligned);
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewords,       pInstance->Encode.pEncodeCodewords       + streamCount * distinctSymbolCountMaxAligned, encodeCodewordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewordLengths, pInstance->Encode.pEncodeCodewordLengths + streamCount * distinctSymbolCountMaxAligned, encodeCodewordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    timerHigh("ZeroCounts: Huffman Encode");

    // encode the zero counts
    std::vector<uint> codewordBitsizeZeroCounts(streamCount);
    huffmanEncode(pInstance, pStreamInfos.data(), streamCount, pInstance->m_codingBlockSize, codewordBitsizeZeroCounts.data());

    timerHigh("ZeroCounts: Download");

    // download zero count codeword stream and offsets
    // for small blocks: download everything in a single memcpy (but more memory traffic)
    //cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pCodewordBuffer + streamCount * symbolStreamMaxElemsAligned, dpCodewordStreams, streamCount * symbolStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));
    // for large blocks: download only getNumUintsForBits(codewordBitsizeZeroCounts[block]) uints per block
    uint* pCodewordBufferZeroCounts = pInstance->Encode.pCodewordBuffer + streamCount * symbolStreamMaxElemsAligned;
    for(uint block = 0; block < streamCount; block++) {
        uint offset = block * symbolStreamMaxElemsAligned;
        uint numBytes = getNumUintsForBits(codewordBitsizeZeroCounts[block]) * sizeof(uint);
        cudaSafeCall(cudaMemcpyAsync(pCodewordBufferZeroCounts + offset, dpCodewordStreams + offset, numBytes, cudaMemcpyDeviceToHost, pInstance->m_stream));
    }
    // offsets are small, so always download everything in a single memcpy
    uint* pOffsetBufferZeroCounts = pInstance->Encode.pOffsetBuffer + streamCount * offsetStreamMaxElemsAligned;
    cudaSafeCall(cudaMemcpyAsync(pOffsetBufferZeroCounts, dpOffsets, streamCount * offsetStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));

    cudaSafeCall(cudaDeviceSynchronize());

    timerLow();
    timerHigh();


    // write to bitstream
    //#pragma omp parallel for if(!singleBitStream) TODO: need to check that bitstreams are unique!
    for(int block = 0; block < int(streamCount); block++) {
        BitStream& bitStream = *ppBitStreams[singleBitStream ? 0 : block];
        //uint bitStreamPosStart = bitStream.getBitPosition();
        //uint bitStreamPos = bitStreamPosStart;

        // write compacted symbol count
        bitStream.writeAligned(&symbolCountsCompact[block], 1);

        // 1. compacted symbols
        // write encode table
        symbolEncodeTables[block].writeToBitStream(pInstance, bitStream);

        //g_bitsSymbolEncodeTables += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // write symbol codeword stream
        bitStream.writeAligned(&codewordBitsizeSymbols[block], 1);
        uint codewordUints = getNumUintsForBits(codewordBitsizeSymbols[block]);
        uint* pCodewordBuffer = pInstance->Encode.pCodewordBuffer + block * symbolStreamMaxElemsAligned;
        //cudaSafeCall(cudaEventSynchronize(pInstance->Encode.pSyncEvents[block]));
        bitStream.writeAligned(pCodewordBuffer, codewordUints);

        //g_bitsSymbolCodewords += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // make symbol offsets incremental and write
        uint numOffsets = getNumOffsets(symbolCountsCompact[block], pInstance->m_codingBlockSize);
        uint* pOffsetBuffer = pInstance->Encode.pOffsetBuffer + block * offsetStreamMaxElemsAligned;
        packInc16CPU(pOffsetBuffer, (ushort*)pOffsetBuffer, numOffsets);
        bitStream.writeAligned((ushort*)pOffsetBuffer, numOffsets);

        //g_bitsSymbolOffsets += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();


        // 2. zero counts
        // write encode table
        zeroCountEncodeTables[block].writeToBitStream(pInstance, bitStream);

        //g_bitsZeroCountEncodeTables += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // write zero count codeword stream
        bitStream.writeAligned(&codewordBitsizeZeroCounts[block], 1);
        codewordUints = getNumUintsForBits(codewordBitsizeZeroCounts[block]);
        pCodewordBuffer = pInstance->Encode.pCodewordBuffer + (streamCount + block) * symbolStreamMaxElemsAligned;
        //cudaSafeCall(cudaEventSynchronize(pInstance->Encode.pSyncEvents[streamCount + block]));
        bitStream.writeAligned(pCodewordBuffer, codewordUints);

        //g_bitsZeroCountCodewords += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // make zero count offsets incremental and write
        pOffsetBuffer = pInstance->Encode.pOffsetBuffer + (streamCount + block) * offsetStreamMaxElemsAligned;
        packInc16CPU(pOffsetBuffer, (ushort*)pOffsetBuffer, numOffsets);
        bitStream.writeAligned((ushort*)pOffsetBuffer, numOffsets);

        //g_bitsZeroCountOffsets += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        //g_bitsTotal += bitStreamPos - bitStreamPosStart;
    }
    //g_totalEncodedCount += streamCount * symbolCountPerStream;

    cudaSafeCall(cudaEventRecord(pInstance->Encode.encodeFinishedEvent, pInstance->m_stream));

    pInstance->releaseBuffers(4 + 2 * streamCount);

    return true;
}

template<typename Symbol>
bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    uint pad = WARP_SIZE * pInstance->m_codingBlockSize;
    uint symbolCountPerBlockPadded = (symbolCountPerStream + pad - 1) / pad * pad;

    uint symbolStreamMaxBytes = symbolCountPerStream * sizeof(Symbol);
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize) * sizeof(uint);
    Instance::EncodeResources::DecodeResources& resources = pInstance->Encode.GetDecodeResources();
    cudaSafeCall(cudaEventSynchronize(resources.syncEvent));

    Symbol* dpSymbolStreamCompacted = pInstance->getBuffer<Symbol>(streamCount * symbolCountPerBlockPadded);
    Symbol* dpZeroCounts            = pInstance->getBuffer<Symbol>(streamCount * symbolCountPerBlockPadded);
    uint* dpOffsets = (uint*)pInstance->getBuffer<byte>(streamCount * offsetStreamMaxBytes);
    size_t decodeTableSizeMax = getAlignedSize(HuffmanDecodeTable::computeMaxGPUSize(pInstance), 128);
    byte* dpDecodeTables = pInstance->getBuffer<byte>(streamCount * decodeTableSizeMax);
    for(uint block = 0; block < streamCount; block++) {
        // get GPU buffers
        uint* dpCodewordStream = (uint*)pInstance->getBuffer<byte>(symbolStreamMaxBytes);

        // fill stream infos
        // all buffers except the symbol stream buffers are shared between compacted symbols and zero counts
        HuffmanGPUStreamInfo& streamInfoSymbols = resources.pSymbolStreamInfos[block];
        HuffmanGPUStreamInfo& streamInfoZeroCounts = resources.pZeroCountStreamInfos[block];

        streamInfoSymbols.dpSymbolStream = (byte*)(dpSymbolStreamCompacted + block * symbolCountPerBlockPadded);
        streamInfoZeroCounts.dpSymbolStream = (byte*)(dpZeroCounts + block * symbolCountPerBlockPadded);

        streamInfoSymbols.dpCodewordStream = streamInfoZeroCounts.dpCodewordStream = dpCodewordStream;
        // streamInfo.dpOffsets will be filled later, with pointers into our single dpOffsets buffer (see above)
        // streamInfo.dpDecodeTable will be filled later, with pointers into our single dpDecodeTables buffer (see above)
    }

    util::CudaScopedTimer timerLow(pInstance->Encode.timerDecodeLowDetail);
    util::CudaScopedTimer timerHigh(pInstance->Encode.timerDecodeHighDetail);

    timerLow("Huffman Decode Symbols");

    timerHigh("Symbols:    Upload (+read BitStream)");

    uint* pSymbolOffsetsNext = resources.pSymbolOffsets;
    uint* dpOffsetsNext = dpOffsets;
    // read and upload decode tables, upload codeword streams and offsets, and fill stream info for symbols
    std::vector<HuffmanDecodeTable>& symbolDecodeTables = resources.symbolDecodeTables;
    size_t symbolDecodeTablesBufferOffset = 0;
    std::vector<HuffmanDecodeTable>& zeroCountDecodeTables = resources.zeroCountDecodeTables;
    std::vector<const uint*> pZeroCountCodewordStreams(streamCount);
    std::vector<uint> zeroCountCodewordUintCounts(streamCount);
    std::vector<const uint*> pZeroCountOffsets(streamCount);
    for(uint block = 0; block < streamCount; block++) {
        BitStreamReadOnly& bitStream = *ppBitStreams[singleBitStream ? 0 : block];

        HuffmanGPUStreamInfo& streamInfo = resources.pSymbolStreamInfos[block];

        // read compacted symbol count
        bitStream.readAligned(&streamInfo.symbolCount, 1);
        resources.pZeroCountStreamInfos[block].symbolCount = streamInfo.symbolCount;

        // 1. compacted symbols
        // read symbol decode table
        symbolDecodeTables[block].readFromBitStream(pInstance, bitStream);

        // copy decode table into upload buffer
        symbolDecodeTables[block].copyToBuffer(pInstance, resources.pSymbolDecodeTablesBuffer + symbolDecodeTablesBufferOffset);
        streamInfo.dpDecodeTable = dpDecodeTables + symbolDecodeTablesBufferOffset;
        streamInfo.decodeSymbolTableSize = symbolDecodeTables[block].getSymbolTableSize();
        symbolDecodeTablesBufferOffset += getAlignedSize(symbolDecodeTables[block].computeGPUSize(pInstance), 128);

        // upload symbol codewords
        uint codewordBitsize;
        bitStream.readAligned(&codewordBitsize, 1);
        uint codewordUintCount = getNumUintsForBits(codewordBitsize);
        const uint* pCodewordStream = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
        cudaSafeCall(cudaMemcpyAsync(streamInfo.dpCodewordStream, pCodewordStream, codewordUintCount * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
        bitStream.skipBits(codewordUintCount * sizeof(uint) * 8);

        // get symbol offsets pointer
        uint numOffsets = getNumOffsets(streamInfo.symbolCount, pInstance->m_codingBlockSize);
        bitStream.align<uint>();
        const uint* pOffsets = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
        bitStream.skipAligned<ushort>(numOffsets);

        // make offsets absolute (prefix sum)
        unpackInc16CPU(pSymbolOffsetsNext, (const ushort*)pOffsets, numOffsets);
        pSymbolOffsetsNext += numOffsets;
        streamInfo.dpOffsets = dpOffsetsNext;
        dpOffsetsNext += numOffsets;

        // 2. zero counts
        // read zero count decode table
        zeroCountDecodeTables[block].readFromBitStream(pInstance, bitStream);

        // read zero count codewords pointer
        bitStream.readAligned(&codewordBitsize, 1);
        zeroCountCodewordUintCounts[block] = getNumUintsForBits(codewordBitsize);
        pZeroCountCodewordStreams[block] = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
        bitStream.skipBits(zeroCountCodewordUintCounts[block] * sizeof(uint) * 8);

        // read zero count offsets pointer
        bitStream.align<uint>();
        pZeroCountOffsets[block] = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
        bitStream.skipAligned<ushort>(numOffsets);
    }

    // upload decode tables
    cudaSafeCall(cudaMemcpyAsync(dpDecodeTables, resources.pSymbolDecodeTablesBuffer, symbolDecodeTablesBufferOffset, cudaMemcpyHostToDevice, pInstance->m_stream));

    // upload offsets
    size_t offsetCountTotal = pSymbolOffsetsNext - resources.pSymbolOffsets;
    cudaSafeCall(cudaMemcpyAsync(dpOffsets, resources.pSymbolOffsets, offsetCountTotal * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    timerHigh("Symbols:    Huffman Decode");

    // decode symbols
    huffmanDecode(pInstance, resources.pSymbolStreamInfos, streamCount, pInstance->m_codingBlockSize);

    timerLow("Huffman Decode ZeroCounts");

    timerHigh("ZeroCounts: Upload");

    uint* pZeroCountOffsetsNext = resources.pZeroCountOffsets;
    dpOffsetsNext = dpOffsets;
    size_t zeroCountDecodeTablesBufferOffset = 0;
    // upload decode tables, codeword streams and offsets, and fill stream infos for zero counts
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = resources.pZeroCountStreamInfos[block];

        // copy decode table into upload buffer
        zeroCountDecodeTables[block].copyToBuffer(pInstance, resources.pZeroCountDecodeTablesBuffer + zeroCountDecodeTablesBufferOffset);
        streamInfo.dpDecodeTable = dpDecodeTables + zeroCountDecodeTablesBufferOffset;
        streamInfo.decodeSymbolTableSize = zeroCountDecodeTables[block].getSymbolTableSize();
        zeroCountDecodeTablesBufferOffset += getAlignedSize(zeroCountDecodeTables[block].computeGPUSize(pInstance), 128);

        // upload zero count codewords
        cudaSafeCall(cudaMemcpyAsync(streamInfo.dpCodewordStream, pZeroCountCodewordStreams[block], zeroCountCodewordUintCounts[block] * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

        uint numOffsets = getNumOffsets(streamInfo.symbolCount, pInstance->m_codingBlockSize);
        unpackInc16CPU(pZeroCountOffsetsNext, (const ushort*)pZeroCountOffsets[block], numOffsets);
        pZeroCountOffsetsNext += numOffsets;
        streamInfo.dpOffsets = dpOffsetsNext;
        dpOffsetsNext += numOffsets;
    }

    // upload decode tables
    cudaSafeCall(cudaMemcpyAsync(dpDecodeTables, resources.pZeroCountDecodeTablesBuffer, zeroCountDecodeTablesBufferOffset, cudaMemcpyHostToDevice, pInstance->m_stream));

    // upload offsets
    offsetCountTotal = pZeroCountOffsetsNext - resources.pZeroCountOffsets;
    cudaSafeCall(cudaMemcpyAsync(dpOffsets, resources.pZeroCountOffsets, offsetCountTotal * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    timerHigh("ZeroCounts: Huffman Decode");

    // decode zero counts
    huffmanDecode(pInstance, resources.pZeroCountStreamInfos, streamCount, pInstance->m_codingBlockSize);

    timerHigh();

    timerLow("Run Length Decode");

    // run length decode
    std::vector<uint> symbolCountsCompact(streamCount);
    for(uint block = 0; block < streamCount; block++) {
        symbolCountsCompact[block] = resources.pSymbolStreamInfos[block].symbolCount;
    }
    runLengthDecode(pInstance, dpSymbolStreamCompacted, dpZeroCounts, symbolCountsCompact.data(), symbolCountPerBlockPadded, pdpSymbolStreams, symbolCountPerStream, streamCount);

    timerLow();

    cudaSafeCall(cudaEventRecord(resources.syncEvent, pInstance->m_stream));

    pInstance->releaseBuffers(4 + 1 * streamCount);

    return true;
}

template<typename Symbol>
bool encodeHuff(Instance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, /*const*/ Symbol* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    uint symbolStreamMaxBytes = symbolCountPerStream * sizeof(Symbol);
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize) * sizeof(uint);

    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    uint symbolStreamMaxBytesAligned = (uint)getAlignedSize(symbolStreamMaxBytes, 128);
    uint offsetStreamMaxBytesAligned = (uint)getAlignedSize(offsetStreamMaxBytes, 128);
    uint distinctSymbolCountMaxAligned = (uint)getAlignedSize(distinctSymbolCountMax, 128 / sizeof(uint));

    uint symbolStreamMaxElemsAligned = uint(symbolStreamMaxBytesAligned / sizeof(uint));
    uint offsetStreamMaxElemsAligned = uint(offsetStreamMaxBytesAligned / sizeof(uint));

    // get GPU buffers from pInstance
    std::vector<HuffmanGPUStreamInfo> pStreamInfos(streamCount);
    uint* dpCodewordStreams       = pInstance->getBuffer<uint>(streamCount * symbolStreamMaxElemsAligned);
    uint* dpOffsets               = pInstance->getBuffer<uint>(streamCount * offsetStreamMaxElemsAligned);
    uint* dpEncodeCodewords       = pInstance->getBuffer<uint>(streamCount * distinctSymbolCountMaxAligned);
    uint* dpEncodeCodewordLengths = pInstance->getBuffer<uint>(streamCount * distinctSymbolCountMaxAligned);
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpCodewordStream = dpCodewordStreams + block * symbolStreamMaxElemsAligned;
        streamInfo.dpOffsets        = dpOffsets         + block * offsetStreamMaxElemsAligned;

        // dpEncodeCodewords and dpEncodeCodewordLengths will be filled later
    }

    cudaSafeCall(cudaMemsetAsync(dpCodewordStreams, 0, streamCount * symbolStreamMaxBytesAligned, pInstance->m_stream));

    cudaSafeCall(cudaEventSynchronize(pInstance->Encode.encodeFinishedEvent));

    util::CudaScopedTimer timerLow(pInstance->Encode.timerEncodeLowDetail);
    util::CudaScopedTimer timerHigh(pInstance->Encode.timerEncodeHighDetail);

    timerLow("Huffman Encode Symbols");

    timerHigh("Symbols:    Design Huffman Tables");

    for(uint block = 0; block < streamCount; block++) {
        // padding for histogram (which wants the element count to be a multiple of 8)
        histogramPadData(pInstance, pdpSymbolStreams[block], symbolCountPerStream);
    }

    // build encode tables
    std::vector<HuffmanEncodeTable>& symbolEncodeTables = pInstance->Encode.symbolEncodeTables;
    std::vector<uint> symbolCount(streamCount, symbolCountPerStream);
    if(!HuffmanEncodeTable::design(pInstance, symbolEncodeTables.data(), streamCount, (const Symbol**)pdpSymbolStreams, symbolCount.data())) {
        pInstance->releaseBuffers(4);
        return false;
    }

    timerHigh("Symbols:    Upload Huffman Tables");

    // fill stream infos
    uint* dpEncodeCodewordsNext = dpEncodeCodewords;
    uint* dpEncodeCodewordLengthsNext = dpEncodeCodewordLengths;
    uint* pEncodeCodewordsNext = pInstance->Encode.pEncodeCodewords;
    uint* pEncodeCodewordLengthsNext = pInstance->Encode.pEncodeCodewordLengths;
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = pStreamInfos[block];

        streamInfo.dpSymbolStream = (byte*)pdpSymbolStreams[block];
        streamInfo.symbolCount = symbolCount[block];

        streamInfo.dpEncodeCodewords       = dpEncodeCodewordsNext;
        streamInfo.dpEncodeCodewordLengths = dpEncodeCodewordLengthsNext;

        symbolEncodeTables[block].copyToBuffer(pEncodeCodewordsNext, pEncodeCodewordLengthsNext);

        size_t elems = symbolEncodeTables[block].getTableSize();
        pEncodeCodewordsNext        += elems;
        pEncodeCodewordLengthsNext  += elems;
        dpEncodeCodewordsNext       += elems;
        dpEncodeCodewordLengthsNext += elems;
    }

    // upload encode tables
    size_t encodeCodeWordElems = pEncodeCodewordsNext - pInstance->Encode.pEncodeCodewords;
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewords, pInstance->Encode.pEncodeCodewords, encodeCodeWordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(dpEncodeCodewordLengths, pInstance->Encode.pEncodeCodewordLengths, encodeCodeWordElems * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    timerHigh("Symbols:    Huffman Encode");

    // encode the symbols
    std::vector<uint> codewordBitsizeSymbols(streamCount);
    huffmanEncode(pInstance, pStreamInfos.data(), streamCount, pInstance->m_codingBlockSize, codewordBitsizeSymbols.data());

    timerHigh("Symbols:    Download");

    // download encoded symbols and offsets
    //TODO for large blocks, download only getNumUintsForBits(codewordBitsizeSymbols[block]) uints per block?
    cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pCodewordBuffer, dpCodewordStreams, streamCount * symbolStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(pInstance->Encode.pOffsetBuffer,   dpOffsets,         streamCount * offsetStreamMaxBytesAligned, cudaMemcpyDeviceToHost, pInstance->m_stream));
    cudaSafeCall(cudaDeviceSynchronize());

    timerLow();
    timerHigh();

    // write to bitstream
    //#pragma omp parallel for if(!singleBitStream) TODO: need to check that bitstreams are unique!
    for(int block = 0; block < int(streamCount); block++) {
        BitStream& bitStream = *ppBitStreams[singleBitStream ? 0 : block];
        //uint bitStreamPosStart = bitStream.getBitPosition();
        //uint bitStreamPos = bitStreamPosStart;

        // write encode table
        symbolEncodeTables[block].writeToBitStream(pInstance, bitStream);

        //g_bitsSymbolEncodeTables += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // write codeword stream
        bitStream.writeAligned(&codewordBitsizeSymbols[block], 1);
        uint codewordUints = getNumUintsForBits(codewordBitsizeSymbols[block]);
        uint* pCodewordBuffer = pInstance->Encode.pCodewordBuffer + block * symbolStreamMaxElemsAligned;
        bitStream.writeAligned(pCodewordBuffer, codewordUints);

        //g_bitsSymbolCodewords += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();

        // make offsets incremental and write
        uint numOffsets = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize);
        uint* pOffsetBuffer = pInstance->Encode.pOffsetBuffer + block * offsetStreamMaxElemsAligned;
        packInc16CPU(pOffsetBuffer, (ushort*)pOffsetBuffer, numOffsets);
        bitStream.writeAligned((ushort*)pOffsetBuffer, numOffsets);

        //g_bitsSymbolOffsets += bitStream.getBitPosition() - bitStreamPos;
        //bitStreamPos = bitStream.getBitPosition();


        //g_bitsTotal += bitStreamPos - bitStreamPosStart;
    }
    //g_totalEncodedCount += streamCount * symbolCountPerStream;

    cudaSafeCall(cudaEventRecord(pInstance->Encode.encodeFinishedEvent, pInstance->m_stream));

    pInstance->releaseBuffers(4);

    return true;
}

template<typename Symbol>
bool decodeHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    ScopedProfileSample sample0(pInstance->m_pProfiler, "decodeHuff");

    uint symbolStreamMaxBytes = symbolCountPerStream * sizeof(Symbol);
    uint offsetStreamMaxBytes = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize) * sizeof(uint);

    Instance::EncodeResources::DecodeResources& resources = pInstance->Encode.GetDecodeResources();
    { ScopedProfileSample sample0(pInstance->m_pProfiler, "sync");
        cudaSafeCall(cudaEventSynchronize(resources.syncEvent));
    }

    //TODO ? size_t symbolStreamMaxBytesPadded = getAlignedSize(symbolStreamMaxBytes, 128);
    uint* dpCodewordStreams = (uint*)pInstance->getBuffer<byte>(streamCount * symbolStreamMaxBytes);
    uint* dpOffsets = (uint*)pInstance->getBuffer<byte>(streamCount * offsetStreamMaxBytes);
    size_t decodeTableSizeMax = getAlignedSize(HuffmanDecodeTable::computeMaxGPUSize(pInstance), 128);
    byte* dpDecodeTables = pInstance->getBuffer<byte>(streamCount * decodeTableSizeMax);
    for(uint block = 0; block < streamCount; block++) {
        HuffmanGPUStreamInfo& streamInfo = resources.pSymbolStreamInfos[block];

        streamInfo.dpSymbolStream = (byte*)pdpSymbolStreams[block];
        streamInfo.symbolCount = symbolCountPerStream;

        // streamInfo.dpCodewordStream, dpOffsets, dpDecodeTable will be filled later, with pointers into our contiguous buffers (see above)
    }

    util::CudaScopedTimer timerLow(pInstance->Encode.timerDecodeLowDetail);
    util::CudaScopedTimer timerHigh(pInstance->Encode.timerDecodeHighDetail);

    timerLow("Huffman Decode Symbols");

    timerHigh("Symbols:    Upload (+read BitStream)");

    std::vector<HuffmanDecodeTable>& symbolDecodeTables = resources.symbolDecodeTables;
    //if(!singleBitStream) {
    //    // read decode tables
    //    ScopedProfileSample sample1(pInstance->m_pProfiler, "read decode table");
    //    //#pragma omp parallel for
    //    for(int block = 0; block < int(streamCount); block++) {
    //        BitStreamReadOnly& bitStream = *ppBitStreams[block];
    //        symbolDecodeTables[block].readFromBitStream(pInstance, bitStream);
    //    }
    //}

    size_t symbolDecodeTablesBufferOffset = 0;
    uint* pCodewordStreamsNext = resources.pCodewordStreams;
    uint* dpCodewordStreamsNext = dpCodewordStreams;
    uint* pOffsetsNext = resources.pSymbolOffsets;
    uint* dpOffsetsNext = dpOffsets;
    // read and upload decode tables, upload codeword streams and offsets, and fill stream info
    for(uint block = 0; block < streamCount; block++) {
        BitStreamReadOnly& bitStream = *ppBitStreams[singleBitStream ? 0 : block];

        /*if(singleBitStream)*/ {
            // read decode table
            ScopedProfileSample sample1(pInstance->m_pProfiler, "read decode table");
            symbolDecodeTables[block].readFromBitStream(pInstance, bitStream);
        }

        HuffmanGPUStreamInfo& streamInfo = resources.pSymbolStreamInfos[block];

        // copy decode table into upload buffer
        { ScopedProfileSample sample1(pInstance->m_pProfiler, "copy decode table into upload buffer");
            symbolDecodeTables[block].copyToBuffer(pInstance, resources.pSymbolDecodeTablesBuffer + symbolDecodeTablesBufferOffset);
            streamInfo.dpDecodeTable = dpDecodeTables + symbolDecodeTablesBufferOffset;
            streamInfo.decodeSymbolTableSize = symbolDecodeTables[block].getSymbolTableSize();
            symbolDecodeTablesBufferOffset += getAlignedSize(symbolDecodeTables[block].computeGPUSize(pInstance), 128);
        }

        // copy codewords into pinned buffer
        { ScopedProfileSample sample1(pInstance->m_pProfiler, "copy codewords into pinned buffer");
            uint codewordBitsize;
            bitStream.readAligned(&codewordBitsize, 1);
            uint codewordUints = getNumUintsForBits(codewordBitsize);
            const uint* pCodewordStream = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
            bitStream.skipBits(codewordUints * sizeof(uint) * 8);
            memcpy(pCodewordStreamsNext, pCodewordStream, codewordUints * sizeof(uint));
            streamInfo.dpCodewordStream = dpCodewordStreamsNext;
            //TODO align?
            pCodewordStreamsNext += codewordUints;
            dpCodewordStreamsNext += codewordUints;
        }

        // get offsets pointer
        uint numOffsets = getNumOffsets(symbolCountPerStream, pInstance->m_codingBlockSize);
        bitStream.align<uint>();
        const uint* pOffsets = bitStream.getRaw() + bitStream.getBitPosition() / (sizeof(uint)*8);
        bitStream.skipAligned<ushort>(numOffsets);

        // make offsets absolute (prefix sum)
        { ScopedProfileSample sample1(pInstance->m_pProfiler, "make offsets absolute (prefix sum)");
            unpackInc16CPU(pOffsetsNext, (const ushort*)pOffsets, numOffsets);
            streamInfo.dpOffsets = dpOffsetsNext;
            //TODO align?
            pOffsetsNext += numOffsets;
            dpOffsetsNext += numOffsets;
        }
    }

    // upload decode tables
    cudaSafeCall(cudaMemcpyAsync(dpDecodeTables, resources.pSymbolDecodeTablesBuffer, symbolDecodeTablesBufferOffset, cudaMemcpyHostToDevice, pInstance->m_stream));

    // upload codewords
    size_t codewordUintsTotal = pCodewordStreamsNext - resources.pCodewordStreams;
    cudaSafeCall(cudaMemcpyAsync(dpCodewordStreams, resources.pCodewordStreams, codewordUintsTotal * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    // upload offsets
    size_t offsetCountTotal = pOffsetsNext - resources.pSymbolOffsets;
    cudaSafeCall(cudaMemcpyAsync(dpOffsets, resources.pSymbolOffsets, offsetCountTotal * sizeof(uint), cudaMemcpyHostToDevice, pInstance->m_stream));

    timerHigh("Symbols:    Huffman Decode");

    // decode symbols
    //FIXME huffmanDecode requires the symbol streams to be padded, which we can't guarantee here..
    huffmanDecode(pInstance, resources.pSymbolStreamInfos, streamCount, pInstance->m_codingBlockSize);

    timerLow();
    timerHigh();

    cudaSafeCall(cudaEventRecord(resources.syncEvent, pInstance->m_stream));

    pInstance->releaseBuffers(3);

    return true;
}



bool encodeRLHuff(Instance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeRLHuff<Symbol16>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeRLHuff<Symbol16>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(Instance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeHuff<Symbol16>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeHuff<Symbol16>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeRLHuff(Instance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeRLHuff<Symbol32>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeRLHuff<Symbol32>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(Instance* pInstance, BitStream* ppBitStreams[], bool singleBitStream, /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeHuff<Symbol32>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], bool singleBitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeHuff<Symbol32>(pInstance, ppBitStreams, singleBitStream, pdpSymbolStreams, streamCount, symbolCountPerStream);
}


// INTERFACE FUNCTIONS

// single bitstream for all blocks
bool encodeRLHuff(Instance* pInstance, BitStream& bitStream, const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStream* pBitStream = &bitStream;
    return encodeRLHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly& bitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStreamReadOnly* pBitStream = &bitStream;
    return decodeRLHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(Instance* pInstance, BitStream& bitStream, /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStream* pBitStream = &bitStream;
    return encodeHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(Instance* pInstance, BitStreamReadOnly& bitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStreamReadOnly* pBitStream = &bitStream;
    return decodeHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeRLHuff(Instance* pInstance, BitStream& bitStream, const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStream* pBitStream = &bitStream;
    return encodeRLHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly& bitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStreamReadOnly* pBitStream = &bitStream;
    return decodeRLHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(Instance* pInstance, BitStream& bitStream, /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStream* pBitStream = &bitStream;
    return encodeHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(Instance* pInstance, BitStreamReadOnly& bitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    BitStreamReadOnly* pBitStream = &bitStream;
    return decodeHuff(pInstance, &pBitStream, true, pdpSymbolStreams, streamCount, symbolCountPerStream);
}


// separate bitstream for each block (but may contain duplicates)
bool encodeRLHuff(Instance* pInstance, BitStream* ppBitStreams[], const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeRLHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);

}
bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeRLHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(Instance* pInstance, BitStream* ppBitStreams[], /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeRLHuff(Instance* pInstance, BitStream* ppBitStreams[], const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeRLHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeRLHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}

bool encodeHuff(Instance* pInstance, BitStream* ppBitStreams[], /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return encodeHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}
bool decodeHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerStream)
{
    return decodeHuff(pInstance, ppBitStreams, false, pdpSymbolStreams, streamCount, symbolCountPerStream);
}




//void encodeResetBitCounts()
//{
//    g_bitsSymbolEncodeTables = 0;
//    g_bitsSymbolCodewords = 0;
//    g_bitsSymbolOffsets = 0;
//    g_bitsZeroCountEncodeTables = 0;
//    g_bitsZeroCountCodewords = 0;
//    g_bitsZeroCountOffsets = 0;
//    g_bitsTotal = 0;
//    g_totalEncodedCount = 0;
//}

//static float getBitsPercent(uint bitCount)
//{
//    return 100.0f * float(bitCount) / float(g_bitsTotal);
//}

//void encodePrintBitCounts()
//{
//    float bps = float(g_bitsTotal) / float(g_totalEncodedCount);
//    printf("Encoded %u symbols to %u bits (%.2f bps)\n", g_totalEncodedCount, g_bitsTotal, bps);
//    printf("Symbol Encode Tables    : %9u bits (%5.2f%%)\n", g_bitsSymbolEncodeTables,    getBitsPercent(g_bitsSymbolEncodeTables));
//    printf("Symbol Codewords        : %9u bits (%5.2f%%)\n", g_bitsSymbolCodewords,       getBitsPercent(g_bitsSymbolCodewords));
//    printf("Symbol Offsets          : %9u bits (%5.2f%%)\n", g_bitsSymbolOffsets,         getBitsPercent(g_bitsSymbolOffsets));
//    printf("ZeroCount Encode Tables : %9u bits (%5.2f%%)\n", g_bitsZeroCountEncodeTables, getBitsPercent(g_bitsZeroCountEncodeTables));
//    printf("ZeroCount Codewords     : %9u bits (%5.2f%%)\n", g_bitsZeroCountCodewords,    getBitsPercent(g_bitsZeroCountCodewords));
//    printf("ZeroCount Offsets       : %9u bits (%5.2f%%)\n", g_bitsZeroCountOffsets,      getBitsPercent(g_bitsZeroCountOffsets));
//}

}
