#include "memtrace.h"
#include <cudaCompress/HuffmanTable.h>

#include <cassert>
#include <numeric>
#include <queue>

#include <cuda_runtime.h>

#include <cudaCompress/tools/Operator.h>

#include <cudaCompress/cudaUtil.h>

#include <cudaCompress/InstanceImpl.h>
#include <cudaCompress/Histogram.h>
#include <cudaCompress/HuffmanDesign.h>

#include <cudaCompress/reduce/reduce_app.cui>


//#define PINNED_STORAGE


namespace cudaCompress {

HuffmanDecodeTable::HuffmanDecodeTable(const Instance* pInstance)
    : m_pStorage(nullptr), m_symbolTableSize(0)
{
    size_t storageSize = computeMaxGPUSize(pInstance);

#ifdef PINNED_STORAGE
    cudaSafeCall(cudaMallocHost(&m_pStorage, storageSize, cudaHostAllocWriteCombined));
#else
    m_pStorage = new byte[storageSize];
#endif

    byte* pNext = m_pStorage;
    m_pCodewordFirstIndexPerLength = reinterpret_cast<int*>(pNext); pNext += MAX_CODEWORD_BITS * sizeof(int);
    m_pCodewordMinPerLength        = reinterpret_cast<int*>(pNext); pNext += MAX_CODEWORD_BITS * sizeof(int);
    m_pCodewordMaxPerLength        = reinterpret_cast<int*>(pNext); pNext += MAX_CODEWORD_BITS * sizeof(int);
    m_pSymbolTable                 = reinterpret_cast<byte*>(pNext);

    clear();

    cudaSafeCall(cudaEventCreate(&m_uploadSyncEvent, cudaEventDisableTiming));
    cudaSafeCall(cudaEventRecord(m_uploadSyncEvent));
}

HuffmanDecodeTable::HuffmanDecodeTable(HuffmanDecodeTable&& other)
{
    // copy state from other
    memcpy(this, &other, sizeof(HuffmanDecodeTable));

    // clear other
    other.m_pStorage = nullptr;
    other.m_pCodewordFirstIndexPerLength = nullptr;
    other.m_pCodewordMinPerLength = nullptr;
    other.m_pCodewordMaxPerLength = nullptr;
    other.m_pSymbolTable = nullptr;
    other.m_symbolTableSize = 0;
    other.m_uploadSyncEvent = 0;
}

HuffmanDecodeTable::~HuffmanDecodeTable()
{
    if(m_uploadSyncEvent != 0) {
        cudaSafeCall(cudaEventDestroy(m_uploadSyncEvent));
    }

#ifdef PINNED_STORAGE
    cudaSafeCall(cudaFreeHost(m_pStorage));
#else
    delete[] m_pStorage;
#endif
}

HuffmanDecodeTable& HuffmanDecodeTable::operator=(HuffmanDecodeTable&& other)
{
    if(this == &other)
        return *this;

    // release our own resources
#ifdef PINNED_STORAGE
    cudaSafeCall(cudaFreeHost(m_pStorage));
#else
    delete[] m_pStorage;
#endif

    // copy state from other
    memcpy(this, &other, sizeof(HuffmanDecodeTable));

    // clear other
    other.m_pStorage = nullptr;
    other.m_pCodewordFirstIndexPerLength = nullptr;
    other.m_pCodewordMinPerLength = nullptr;
    other.m_pCodewordMaxPerLength = nullptr;
    other.m_pSymbolTable = nullptr;
    other.m_symbolTableSize = 0;

    return *this;
}

void HuffmanDecodeTable::clear()
{
    m_symbolTableSize = 0;

    for(uint i = 0; i < MAX_CODEWORD_BITS; i++) {
        m_pCodewordFirstIndexPerLength[i] = -1;
    }
    for(uint i = 0; i < MAX_CODEWORD_BITS; i++) {
        m_pCodewordMinPerLength[i] = -1;
    }
    for(uint i = 0; i < MAX_CODEWORD_BITS; i++) {
        m_pCodewordMaxPerLength[i] = -1;
    }
}

void HuffmanDecodeTable::readFromBitStream(const Instance* pInstance, BitStreamReadOnly& bitstream)
{
    clear();

    // read codeword count per length
    std::vector<uint> codewordCountPerLength;
    uint codewordCountPerLengthSize = 0;
    bitstream.readBits(codewordCountPerLengthSize, LOG2_MAX_CODEWORD_BITS);
    codewordCountPerLength.reserve(codewordCountPerLengthSize);
    for(uint i = 0; i < codewordCountPerLengthSize; i++) {
        uint codewordCount = 0;
        bitstream.readBits(codewordCount, pInstance->m_log2HuffmanDistinctSymbolCountMax);
        codewordCountPerLength.push_back(codewordCount);
    }

    // read symbol table
    m_symbolTableSize = 0;
    bitstream.readBits(m_symbolTableSize, pInstance->m_log2HuffmanDistinctSymbolCountMax);
    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);
    uint symbolBits = 0;
    bitstream.readBits(symbolBits, longSymbols ? LOG2_MAX_SYMBOL32_BITS : LOG2_MAX_SYMBOL16_BITS);
    // HACK: if symbolBits was 16, then 0 was written to the bitstream (4 least significant bits)
    if(!longSymbols && symbolBits == 0 && m_symbolTableSize > 1) symbolBits = 16;
    for(uint i = 0; i < m_symbolTableSize; i++) {
        uint symbol = 0;
        bitstream.readBits(symbol, symbolBits);
        if(longSymbols) {
            ((Symbol32*)m_pSymbolTable)[i] = Symbol32(symbol);
        } else {
            ((Symbol16*)m_pSymbolTable)[i] = Symbol16(symbol);
        }
    }

    build(codewordCountPerLength);
}

uint HuffmanDecodeTable::computeMaxGPUSize(const Instance* pInstance)
{
    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);
    uint symbolSize = longSymbols ? sizeof(Symbol32) : sizeof(Symbol16);

    uint size = 0;

    size += 3 * MAX_CODEWORD_BITS * sizeof(int);
    size += distinctSymbolCountMax * symbolSize;

    return size;
}

uint HuffmanDecodeTable::computeGPUSize(const Instance* pInstance) const
{
    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);
    uint symbolSize = longSymbols ? sizeof(Symbol32) : sizeof(Symbol16);

    uint size = 0;

    size += 3 * MAX_CODEWORD_BITS * sizeof(int);
    size += m_symbolTableSize * symbolSize;

    return size;
}

void HuffmanDecodeTable::copyToBuffer(const Instance* pInstance, byte* pTable) const
{
    size_t size = computeGPUSize(pInstance);
    memcpy(pTable, m_pStorage, size);
}

void HuffmanDecodeTable::uploadToGPU(const Instance* pInstance, byte* dpTable) const
{
    size_t size = computeGPUSize(pInstance);
    cudaSafeCall(cudaMemcpy(dpTable, m_pStorage, size, cudaMemcpyHostToDevice));
}

void HuffmanDecodeTable::uploadToGPUAsync(const Instance* pInstance, byte* dpTable) const
{
    size_t size = computeGPUSize(pInstance);
    cudaSafeCall(cudaMemcpyAsync(dpTable, m_pStorage, size, cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaEventRecord(m_uploadSyncEvent, pInstance->m_stream));
}

void HuffmanDecodeTable::syncOnLastAsyncUpload() const
{
    cudaSafeCall(cudaEventSynchronize(m_uploadSyncEvent));
}

void HuffmanDecodeTable::build(const std::vector<uint>& codewordCountPerLength)
{
    if(m_symbolTableSize == 0)
        return;

    // count total number of codewords
    uint codewordCount = 0;
    for(uint i = 0; i < codewordCountPerLength.size(); i++) {
        codewordCount += codewordCountPerLength[i];
    }
    if(codewordCountPerLength.empty()) {
        // this can happen when all symbols are the same -> only a single "codeword" with length 0
        codewordCount++;
    }

    assert(codewordCount == m_symbolTableSize);

    // find codeword lengths
    std::vector<uint> codewordLengths;
    codewordLengths.reserve(codewordCount);
    for(uint i = 0; i < codewordCountPerLength.size(); i++) {
        codewordLengths.insert(codewordLengths.cend(), codewordCountPerLength[i], i + 1);
    }

    // find codewords
    std::vector<int> codewords;
    codewords.reserve(codewordCount);

    codewords.push_back(0);
    for(uint index = 1; index < codewordCount; index++) {
        // new codeword = increment previous codeword
        int codeword = codewords[index-1] + 1;
        // append zero bits as required to reach correct length
        uint lengthDiff = codewordLengths[index] - codewordLengths[index-1];
        codeword <<= lengthDiff;

        codewords.push_back(codeword);
    }

    // build indices (by codeword length) into table
    uint codewordLengthMax = uint(codewordCountPerLength.size());
    assert(codewordLengthMax <= MAX_CODEWORD_BITS);
    // loop over codeword lengths (actually (length-1))
    for(uint codewordLength = 0, entry = 0; codewordLength < codewordLengthMax; codewordLength++) {
        if(codewordCountPerLength[codewordLength] > 0) {
            // current entry is first codeword of this length
            m_pCodewordFirstIndexPerLength[codewordLength] = entry;
            // store value of first codeword of this length
            m_pCodewordMinPerLength[codewordLength] = codewords[entry];
            // move to last codeword of this length
            entry += codewordCountPerLength[codewordLength] - 1;
            // store value of last codeword of this length
            m_pCodewordMaxPerLength[codewordLength] = codewords[entry];
            // move to first codeword of next length
            entry++;
        } else {
            m_pCodewordFirstIndexPerLength[codewordLength] = -1;
            m_pCodewordMinPerLength[codewordLength] = -1;
            m_pCodewordMaxPerLength[codewordLength] = -1;
        }
    }
}


//////////////////////////////////////////////////////////////////////////////


size_t HuffmanEncodeTable::getRequiredMemory(const Instance* pInstance)
{
    uint tableCountMax = pInstance->m_streamCountMax;
    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    size_t size = 0;

    // dpHistograms
    for(uint i = 0; i < tableCountMax; i++) {
        size += getAlignedSize(distinctSymbolCountMax * sizeof(uint), 128);
    }

    // dpReduceOut
    size += getAlignedSize(tableCountMax * sizeof(Symbol32), 128);

    return size;
}

void HuffmanEncodeTable::init(Instance* pInstance)
{
    uint tableCountMax = pInstance->m_streamCountMax;
    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;

    uint distinctSymbolCountMaxAligned = (uint)getAlignedSize(distinctSymbolCountMax, 128 / sizeof(uint));

    cudaSafeCall(cudaMallocHost(&pInstance->HuffmanTable.pReadback, tableCountMax * distinctSymbolCountMaxAligned * sizeof(uint)));
}

void HuffmanEncodeTable::shutdown(Instance* pInstance)
{
    cudaSafeCall(cudaFreeHost(pInstance->HuffmanTable.pReadback));
    pInstance->HuffmanTable.pReadback = nullptr;
}


HuffmanEncodeTable::HuffmanEncodeTable(const Instance* pInstance)
    : m_symbolMax(0), m_codewordTableSize(0)
{
    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;
    uint codewordTableSizeMax = distinctSymbolCountMax;
    //cudaSafeCall(cudaMallocHost(&m_pCodewords,       codewordTableSizeMax * sizeof(uint), cudaHostAllocWriteCombined));
    //cudaSafeCall(cudaMallocHost(&m_pCodewordLengths, codewordTableSizeMax * sizeof(uint), cudaHostAllocWriteCombined));
    m_pCodewords = new uint[codewordTableSizeMax];
    m_pCodewordLengths = new uint[codewordTableSizeMax];
}

HuffmanEncodeTable::HuffmanEncodeTable(HuffmanEncodeTable&& other)
{
    m_symbolMax = other.m_symbolMax;
    m_symbols.swap(other.m_symbols);
    m_codewordCountPerLength.swap(other.m_codewordCountPerLength);

    m_pCodewords = other.m_pCodewords;
    other.m_pCodewords = nullptr;
    m_pCodewordLengths = other.m_pCodewordLengths;
    other.m_pCodewordLengths = nullptr;
    m_codewordTableSize = other.m_codewordTableSize;
    other.m_codewordTableSize = 0;
}

HuffmanEncodeTable::~HuffmanEncodeTable()
{
    clear();

    //cudaSafeCall(cudaFreeHost(m_pCodewords));
    //cudaSafeCall(cudaFreeHost(m_pCodewordLengths));
    delete[] m_pCodewords;
    delete[] m_pCodewordLengths;
}


HuffmanEncodeTable& HuffmanEncodeTable::operator=(HuffmanEncodeTable&& other)
{
    if(this == &other)
        return *this;

    m_symbolMax = other.m_symbolMax;
    m_symbols.swap(other.m_symbols);
    m_codewordCountPerLength.swap(other.m_codewordCountPerLength);

    cudaSafeCall(cudaFreeHost(m_pCodewords));
    m_pCodewords = other.m_pCodewords;
    other.m_pCodewords = nullptr;
    cudaSafeCall(cudaFreeHost(m_pCodewordLengths));
    m_pCodewordLengths = other.m_pCodewordLengths;
    other.m_pCodewordLengths = nullptr;
    m_codewordTableSize = other.m_codewordTableSize;
    other.m_codewordTableSize = 0;

    return *this;
}


void HuffmanEncodeTable::clear()
{
    m_codewordTableSize = 0;

    m_symbolMax = 0;
    m_symbols.clear();
    m_codewordCountPerLength.clear();
}


bool HuffmanEncodeTable::design(Instance* pInstance, HuffmanEncodeTable* pTables, uint tableCount, const Symbol16** pdpSymbolStreams, const uint* pSymbolCountPerStream)
{
    return design<Symbol16>(pInstance, pTables, tableCount, pdpSymbolStreams, pSymbolCountPerStream);
}

bool HuffmanEncodeTable::design(Instance* pInstance, HuffmanEncodeTable* pTables, uint tableCount, const Symbol32** pdpSymbolStreams, const uint* pSymbolCountPerStream)
{
    return design<Symbol32>(pInstance, pTables, tableCount, pdpSymbolStreams, pSymbolCountPerStream);
}

template<typename Symbol>
bool HuffmanEncodeTable::design(Instance* pInstance, HuffmanEncodeTable* pTables, uint tableCount, const Symbol** pdpSymbolStreams, const uint* pSymbolCountPerStream)
{
    Symbol* dpReduceOut = pInstance->getBuffer<Symbol>(tableCount);


    for(uint i = 0; i < tableCount; i++) {
        pTables[i].clear();
    }

    // find max symbol
    for(uint i = 0; i < tableCount; i++) {
        reduceArray<Symbol, OperatorMax<Symbol>>(dpReduceOut + i, pdpSymbolStreams[i], pSymbolCountPerStream[i], pInstance->m_pReducePlan);
        cudaCheckMsg("HuffmanEncodeTable::design: Error in reduceArray");
    }
    Symbol* pTableSymbolMax = (Symbol*)pInstance->HuffmanTable.pReadback;
    cudaSafeCall(cudaMemcpy(pTableSymbolMax, dpReduceOut, tableCount * sizeof(Symbol), cudaMemcpyDeviceToHost));
    Symbol symbolMax = 0;
    for(uint i = 0; i < tableCount; i++) {
        pTables[i].m_symbolMax = pTableSymbolMax[i];
        symbolMax = max(symbolMax, pTables[i].m_symbolMax);
    }


    uint distinctSymbolCountMax = 1 << pInstance->m_log2HuffmanDistinctSymbolCountMax;
    uint distinctSymbolCount = symbolMax + 1;
    if(distinctSymbolCount > distinctSymbolCountMax) {
        //TODO maybe clamp values instead of failing?
        printf("WARNING: distinctSymbolCount == %u > %u, huffman table design failed.\n", distinctSymbolCount, distinctSymbolCountMax);
#ifdef _DEBUG
        __debugbreak();
#endif
        pInstance->releaseBuffers(1);
        return false;
    }


    uint distinctSymbolCountAligned = (uint)getAlignedSize(distinctSymbolCount, 128 / sizeof(uint));
    uint* dpHistograms = pInstance->getBuffer<uint>(tableCount * distinctSymbolCountAligned);

    std::vector<uint*> pdpHistograms(tableCount);
    for(uint i = 0; i < tableCount; i++) {
        pdpHistograms[i] = dpHistograms + i * distinctSymbolCountAligned;
    }


    // find symbol probabilities
    assert(distinctSymbolCount <= distinctSymbolCountMax);
    histogram(pInstance, pdpHistograms.data(), tableCount, pdpSymbolStreams, pSymbolCountPerStream, distinctSymbolCount);

    cudaSafeCall(cudaMemcpy(pInstance->HuffmanTable.pReadback, dpHistograms, tableCount * distinctSymbolCountAligned * sizeof(uint), cudaMemcpyDeviceToHost));

    #pragma omp parallel for
    for(int i = 0; i < int(tableCount); i++) {
        // build actual encode table
        uint distinctSymbolCountThisTable = pTables[i].m_symbolMax + 1;
        pTables[i].build(pInstance->HuffmanTable.pReadback + i * distinctSymbolCountAligned, distinctSymbolCountThisTable);
    }
    
    pInstance->releaseBuffers(2);

    return true;
}

void HuffmanEncodeTable::copyToBuffer(uint* pCodewords, uint* pCodewordLengths) const
{
    size_t size = m_codewordTableSize * sizeof(uint);
    memcpy(pCodewords,       m_pCodewords,       size);
    memcpy(pCodewordLengths, m_pCodewordLengths, size);
}

void HuffmanEncodeTable::uploadToGPU(uint* dpCodewords, uint* dpCodewordLengths) const
{
    size_t size = m_codewordTableSize * sizeof(uint);
    cudaSafeCall(cudaMemcpy(dpCodewords,       m_pCodewords,       size, cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(dpCodewordLengths, m_pCodewordLengths, size, cudaMemcpyHostToDevice));
}

void HuffmanEncodeTable::uploadToGPUAsync(const Instance* pInstance, uint* dpCodewords, uint* dpCodewordLengths) const
{
    size_t size = m_codewordTableSize * sizeof(uint);
    cudaSafeCall(cudaMemcpyAsync(dpCodewords,       m_pCodewords,       size, cudaMemcpyHostToDevice, pInstance->m_stream));
    cudaSafeCall(cudaMemcpyAsync(dpCodewordLengths, m_pCodewordLengths, size, cudaMemcpyHostToDevice, pInstance->m_stream));
}

void HuffmanEncodeTable::writeToBitStream(const Instance* pInstance, BitStream& bitstream) const
{
    // write #codewords per length
    bitstream.writeBits(uint(m_codewordCountPerLength.size()), LOG2_MAX_CODEWORD_BITS);
    for(uint i = 0; i < m_codewordCountPerLength.size(); i++) {
        bitstream.writeBits(m_codewordCountPerLength[i], pInstance->m_log2HuffmanDistinctSymbolCountMax);
    }
    // write symbols (ordered by codeword length)
    bitstream.writeBits(uint(m_symbols.size()), pInstance->m_log2HuffmanDistinctSymbolCountMax);
    uint symbolBits = getRequiredBits(m_symbolMax);
    bool longSymbols = (pInstance->m_log2HuffmanDistinctSymbolCountMax > 16);
    bitstream.writeBits(symbolBits, longSymbols ? LOG2_MAX_SYMBOL32_BITS : LOG2_MAX_SYMBOL16_BITS); // note: if symbolBits is 16, this will actually write out 0...
    for(uint i = 0; i < m_symbols.size(); i++) {
        bitstream.writeBits(m_symbols[i], symbolBits);
    }
}

void HuffmanEncodeTable::build(const uint* pSymbolProbabilities, uint distinctSymbolCount)
{
    std::vector<HuffmanTreeNode> huffmanNodes(2 * distinctSymbolCount - 1);
    uint nextNodeIndex = 0;

    // build list of all used symbols, packed in HuffmanTreeNodes
    // these will be the leaves of the huffman tree
    std::vector<HuffmanTreeNode*> treeLeaves;
    for(uint symbol = 0; symbol < distinctSymbolCount; symbol++) {
        if(pSymbolProbabilities[symbol] > 0) {
            huffmanNodes[nextNodeIndex].init(symbol, pSymbolProbabilities[symbol]);
            treeLeaves.push_back(&huffmanNodes[nextNodeIndex]);
            nextNodeIndex++;
        }
    }

    if(treeLeaves.empty())
        return;

    // list of huffman nodes to process
    std::priority_queue<HuffmanTreeNode*, std::vector<HuffmanTreeNode*>, HuffmanTreeNodeProbabilityInvComparer> treeNodesTodo(treeLeaves.begin(), treeLeaves.end());

    // build the huffman tree by successively combining the lowest-probability nodes
    while(treeNodesTodo.size() > 1) {
        uint newNodeIndex = nextNodeIndex++;
        HuffmanTreeNode& newNode = huffmanNodes[newNodeIndex];

        newNode.init(INVALID_SYMBOL32, 0);

        // get nodes with lowest probability as children
        HuffmanTreeNode* pLeftChild = treeNodesTodo.top();
        treeNodesTodo.pop();
        HuffmanTreeNode* pRightChild = treeNodesTodo.top();
        treeNodesTodo.pop();

        newNode.m_pLeftChild = pLeftChild;
        newNode.m_pRightChild = pRightChild;

        // combine probabilities
        newNode.m_probability = pLeftChild->m_probability + pRightChild->m_probability;

        // insert into todo list
        treeNodesTodo.push(&newNode);
    }

    HuffmanTreeNode& rootNode = *treeNodesTodo.top();

    // assign codeword length = tree level
    uint codewordLengthMax = (uint)rootNode.assignCodewordLength(0);

    // sort leaves (ie actual symbols) by codeword length
    std::sort(treeLeaves.begin(), treeLeaves.end(), HuffmanTreeNodeCodewordLengthComparer());

    // fill codeword count list and symbol list from leaves list
    m_codewordCountPerLength.resize(codewordLengthMax);
    m_symbols.resize(treeLeaves.size());
    for(uint i = 0; i < treeLeaves.size(); i++) {
        const HuffmanTreeNode& node = *treeLeaves[i];

        if(node.m_codewordLength > 0)
            m_codewordCountPerLength[node.m_codewordLength - 1]++;
        m_symbols[i] = node.m_symbol;
    }

    // count total number of codewords
    uint codewordCount = std::accumulate(m_codewordCountPerLength.begin(), m_codewordCountPerLength.end(), 0);
    if(m_codewordCountPerLength.empty()) {
        // this can happen when all symbols are the same -> only a single "codeword" with length 0
        codewordCount++;
    }

    // make array of codeword lengths (ordered by codeword index)
    uint* pCodewordLengthsByIndex = new uint[codewordCount];
    if(!m_codewordCountPerLength.empty()) {
        uint index = 0;
        for(uint codewordLength = 1; codewordLength <= codewordLengthMax; codewordLength++) {
            for(uint i = 0; i < m_codewordCountPerLength[codewordLength - 1]; i++) {
                pCodewordLengthsByIndex[index++] = codewordLength;
            }
        }
    } else {
        pCodewordLengthsByIndex[0] = 0;
    }

    // assign codewords (ordered by codeword index)
    uint* pCodewordsByIndex = new uint[codewordCount];
    pCodewordsByIndex[0] = 0;
    for(uint index = 1; index < codewordCount; index++) {
        // new codeword = increment previous codeword
        pCodewordsByIndex[index] = pCodewordsByIndex[index-1] + 1;
        // append zero bits as required to reach correct length
        uint lengthDiff = pCodewordLengthsByIndex[index] - pCodewordLengthsByIndex[index-1];
        pCodewordsByIndex[index] <<= lengthDiff;
    }

    m_codewordTableSize = distinctSymbolCount;

    // fill tables with invalid values
    memset(m_pCodewords,        0, m_codewordTableSize * sizeof(uint));
    memset(m_pCodewordLengths, -1, m_codewordTableSize * sizeof(uint));

    // reorder codewords and lengths by symbol
    for(uint index = 0; index < codewordCount; index++) {
        m_pCodewords      [m_symbols[index]] = pCodewordsByIndex      [index];
        m_pCodewordLengths[m_symbols[index]] = pCodewordLengthsByIndex[index];
    }

    delete[] pCodewordsByIndex;
    delete[] pCodewordLengthsByIndex;
}

}
