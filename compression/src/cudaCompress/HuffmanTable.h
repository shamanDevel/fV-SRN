#ifndef __TUM3D_CUDACOMPRESS__HUFFMAN_TABLE_H__
#define __TUM3D_CUDACOMPRESS__HUFFMAN_TABLE_H__


#include <cudaCompress/global.h>

#include <vector>

#include <cudaCompress/BitStream.h>

#include <cudaCompress/EncodeCommon.h>


namespace cudaCompress {

class Instance;


class HuffmanDecodeTable
{
public:
    HuffmanDecodeTable(const Instance* pInstance);
    HuffmanDecodeTable(HuffmanDecodeTable&& other);
    ~HuffmanDecodeTable();

    HuffmanDecodeTable& operator=(HuffmanDecodeTable&& other);

    void clear();
    void readFromBitStream(const Instance* pInstance, BitStreamReadOnly& bitstream);

    uint getSymbolTableSize() const { return m_symbolTableSize; }

    static uint computeMaxGPUSize(const Instance* pInstance);
    uint computeGPUSize(const Instance* pInstance) const;
    void copyToBuffer(const Instance* pInstance, byte* pTable) const;
    void uploadToGPU(const Instance* pInstance, byte* dpTable) const;
    void uploadToGPUAsync(const Instance* pInstance, byte* dpTable) const;
    void syncOnLastAsyncUpload() const;

private:
    byte* m_pStorage;

    // indexed by codeword length
    // these are just pointers into m_pCodewordIndex
    int* m_pCodewordFirstIndexPerLength;
    int* m_pCodewordMinPerLength;
    int* m_pCodewordMaxPerLength;

    // indexed by codeword index
    byte* m_pSymbolTable;
    uint m_symbolTableSize;

    cudaEvent_t m_uploadSyncEvent;

    void build(const std::vector<uint>& codewordCountPerLength);

    // don't allow copy or assignment
    HuffmanDecodeTable(const HuffmanDecodeTable&);
    void operator=(const HuffmanDecodeTable&);
};

class HuffmanEncodeTable
{
public:
    static size_t getRequiredMemory(const Instance* pInstance);
    static void init(Instance* pInstance);
    static void shutdown(Instance* pInstance);

    HuffmanEncodeTable(const Instance* pInstance);
    HuffmanEncodeTable(HuffmanEncodeTable&& other);
    ~HuffmanEncodeTable();

    HuffmanEncodeTable& operator=(HuffmanEncodeTable&& other);

    void clear();
    static bool design(Instance* pInstance, HuffmanEncodeTable* pTables, uint tableCount, const Symbol16** pdpSymbolStreams, const uint* pSymbolCountPerStream);
    static bool design(Instance* pInstance, HuffmanEncodeTable* pTables, uint tableCount, const Symbol32** pdpSymbolStreams, const uint* pSymbolCountPerStream);

    uint getTableSize() const { return m_codewordTableSize; }
    void copyToBuffer(uint* pCodewords, uint* pCodewordLengths) const;
    void uploadToGPU(uint* dpCodewords, uint* dpCodewordLengths) const;
    void uploadToGPUAsync(const Instance* pInstance, uint* dpCodewords, uint* dpCodewordLengths) const;

    void writeToBitStream(const Instance* pInstance, BitStream& bitstream) const;

private:
    template<typename Symbol>
    static bool design(Instance* pInstance, HuffmanEncodeTable* pTables, uint tableCount, const Symbol** pdpSymbolStreams, const uint* pSymbolCountPerStream);
    void build(const uint* pSymbolProbabilities, uint distinctSymbolCount);

    // data to be written to bitstream
    uint                m_symbolMax;
    std::vector<uint>   m_symbols;
    std::vector<uint>   m_codewordCountPerLength;

    // the actual encode table, indexed by symbol
    uint* m_pCodewords;
    uint* m_pCodewordLengths;
    uint m_codewordTableSize;

    // don't allow copy or assignment
    HuffmanEncodeTable(const HuffmanEncodeTable&);
    HuffmanEncodeTable& operator=(const HuffmanEncodeTable&);
};

}


#endif
