#ifndef __TUM3D_CUDACOMPRESS__HUFFMAN_H__
#define __TUM3D_CUDACOMPRESS__HUFFMAN_H__


#include <cudaCompress/global.h>

#include <cudaCompress/EncodeCommon.h>


namespace cudaCompress {

class Instance;


//TODO make separate versions for encoder and decoder?
struct HuffmanGPUStreamInfo
{
    // raw data
    byte* dpSymbolStream;

    // encoded data
    uint* dpCodewordStream;
    uint* dpOffsets;

    // common info
    uint symbolCount;

    // encoder-only info
    uint* dpEncodeCodewords;
    uint* dpEncodeCodewordLengths;

    // decoder-only info
    byte* dpDecodeTable;
    uint decodeSymbolTableSize;
};


size_t huffmanGetRequiredMemory(const Instance* pInstance);
bool huffmanInit(Instance* pInstance);
bool huffmanShutdown(Instance* pInstance);

// note: codingBlockSize has to be a power of 2 between 32 and 256
// note: huffmanEncode assumes that dpCodewordStream is already cleared to all zeros!
// note: huffmanDecode does *not* sync on the upload  of pStreamInfos to complete!!
// note: huffmanDecode requires that the length of each symbol stream (dpSymbolStream) is a multiple of codingBlockSize*WARP_SIZE
//       (transpose kernel requires this)
bool huffmanEncode(Instance* pInstance, const HuffmanGPUStreamInfo* pStreamInfos, uint streamCount, uint codingBlockSize, uint* pCompressedSizeBits);
bool huffmanDecode(Instance* pInstance, const HuffmanGPUStreamInfo* pStreamInfos, uint streamCount, uint codingBlockSize);

}


#endif
