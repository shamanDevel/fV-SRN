#ifndef __TUM3D_CUDACOMPRESS__ENCODE_H__
#define __TUM3D_CUDACOMPRESS__ENCODE_H__


#include <cudaCompress/global.h>

#include <cudaCompress/BitStream.h>

#include <cudaCompress/EncodeCommon.h>


namespace cudaCompress {

class Instance;


size_t encodeGetRequiredMemory(const Instance* pInstance);
bool encodeInit(Instance* pInstance);
bool encodeShutdown(Instance* pInstance);

// note: all decode* functions assume that the bitstream memory is already page-locked; performace might degrade slightly if it isn't
// note: for decodeRLHuff, the output arrays (pdpSymbolStreams) must be zeroed!

//FIXME: encodeHuff modifies pdpSymbolStreams (adds padding for histogram)
//FIXME: decodeHuff expects its output arrays (pdpSymbolStreams) to be padded to a multiple of WARP_SIZE*codingBlockSize

// single bitstream for all blocks
CUCOMP_DLL bool encodeRLHuff(Instance* pInstance, BitStream& bitStream, const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
CUCOMP_DLL bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly& bitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

CUCOMP_DLL bool encodeHuff(Instance* pInstance, BitStream& bitStream, /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
CUCOMP_DLL bool decodeHuff(Instance* pInstance, BitStreamReadOnly& bitStream, Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

CUCOMP_DLL bool encodeRLHuff(Instance* pInstance, BitStream& bitStream, const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
CUCOMP_DLL bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly& bitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

CUCOMP_DLL bool encodeHuff(Instance* pInstance, BitStream& bitStream, /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
CUCOMP_DLL bool decodeHuff(Instance* pInstance, BitStreamReadOnly& bitStream, Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);


// separate bitstream for each block (but may contain duplicates)
CUCOMP_DLL bool encodeRLHuff(Instance* pInstance, BitStream* ppBitStreams[], const Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
CUCOMP_DLL bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

CUCOMP_DLL bool encodeHuff(Instance* pInstance, BitStream* ppBitStreams[], /*const*/ Symbol16* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
CUCOMP_DLL bool decodeHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol16* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

CUCOMP_DLL bool encodeRLHuff(Instance* pInstance, BitStream* ppBitStreams[], const Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
CUCOMP_DLL bool decodeRLHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);

CUCOMP_DLL bool encodeHuff(Instance* pInstance, BitStream* ppBitStreams[], /*const*/ Symbol32* const pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);
CUCOMP_DLL bool decodeHuff(Instance* pInstance, BitStreamReadOnly* ppBitStreams[], Symbol32* pdpSymbolStreams[], uint streamCount, uint symbolCountPerBlock);


//void encodeResetBitCounts();
//void encodePrintBitCounts();

}


#endif
