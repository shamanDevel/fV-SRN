#ifndef __TUM3D_CUDACOMPRESS__RUN_LENGTH_H__
#define __TUM3D_CUDACOMPRESS__RUN_LENGTH_H__


#include <cudaCompress/global.h>

#include <vector>

#include <cudaCompress/EncodeCommon.h>


namespace cudaCompress {

class Instance;


size_t runLengthGetRequiredMemory(const Instance* pInstance);
bool runLengthInit(Instance* pInstance);
bool runLengthShutdown(Instance* pInstance);

// note: for decode, output arrays (pdpSymbols) must be zeroed!

bool runLengthEncode(Instance* pInstance, Symbol16** pdpSymbolsCompact, Symbol16** pdpZeroCounts, const Symbol16** pdpSymbols, const uint* pSymbolCount, uint streamCount, uint zeroCountMax, uint* pSymbolCountCompact);
bool runLengthDecode(Instance* pInstance, const Symbol16** pdpSymbolsCompact, const Symbol16** pdpZeroCounts, const uint* pSymbolCountCompact, Symbol16** pdpSymbols, const uint* pSymbolCount, uint streamCount);
bool runLengthDecode(Instance* pInstance, const Symbol16* dpSymbolsCompact, const Symbol16* dpZeroCounts, const uint* pSymbolCountCompact, uint stride, Symbol16** pdpSymbols, uint symbolCount, uint streamCount);

bool runLengthEncode(Instance* pInstance, Symbol32** pdpSymbolsCompact, Symbol32** pdpZeroCounts, const Symbol32** pdpSymbols, const uint* pSymbolCount, uint streamCount, uint zeroCountMax, uint* pSymbolCountCompact);
bool runLengthDecode(Instance* pInstance, const Symbol32** pdpSymbolsCompact, const Symbol32** pdpZeroCounts, const uint* pSymbolCountCompact, Symbol32** pdpSymbols, const uint* pSymbolCount, uint streamCount);
bool runLengthDecode(Instance* pInstance, const Symbol32* dpSymbolsCompact, const Symbol32* dpZeroCounts, const uint* pSymbolCountCompact, uint stride, Symbol32** pdpSymbols, uint symbolCount, uint streamCount);


}


#endif
