#ifndef __TUM3D_CUDACOMPRESS__PACK_INC_H__
#define __TUM3D_CUDACOMPRESS__PACK_INC_H__


#include <cudaCompress/global.h>


namespace cudaCompress {

class Instance;


size_t packIncGetRequiredMemory(const Instance* pInstance);
bool packIncInit(Instance* pInstance);
bool packIncShutdown(Instance* pInstance);

// dpValues must be monotonically increasing for packInc
// dpValues and dpPackedValueIncrements may be the same
bool packInc(Instance* pInstance, const uint* dpValues, uint* dpPackedValueIncrements, uint valueCount, uint& bitsPerValue);
bool unpackInc(Instance* pInstance, uint* dpValues, const uint* dpPackedValueIncrements, uint valueCount, uint bitsPerValue);

// simple versions that always use 16 bit per value increment
bool packInc16(Instance* pInstance, const uint* dpValues, ushort* dpValueIncrements, uint valueCount);
bool unpackInc16(Instance* pInstance, uint* dpValues, const ushort* dpValueIncrements, uint valueCount);

void packInc16CPU(const uint* pValues, ushort* pValueIncrements, uint valueCount); // works in-place
void unpackInc16CPU(uint* pValues, const ushort* pValueIncrements, uint valueCount); // does *not* work in-place

}


#endif
