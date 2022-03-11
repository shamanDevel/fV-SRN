#ifndef __TUM3D_CUDACOMPRESS__UTIL_H__
#define __TUM3D_CUDACOMPRESS__UTIL_H__


#include <cudaCompress/global.h>


#ifndef __CUDACC__
#include <algorithm>
using std::min;
using std::max;
#endif

template<typename T>
inline T clamp(T val, T valMin, T valMax)
{
    return min(max(val, valMin), valMax);
}


namespace cudaCompress {

inline uint getRequiredBits(uint maxValue)
{
    uint requiredBits = 0;
    while(maxValue > 0) {
        requiredBits++;
        maxValue >>= 1;
    }

    return requiredBits;
}

inline size_t getAlignedSize(size_t size, uint numBytes)
{
    return (size + numBytes - 1) / numBytes * numBytes;
}

template<typename T>
inline void align(T*& pData, uint numBytes)
{
    pData = (T*)getAlignedSize(size_t(pData), numBytes);
}

}


#endif
