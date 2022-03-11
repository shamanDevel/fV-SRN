#ifndef __TUM3D_CUDACOMPRESS__BITS_H__
#define __TUM3D_CUDACOMPRESS__BITS_H__


#include <cudaCompress/global.h>


namespace cudaCompress {

namespace util {

//WARNING: for now, elemCount has to be a multiple of 4096 (or the arrays padded accordingly)
//TODO: fix this!

// store least significant bit from each element in dpData in dpBits
CUCOMP_DLL bool getLSB(const short* dpData, uint* dpBits, uint elemCount);

// remove least significant bit from each element in dpData (ie round down) and store in dpBits (if non-null)
// shift is added to each value before the bit is removed
CUCOMP_DLL bool removeLSB(short* dpData, uint* dpBits, uint elemCount, short shift = 0);

// append corresponding bit from dpBits onto each element in dpData (as least significant bit)
// shift is added to each value after the bit has been appended
CUCOMP_DLL bool appendLSB(short* dpData, const uint* dpBits, uint elemCount, short shift = 0);

// add shift to each element of dpData
CUCOMP_DLL bool shiftElems(short* dpData, uint elemCount, short shift);

}

}


#endif
