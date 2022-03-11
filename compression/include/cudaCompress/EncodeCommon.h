#ifndef __TUM3D_CUDACOMPRESS__ENCODE_COMMON_H__
#define __TUM3D_CUDACOMPRESS__ENCODE_COMMON_H__


#include <cudaCompress/global.h>


namespace cudaCompress {

typedef ushort Symbol16;
typedef uint   Symbol32;

const Symbol16 INVALID_SYMBOL16 = Symbol16(-1);
const Symbol32 INVALID_SYMBOL32 = Symbol32(-1);

const uint LOG2_MAX_SYMBOL16_BITS = 4;
const uint MAX_SYMBOL16_BITS = (1 << LOG2_MAX_SYMBOL16_BITS); // = sizeof(Symbol16) * 8
static_assert(MAX_SYMBOL16_BITS == sizeof(Symbol16) * 8, "Inconsistent constants: MAX_SYMBOL16_BITS is not equal to sizeof(Symbol16) * 8");

const uint LOG2_MAX_SYMBOL32_BITS = 5;
const uint MAX_SYMBOL32_BITS = (1 << LOG2_MAX_SYMBOL32_BITS); // = sizeof(Symbol32) * 8
static_assert(MAX_SYMBOL32_BITS == sizeof(Symbol32) * 8, "Inconsistent constants: MAX_SYMBOL32_BITS is not equal to sizeof(Symbol32) * 8");

const uint LOG2_MAX_CODEWORD_BITS = 5;
const uint MAX_CODEWORD_BITS = (1 << LOG2_MAX_CODEWORD_BITS);

const uint LOG2_HUFFMAN_LOOKUP_SIZE = 7;
const uint HUFFMAN_LOOKUP_SIZE = (1 << LOG2_HUFFMAN_LOOKUP_SIZE);


const uint ZERO_COUNT_MAX = 255;

}


#endif
