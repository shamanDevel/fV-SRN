#ifndef __TUM3D_CUDACOMPRESS__YCOCG_H__
#define __TUM3D_CUDACOMPRESS__YCOCG_H__


#include <cudaCompress/global.h>

#include <vector_types.h>


namespace cudaCompress {

namespace util {

// RGB <-> YCoCg color space conversion
// dpTarget and dpData may point to the same memory
CUCOMP_DLL void convertRGBtoYCoCg(uchar3* dpTarget, const uchar3* dpData, int pixelCount);
CUCOMP_DLL void convertYCoCgtoRGB(uchar3* dpTarget, const uchar3* dpData, int pixelCount);

CUCOMP_DLL void convertRGBtoYCoCg(uchar4* dpTarget, const uchar4* dpData, int pixelCount);
CUCOMP_DLL void convertYCoCgtoRGB(uchar4* dpTarget, const uchar4* dpData, int pixelCount);

}

}


#endif
