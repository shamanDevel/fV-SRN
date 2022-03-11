#ifndef __TUM3D__DWT_H__
#define __TUM3D__DWT_H__


#include <cudaCompress/global.h>

#include <cuda_runtime.h>


namespace cudaCompress {

namespace util {


// for all DWT functions: all sizes have to be even!


// perform scalar DWT on one channel of an array with srcChannelCount interleaved channels (ie elemStride is srcChannelCount)
// dpSource points to the correct channel within the first data element
// srcChannelCount has to be between 1 and 4
CUCOMP_DLL void dwtFloat2DForward(
    float* dpDest, float* dpBuffer, const float* dpSource,
    int sizeX, int sizeY, int srcChannelCount = 1,
    int dstRowPitch = 0,
    int srcRowPitch = 0,
    cudaStream_t stream = 0);
CUCOMP_DLL void dwtFloat3DForward(
    float* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpSource,
    int sizeX, int sizeY, int sizeZ, int srcChannelCount = 1,
    int dstRowPitch = 0, int dstSlicePitch = 0,
    int srcRowPitch = 0, int srcSlicePitch = 0,
    cudaStream_t stream = 0);
// perform scalar IDWT into one channel of an array with dstChannelCount interleaved channels (ie elemStride is dstChannelCount)
// dpDest points to the correct channel within the first data element
// dstChannelCount has to be between 1 and 4
CUCOMP_DLL void dwtFloat2DInverse(
    float* dpDest, float* dpBuffer, const float* dpSource,
    int sizeX, int sizeY, int dstChannelCount = 1,
    int dstRowPitch = 0,
    int srcRowPitch = 0,
    cudaStream_t stream = 0);
CUCOMP_DLL void dwtFloat3DInverse(
    float* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpSource,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount = 1,
    int dstRowPitch = 0, int dstSlicePitch = 0,
    int srcRowPitch = 0, int srcSlicePitch = 0,
    cudaStream_t stream = 0);
// version of dwtFloatInverse that also performs de-quantization of highpass bands
// dppHighpass is an array (size 3 for 2D, 7 for 3D) of pointers to the quantized highpass bands
CUCOMP_DLL void dwtFloat2DInverseFromSymbols(
    float* dpDest, float* dpBuffer,
    const float* dpLowpass, const ushort*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int dstChannelCount = 1,
    int dstRowPitch = 0,
    int lowpassRowPitch = 0,
    cudaStream_t stream = 0);
CUCOMP_DLL void dwtFloat3DInverseFromSymbols(
    float* dpDest, float* dpBuffer2, float* dpBuffer1,
    const float* dpLowpass, const ushort*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount = 1,
    int dstRowPitch = 0, int dstSlicePitch = 0,
    int lowpassRowPitch = 0, int lowpassSlicePitch = 0,
    cudaStream_t stream = 0);
CUCOMP_DLL void dwtFloat2DInverseFromSymbols(
    float* dpDest, float* dpBuffer,
    const float* dpLowpass, const uint*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int dstChannelCount = 1,
    int dstRowPitch = 0,
    int lowpassRowPitch = 0,
    cudaStream_t stream = 0);
CUCOMP_DLL void dwtFloat3DInverseFromSymbols(
    float* dpDest, float* dpBuffer2, float* dpBuffer1,
    const float* dpLowpass, const uint*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount = 1,
    int dstRowPitch = 0, int dstSlicePitch = 0,
    int lowpassRowPitch = 0, int lowpassSlicePitch = 0,
    cudaStream_t stream = 0);

// perform scalar DWT on one channel of a byte array with srcChannelCount interleaved channels (ie elemStride is srcChannelCount)
// dpSource points to the correct channel within the first data element
// srcChannelCount has to be between 1 and 4
CUCOMP_DLL void dwtFloat2DForwardFromByte(
    float* dpDest, float* dpBuffer, const byte* dpSource,
    int sizeX, int sizeY, int srcChannelCount = 1,
    int dstRowPitch = 0,
    int srcRowPitch = 0,
    cudaStream_t stream = 0);
CUCOMP_DLL void dwtFloat3DForwardFromByte(
    float* dpDest, float* dpBuffer2, float* dpBuffer1, const byte* dpSource,
    int sizeX, int sizeY, int sizeZ, int srcChannelCount = 1,
    int dstRowPitch = 0, int dstSlicePitch = 0,
    int srcRowPitch = 0, int srcSlicePitch = 0,
    cudaStream_t stream = 0);
// perform scalar IDWT into one channel of a byte array with dstChannelCount interleaved channels (ie elemStride is dstChannelCount)
// dpDest points to the correct channel within the first data element
// dstChannelCount has to be between 1 and 4
CUCOMP_DLL void dwtFloat2DInverseToByte(
    byte* dpDest, float* dpBuffer, const float* dpSource,
    int sizeX, int sizeY, int dstChannelCount = 1,
    int dstRowPitch = 0,
    int srcRowPitch = 0,
    cudaStream_t stream = 0);
CUCOMP_DLL void dwtFloat3DInverseToByte(
    byte* dpDest, float* dpBuffer2, float* dpBuffer1, const float* dpSource,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount = 1,
    int dstRowPitch = 0, int dstSlicePitch = 0,
    int srcRowPitch = 0, int srcSlicePitch = 0,
    cudaStream_t stream = 0);

CUCOMP_DLL void dwtFloat2DInverseFromSymbolsToByte(
    byte* dpDest, float* dpBuffer,
    const float* dpLowpass, const ushort*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int dstChannelCount = 1,
    int dstRowPitch = 0,
    int lowpassRowPitch = 0,
    cudaStream_t stream = 0);
CUCOMP_DLL void dwtFloat3DInverseFromSymbolsToByte(
    byte* dpDest, float* dpBuffer2, float* dpBuffer1,
    const float* dpLowpass, const ushort*const* dppHighpass, float quantStep,
    int sizeX, int sizeY, int sizeZ, int dstChannelCount = 1,
    int dstRowPitch = 0, int dstSlicePitch = 0,
    int lowpassRowPitch = 0, int lowpassSlicePitch = 0,
    cudaStream_t stream = 0);

// lowpass-only version of forward DWT: stores only lowpass, discards highpass
CUCOMP_DLL void dwtFloat2DForwardLowpassOnlyFromByte(
    float* dpDest, float* dpBuffer, const byte* dpSource,
    int sizeX, int sizeY, int srcChannelCount = 1,
    int dstRowPitch = 0,
    int srcRowPitch = 0,
    cudaStream_t stream = 0);




// integer DWT (lifting) based on CDF 5/3 wavelet

CUCOMP_DLL void dwtIntForward(
    short* dpDest, short* dpBuffer, const short* dpSource,
    int sizeX, int sizeY, int sizeZ,
    int dstRowPitch = 0, int dstSlicePitch = 0,
    int srcRowPitch = 0, int srcSlicePitch = 0,
    cudaStream_t stream = 0);

CUCOMP_DLL void dwtIntInverse(
    short* dpDest, short* dpBuffer, const short* dpSource,
    int sizeX, int sizeY, int sizeZ,
    int dstRowPitch = 0, int dstSlicePitch = 0,
    int srcRowPitch = 0, int srcSlicePitch = 0,
    cudaStream_t stream = 0);

CUCOMP_DLL void dwtIntForwardLowpassOnly(
    short* dpDest, short* dpBuffer, const short* dpSource,
    int sizeX, int sizeY,
    int dstRowPitch = 0,
    int srcRowPitch = 0,
    cudaStream_t stream = 0);

}

}


#endif
