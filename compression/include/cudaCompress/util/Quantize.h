#ifndef __TUM3D_CUDACOMPRESS__QUANTIZE_H__
#define __TUM3D_CUDACOMPRESS__QUANTIZE_H__


#include <cudaCompress/global.h>

#include <cuda_runtime.h>


namespace cudaCompress {

class Instance;

namespace util {

enum EQuantizeType
{
    QUANTIZE_DEADZONE = 0, // midtread quantizer with twice larger zero bin
    QUANTIZE_UNIFORM,      // standard uniform midtread quantizer
    QUANTIZE_COUNT
};


// quantize float to byte (byte array can be multi-channel; quant step 1; shift +128 to make unsigned)
CUCOMP_DLL void floatToByte2D(byte* dpQuant, uint channelCount, uint channel, const float* dpData, uint sizeX, uint sizeY, uint rowPitchSrc = 0);
CUCOMP_DLL void byteToFloat2D(float* dpData, const byte* dpQuant, uint channelCount, uint channel, uint sizeX, uint sizeY, uint rowPitchDst = 0);


// convert signed shorts to symbols (>= 0 -> even, < 0 -> odd)
CUCOMP_DLL void symbolize(ushort* dpSymbols, const short* dpData, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchSrc = 0, uint slicePitchSrc = 0);
CUCOMP_DLL void unsymbolize(short* dpData, const ushort* dpSymbols, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchDst = 0, uint slicePitchDst = 0);


// quantize float to symbols (>= 0 -> even, < 0 -> odd)
CUCOMP_DLL void quantizeToSymbols(ushort* dpSymbols, const float* dpData, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchSrc = 0, uint slicePitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void quantizeToSymbolsRoundtrip(ushort* dpSymbols, float* dpData, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchSrc = 0, uint slicePitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void unquantizeFromSymbols(float* dpData, const ushort* dpSymbols, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchDst = 0, uint slicePitchDst = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);

CUCOMP_DLL void quantizeToSymbols(uint* dpSymbols, const float* dpData, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchSrc = 0, uint slicePitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void quantizeToSymbolsRoundtrip(uint* dpSymbols, float* dpData, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchSrc = 0, uint slicePitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void unquantizeFromSymbols(float* dpData, const uint* dpSymbols, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchDst = 0, uint slicePitchDst = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);

// 2D convenience versions
CUCOMP_DLL void quantizeToSymbols2D(ushort* dpSymbols, const float* dpData, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void quantizeToSymbolsRoundtrip2D(ushort* dpSymbols, float* dpData, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void unquantizeFromSymbols2D(float* dpData, const ushort* dpSymbols, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchDst = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);

CUCOMP_DLL void quantizeToSymbols2D(uint* dpSymbols, const float* dpData, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void quantizeToSymbolsRoundtrip2D(uint* dpSymbols, float* dpData, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void unquantizeFromSymbols2D(float* dpData, const uint* dpSymbols, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchDst = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);


// quantize float to (signed) shorts
CUCOMP_DLL void quantizeToShort(short* dpQuant, const float* dpData, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchSrc = 0, uint slicePitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void unquantizeFromShort(float* dpData, const short* dpQuant, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchDst = 0, uint slicePitchDst = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);


// convert reference to float (with shift -128), subtract from data, quantize result to symbol
CUCOMP_DLL void quantizeDifferenceToSymbols2D(ushort* dpSymbols, const float* dpData, float quantizationStep, const byte* dpReference, uint channelCount, uint channel, uint sizeX, uint sizeY, uint rowPitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void quantizeDifferenceToSymbolsRoundtrip2D(ushort* dpSymbols, float* dpData, float quantizationStep, const byte* dpReference, uint channelCount, uint channel, uint sizeX, uint sizeY, uint rowPitchSrc = 0, EQuantizeType quantType = QUANTIZE_DEADZONE);
CUCOMP_DLL void unquantizeDifferenceFromSymbols2D(float* dpData, const ushort* dpSymbols, float quantizationStep, const byte* dpReference, uint channelCount, uint channel, uint sizeX, uint sizeY, uint rowPitchDst = 0, EQuantizeType quantType = QUANTIZE_DEADZONE, cudaStream_t stream = 0);


// quantization-related helper: get max value
CUCOMP_DLL void getMaxAbs(Instance* pInstance, const float* dpImage, uint elemCount, float* dpValMax);

}

}


#endif
