#include "memtrace.h"
#include <cudaCompress/util/Quantize.h>

#include <cudaCompress/tools/Operator.h>
#include <cudaCompress/tools/Functor.h>

#include <cudaCompress/cudaUtil.h>
#include <cudaCompress/InstanceImpl.h>

#include <cudaCompress/reduce/reduce_app.cui>

#include "QuantizeKernels.cui"


namespace cudaCompress {

namespace util {


void floatToByte2D(byte* dpQuant, uint channelCount, uint channel, const float* dpData, uint sizeX, uint sizeY, uint rowPitchSrc)
{
    if(rowPitchSrc == 0) rowPitchSrc = sizeX;

    dim3 blockSize(64, 4);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    dim3 blockCount(blockCountX, blockCountY);

    quantize2Dkernel<<<blockCount, blockSize>>>(dpQuant + channel, channelCount, dpData, sizeX, sizeY, rowPitchSrc);
    cudaCheckMsg("quantize2Dkernel execution failed");
}

void byteToFloat2D(float* dpData, const byte* dpQuant, uint channelCount, uint channel, uint sizeX, uint sizeY, uint rowPitchDst)
{
    if(rowPitchDst == 0) rowPitchDst = sizeX;

    dim3 blockSize(64, 4);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    dim3 blockCount(blockCountX, blockCountY);

    unquantize2Dkernel<<<blockCount, blockSize>>>(dpData, dpQuant + channel, channelCount, sizeX, sizeY, rowPitchDst);
    cudaCheckMsg("unquantize2Dkernel execution failed");
}



void symbolize(ushort* dpSymbols, const short* dpData, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchSrc, uint slicePitchSrc)
{
    if(rowPitchSrc == 0) rowPitchSrc = sizeX;
    if(slicePitchSrc == 0) slicePitchSrc = rowPitchSrc * sizeY;

    dim3 blockSize(64, 4, 1);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    uint blockCountZ = (sizeZ + blockSize.z - 1) / blockSize.z;
    dim3 blockCount(blockCountX, blockCountY, blockCountZ);

    symbolizeKernel<<<blockCount, blockSize>>>(dpSymbols, dpData, sizeX, sizeY, sizeZ, rowPitchSrc, slicePitchSrc);
    cudaCheckMsg("symbolizeKernel execution failed");
}

void unsymbolize(short* dpData, const ushort* dpSymbols, uint sizeX, uint sizeY, uint sizeZ, uint rowPitchDst, uint slicePitchDst)
{
    if(rowPitchDst == 0) rowPitchDst = sizeX;

    dim3 blockSize(64, 4, 1);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    uint blockCountZ = (sizeZ + blockSize.z - 1) / blockSize.z;
    dim3 blockCount(blockCountX, blockCountY, blockCountZ);

    unsymbolizeKernel<<<blockCount, blockSize>>>(dpData, dpSymbols, sizeX, sizeY, sizeZ, rowPitchDst, slicePitchDst);
    cudaCheckMsg("unsymbolizeKernel execution failed");
}


template<typename Symbol>
void quantizeToSymbols(Symbol* dpQuant, const float* dpData, uint sizeX, uint sizeY, uint sizeZ,
                       float quantizationStep, uint rowPitchSrc, uint slicePitchSrc, EQuantizeType quantType)
{
    if(rowPitchSrc == 0) rowPitchSrc = sizeX;
    if(slicePitchSrc == 0) slicePitchSrc = rowPitchSrc * sizeY;

    dim3 blockSize(64, 4, 1);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    uint blockCountZ = (sizeZ + blockSize.z - 1) / blockSize.z;
    dim3 blockCount(blockCountX, blockCountY, blockCountZ);

    #define CASE(Type) \
        case Type: \
            quantizeToSymbolsKernel<Symbol, Type> \
            <<<blockCount, blockSize>>> \
            (dpQuant, dpData, 1.0f / quantizationStep, sizeX, sizeY, sizeZ, rowPitchSrc, slicePitchSrc); \
            break;
    switch(quantType)
    {
        CASE(QUANTIZE_DEADZONE);
        CASE(QUANTIZE_UNIFORM);
    }
    #undef CASE
    cudaCheckMsg("quantizeToSymbolsKernel execution failed");
}

template<typename Symbol>
void quantizeToSymbolsRoundtrip(Symbol* dpQuant, float* dpData, uint sizeX, uint sizeY, uint sizeZ,
                                float quantizationStep, uint rowPitchSrc, uint slicePitchSrc, EQuantizeType quantType)
{
    if(rowPitchSrc == 0) rowPitchSrc = sizeX;
    if(slicePitchSrc == 0) slicePitchSrc = rowPitchSrc * sizeY;

    dim3 blockSize(64, 4, 1);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    uint blockCountZ = (sizeZ + blockSize.z - 1) / blockSize.z;
    dim3 blockCount(blockCountX, blockCountY, blockCountZ);

    #define CASE(Type) \
        case Type: \
            quantizeToSymbolsRoundtripKernel<Symbol, Type> \
            <<<blockCount, blockSize>>> \
            (dpQuant, dpData, quantizationStep, 1.0f / quantizationStep, sizeX, sizeY, sizeZ, rowPitchSrc, slicePitchSrc); \
            break;
    switch(quantType)
    {
        CASE(QUANTIZE_DEADZONE);
        CASE(QUANTIZE_UNIFORM);
    }
    #undef CASE
    cudaCheckMsg("quantizeToSymbolsRoundtripKernel execution failed");
}

template<typename Symbol>
void unquantizeFromSymbols(float* dpData, const Symbol* dpQuant, uint sizeX, uint sizeY, uint sizeZ,
                           float quantizationStep, uint rowPitchDst, uint slicePitchDst, EQuantizeType quantType)
{
    if(rowPitchDst == 0) rowPitchDst = sizeX;
    if(slicePitchDst == 0) slicePitchDst = rowPitchDst * sizeY;

    dim3 blockSize(64, 4, 1);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    uint blockCountZ = (sizeZ + blockSize.z - 1) / blockSize.z;
    dim3 blockCount(blockCountX, blockCountY, blockCountZ);

    #define CASE(Type) \
        case Type: \
            unquantizeFromSymbolsKernel<Symbol, Type> \
            <<<blockCount, blockSize>>> \
            (dpData, dpQuant, quantizationStep, sizeX, sizeY, sizeZ, rowPitchDst, slicePitchDst); \
            break;
    switch(quantType)
    {
        CASE(QUANTIZE_DEADZONE);
        CASE(QUANTIZE_UNIFORM);
    }
    #undef CASE
    cudaCheckMsg("unquantizeFromSymbolsKernel execution failed");
}



void quantizeToSymbols(ushort* dpSymbols, const float* dpData, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchSrc, uint slicePitchSrc, EQuantizeType quantType)
{
    quantizeToSymbols<ushort>(dpSymbols, dpData, sizeX, sizeY, sizeZ, quantizationStep, rowPitchSrc, slicePitchSrc, quantType);
}
void quantizeToSymbolsRoundtrip(ushort* dpSymbols, float* dpData, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchSrc, uint slicePitchSrc, EQuantizeType quantType)
{
    quantizeToSymbolsRoundtrip<ushort>(dpSymbols, dpData, sizeX, sizeY, sizeZ, quantizationStep, rowPitchSrc, slicePitchSrc, quantType);
}
void unquantizeFromSymbols(float* dpData, const ushort* dpSymbols, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchDst, uint slicePitchDst, EQuantizeType quantType)
{
    unquantizeFromSymbols<ushort>(dpData, dpSymbols, sizeX, sizeY, sizeZ, quantizationStep, rowPitchDst, slicePitchDst, quantType);
}


void quantizeToSymbols(uint* dpSymbols, const float* dpData, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchSrc, uint slicePitchSrc, EQuantizeType quantType)
{
    quantizeToSymbols<uint>(dpSymbols, dpData, sizeX, sizeY, sizeZ, quantizationStep, rowPitchSrc, slicePitchSrc, quantType);
}
void quantizeToSymbolsRoundtrip(uint* dpSymbols, float* dpData, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchSrc, uint slicePitchSrc, EQuantizeType quantType)
{
    quantizeToSymbolsRoundtrip<uint>(dpSymbols, dpData, sizeX, sizeY, sizeZ, quantizationStep, rowPitchSrc, slicePitchSrc, quantType);
}
void unquantizeFromSymbols(float* dpData, const uint* dpSymbols, uint sizeX, uint sizeY, uint sizeZ, float quantizationStep, uint rowPitchDst, uint slicePitchDst, EQuantizeType quantType)
{
    unquantizeFromSymbols<uint>(dpData, dpSymbols, sizeX, sizeY, sizeZ, quantizationStep, rowPitchDst, slicePitchDst, quantType);
}


void quantizeToSymbols2D(ushort* dpSymbols, const float* dpData, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchSrc, EQuantizeType quantType)
{
    quantizeToSymbols(dpSymbols, dpData, sizeX, sizeY, 1, quantizationStep, rowPitchSrc, 0, quantType);
}
void quantizeToSymbolsRoundtrip2D(ushort* dpSymbols, float* dpData, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchSrc, EQuantizeType quantType)
{
    quantizeToSymbolsRoundtrip(dpSymbols, dpData, sizeX, sizeY, 1, quantizationStep, rowPitchSrc, 0, quantType);
}
void unquantizeFromSymbols2D(float* dpData, const ushort* dpSymbols, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchDst, EQuantizeType quantType)
{
    unquantizeFromSymbols(dpData, dpSymbols, sizeX, sizeY, 1, quantizationStep, rowPitchDst, 0, quantType);
}


void quantizeToSymbols2D(uint* dpSymbols, const float* dpData, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchSrc, EQuantizeType quantType)
{
    quantizeToSymbols(dpSymbols, dpData, sizeX, sizeY, 1, quantizationStep, rowPitchSrc, 0, quantType);
}
void quantizeToSymbolsRoundtrip2D(uint* dpSymbols, float* dpData, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchSrc, EQuantizeType quantType)
{
    quantizeToSymbolsRoundtrip(dpSymbols, dpData, sizeX, sizeY, 1, quantizationStep, rowPitchSrc, 0, quantType);
}
void unquantizeFromSymbols2D(float* dpData, const uint* dpSymbols, uint sizeX, uint sizeY, float quantizationStep, uint rowPitchDst, EQuantizeType quantType)
{
    unquantizeFromSymbols(dpData, dpSymbols, sizeX, sizeY, 1, quantizationStep, rowPitchDst, 0, quantType);
}



void quantizeToShort(short* dpQuant, const float* dpData, uint sizeX, uint sizeY, uint sizeZ,
                     float quantizationStep, uint rowPitchSrc, uint slicePitchSrc, EQuantizeType quantType)
{
    if(rowPitchSrc == 0) rowPitchSrc = sizeX;
    if(slicePitchSrc == 0) slicePitchSrc = rowPitchSrc * sizeY;

    dim3 blockSize(64, 4, 1);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    uint blockCountZ = (sizeZ + blockSize.z - 1) / blockSize.z;
    dim3 blockCount(blockCountX, blockCountY, blockCountZ);

    #define CASE(Type) \
        case Type: \
            quantizeToShortKernel<Type> \
            <<<blockCount, blockSize>>> \
            (dpQuant, dpData, 1.0f / quantizationStep, sizeX, sizeY, sizeZ, rowPitchSrc, slicePitchSrc); \
            break;
    switch(quantType)
    {
        CASE(QUANTIZE_DEADZONE);
        CASE(QUANTIZE_UNIFORM);
    }
    #undef CASE
    cudaCheckMsg("quantizeToShortKernel execution failed");
}

void unquantizeFromShort(float* dpData, const short* dpQuant, uint sizeX, uint sizeY, uint sizeZ,
                         float quantizationStep, uint rowPitchDst, uint slicePitchDst, EQuantizeType quantType)
{
    if(rowPitchDst == 0) rowPitchDst = sizeX;
    if(slicePitchDst == 0) slicePitchDst = rowPitchDst * sizeY;

    dim3 blockSize(64, 4, 1);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    uint blockCountZ = (sizeZ + blockSize.z - 1) / blockSize.z;
    dim3 blockCount(blockCountX, blockCountY, blockCountZ);

    #define CASE(Type) \
        case Type: \
            unquantizeFromShortKernel<Type> \
            <<<blockCount, blockSize>>> \
            (dpData, dpQuant, quantizationStep, sizeX, sizeY, sizeZ, rowPitchDst, slicePitchDst); \
            break;
    switch(quantType)
    {
        CASE(QUANTIZE_DEADZONE);
        CASE(QUANTIZE_UNIFORM);
    }
    #undef CASE
    cudaCheckMsg("unquantizeFromShortKernel execution failed");
}



void quantizeDifferenceToSymbols2D(ushort* dpSymbols, const float* dpData, float quantizationStep,
                                   const byte* dpReference, uint channelCount, uint channel,
                                   uint sizeX, uint sizeY, uint rowPitchSrc, EQuantizeType quantType)
{
    if(rowPitchSrc == 0) rowPitchSrc = sizeX;

    dim3 blockSize(64, 4);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    dim3 blockCount(blockCountX, blockCountY);

    #define CASE(Type) \
        case Type: \
            quantizeDifferenceToSymbols2Dkernel<Type> \
            <<<blockCount, blockSize>>> \
            (dpSymbols, dpData, 1.0f / quantizationStep, dpReference + channel, channelCount, sizeX, sizeY, rowPitchSrc); \
            break;
    switch(quantType)
    {
        CASE(QUANTIZE_DEADZONE);
        CASE(QUANTIZE_UNIFORM);
    }
    #undef CASE
    cudaCheckMsg("quantizeDifferenceToSymbols2Dkernel execution failed");
}

void quantizeDifferenceToSymbolsRoundtrip2D(ushort* dpSymbols, float* dpData, float quantizationStep,
                                            const byte* dpReference, uint channelCount, uint channel,
                                            uint sizeX, uint sizeY, uint rowPitchSrc, EQuantizeType quantType)
{
    if(rowPitchSrc == 0) rowPitchSrc = sizeX;

    dim3 blockSize(64, 4);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    dim3 blockCount(blockCountX, blockCountY);

    #define CASE(Type) \
        case Type: \
            quantizeDifferenceToSymbolsRoundtrip2Dkernel<Type> \
            <<<blockCount, blockSize>>> \
            (dpSymbols, dpData, quantizationStep, 1.0f / quantizationStep, dpReference + channel, channelCount, sizeX, sizeY, rowPitchSrc); \
            break;
    switch(quantType)
    {
        CASE(QUANTIZE_DEADZONE);
        CASE(QUANTIZE_UNIFORM);
    }
    #undef CASE
    cudaCheckMsg("quantizeDifferenceToSymbolsRoundtrip2Dkernel execution failed");
}

void unquantizeDifferenceFromSymbols2D(float* dpData, const ushort* dpSymbols, float quantizationStep,
                                       const byte* dpReference, uint channelCount, uint channel,
                                       uint sizeX, uint sizeY, uint rowPitchDst, EQuantizeType quantType, cudaStream_t stream)
{
    if(rowPitchDst == 0) rowPitchDst = sizeX;

    dim3 blockSize(64, 4);
    uint blockCountX = (sizeX + blockSize.x - 1) / blockSize.x;
    uint blockCountY = (sizeY + blockSize.y - 1) / blockSize.y;
    dim3 blockCount(blockCountX, blockCountY);

    #define CASE(Type) \
        case Type: \
            unquantizeDifferenceFromSymbols2Dkernel<Type> \
            <<<blockCount, blockSize, 0, stream>>> \
            (dpData, dpSymbols, quantizationStep, dpReference + channel, channelCount, sizeX, sizeY, rowPitchDst); \
            break;
    switch(quantType)
    {
        CASE(QUANTIZE_DEADZONE);
        CASE(QUANTIZE_UNIFORM);
    }
    #undef CASE
    cudaCheckMsg("unquantizeDifferenceFromSymbols2Dkernel execution failed");
}


void getMaxAbs(Instance* pInstance, const float* dpImage, uint elemCount, float* dpValMax)
{
    if(dpValMax != nullptr) {
        reduceArray<float, OperatorMax<float>, FunctorAbs<float>>(dpValMax, dpImage, elemCount, pInstance->m_pReducePlan);
        cudaCheckMsg("getVolumeFloatMaxAbs: Error in reduceArray");
    }
}


}

}
