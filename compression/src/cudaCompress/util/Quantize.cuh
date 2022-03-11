#ifndef __TUM3D_CUDACOMPRESS__QUANTIZE_CUH__
#define __TUM3D_CUDACOMPRESS__QUANTIZE_CUH__


#include <cudaCompress/global.h>

#include <cudaCompress/util/Quantize.h> // for EQuantizeType


namespace cudaCompress {

namespace util {


__device__ inline uchar floatToByte(float val)
{
    return uchar(min(max(val + 128.5f, 0.0f), 255.0f));
}

__device__ inline float byteToFloat(uchar val)
{
    return float(val) - 128.0f;
}


// negative -> -1   non-negative -> 0
__device__ inline int getNegativeSign(short val)
{
    return (val >> 15);
}
__device__ inline int getNegativeSign(int val)
{
    return (val >> 31);
}

// negative -> -1   non-negative -> 1
__device__ inline float getSignAsFloat(int val)
{
    //if(val < 0) return -1.0f;
    //if(val > 0) return 1.0f;
    //return 0.0f;
    float t = val < 0 ? -1.0f : 0.0f;
    return val > 0 ? 1.0f : t;
}
__device__ inline float getSign(float val)
{
    //if(val < 0.0f) return -1.0f;
    //if(val > 0.0f) return 1.0f;
    //return 0.0f;
    float t = val < 0.0f ? -1.0f : 0.0f;
    return val > 0.0f ? 1.0f : t;
}


__device__ inline uint symbolize(int value)
{
    // map >= 0 to even, < 0 to odd
    return 2 * abs(value) + getNegativeSign(value);
}

__device__ inline int unsymbolize(uint symbol)
{
    int negative = symbol % 2;
    // map even to >= 0, odd to < 0
    return (1 - 2 * negative) * ((symbol + negative) / 2);
}


template<EQuantizeType Q>
struct Quantize
{
    __device__ static inline int quantize(float value, float quantizationStepInv);
};

template<>
struct Quantize<QUANTIZE_DEADZONE>
{
    __device__ static inline int quantize(float value, float quantizationStepInv)
    {
        // round-to-zero -> deadzone quantization with twice larger zero bin
        return int(value * quantizationStepInv);
    }
};

template<>
struct Quantize<QUANTIZE_UNIFORM>
{
    __device__ static inline int quantize(float value, float quantizationStepInv)
    {
        // standard midtread quantization
        return int(value * quantizationStepInv + getSign(value) * 0.5f);
    }
};


template<EQuantizeType Q>
struct Unquantize
{
    __device__ static inline float unquantize(int quant, float quantizationStep);
};

template<>
struct Unquantize<QUANTIZE_DEADZONE>
{
    __device__ static inline float unquantize(int quant, float quantizationStep)
    {
        // deadzone quantization: unquantize into middle of bin
        return (float(quant) + getSignAsFloat(quant) * 0.5f) * quantizationStep;
    }
};

template<>
struct Unquantize<QUANTIZE_UNIFORM>
{
    __device__ static inline float unquantize(int quant, float quantizationStep)
    {
        // midtread quantization
        return float(quant) * quantizationStep;
    }
};



template<EQuantizeType Q>
__device__ inline int quantize(float value, float quantizationStepInv)
{
    return Quantize<Q>::quantize(value, quantizationStepInv);
}

template<EQuantizeType Q>
__device__ inline float unquantize(int quant, float quantizationStep)
{
    return Unquantize<Q>::unquantize(quant, quantizationStep);
}


template<EQuantizeType Q>
__device__ inline uint quantizeToSymbol(float value, float quantizationStepInv)
{
    return symbolize(quantize<Q>(value, quantizationStepInv));
}

template<EQuantizeType Q>
__device__ inline float unquantizeFromSymbol(uint symbol, float quantizationStep)
{
    return unquantize<Q>(unsymbolize(symbol), quantizationStep);
}


}

}


#endif
