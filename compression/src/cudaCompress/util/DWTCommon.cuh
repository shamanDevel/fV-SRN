#ifndef __TUM3D__DWT_COMMON_CUH__
#define __TUM3D__DWT_COMMON_CUH__


namespace cudaCompress {

namespace util {


// TYPE CONVERSIONS

template<typename TIn>
__device__ inline float toFloat(TIn value);
// not defined - only explicit specializations are allowed!
template<>
__device__ inline float toFloat<byte>(byte value)
{
    return float(value) - 128.0f;
}
template<>
__device__ inline float toFloat<float>(float value)
{
    // nop
    return value;
}

template<typename TOut>
__device__ inline TOut fromFloat(float value);
// not defined - only explicit specializations are allowed!
template<>
__device__ inline byte fromFloat<byte>(float value)
{
    return byte(min(max(value + 128.5f, 0.0f), 255.0f));
}
template<>
__device__ inline float fromFloat<float>(float value)
{
    // nop
    return value;
}


// INDEX MIRRORING

//__device__ inline int mirrorLeft(int index)
//{
//    return abs(index);
//}
//__device__ inline int mirrorRight(int index, int size)
//{
//    return (size-1) - abs((size-1) - index);
//}
// use macros (instead of inline functions) for nsight...
#define mirrorLeft(index) abs(index)
#define mirrorRight(index, size) (((size)-1) - abs(((size)-1) - (index)))

__device__ inline int getSign(int val)
{
    // negative ->  1  positive -> 0
    //return ((val & 0x80000000) >> 31);
    // negative -> -1  positive -> 0
    return (val >> 31);
}
__device__ inline int mirrorLeftRepeat(int index)
{
    return abs(index) + getSign(index);
    //if(index < 0) index = -1 - index;
    //return index;
}
__device__ inline int mirrorRightRepeat(int index, int size)
{
    if(index >= size) return 2*size - (index+1);
    return index;
    // this is slightly slower:
    //int sign = getSign(size - index - 1);
    //return 2 * sign * (index - size) + index + sign;
}


// HALO TESTS

template<int diff>
__device__ inline bool leftHaloTestImpl(int thread)
{
    return thread >= diff;
}
template<>
__device__ inline bool leftHaloTestImpl<0>(int thread)
{
    return true;
}
template<int blockSize, int filterOffset>
__device__ inline bool leftHaloTest(int thread)
{
    //if(blockSize == filterOffset) return true;
    //return thread >= blockSize - filterOffset;
    return leftHaloTestImpl<blockSize - filterOffset>(thread);
}

template<int diff, int filterOffset>
struct rightHaloTestImpl {
    static __device__ inline bool perform(int thread) {
        return thread < filterOffset;
    }
};
template<int filterOffset>
struct rightHaloTestImpl<0, filterOffset> {
    static __device__ inline bool perform(int thread) {
        return true;
    }
};
template<int blockSize, int filterOffset>
__device__ inline bool rightHaloTest(int thread)
{
    return rightHaloTestImpl<blockSize - filterOffset, filterOffset>::perform(thread);
}


// EVEN/ODD TESTS

__device__ inline bool isEven(int index)
{
    return index % 2 == 0;
}
__device__ inline bool isOdd(int index)
{
    return index % 2 == 1;
}


// INT DIVISION WITH ROUNDING

__device__ __host__ inline short div2(short val)
{
    val += (val >= 0) ? 1 : -1;
    return val / 2;
}
__device__ __host__ inline short div4(short val)
{
    val += (val >= 0) ? 2 : -2;
    return val / 4;
}


}

}


#endif
