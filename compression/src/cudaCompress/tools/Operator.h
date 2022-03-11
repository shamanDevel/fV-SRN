#ifndef __operator_h__
#define __operator_h__


#include <limits>
#include <cfloat>
#include <climits>

#include <cuda_runtime.h>


template <typename T>
class OperatorAdd
{
public:
    __host__ __device__ inline T operator()(const T a, const T b) { return a + b; }
    __host__ __device__ inline T identity() { return T(); }
};


template <typename T>
class OperatorMultiply
{
public:
    __host__ __device__ inline T operator()(const T a, const T b) { return a * b; }
    __host__ __device__ inline T identity() { return (T)1; }
};


template <typename T>
class OperatorMax
{
public:
    __host__ __device__ inline T operator() (const T a, const T b) const { return max(a, b); }
    __host__ __device__ inline T identity() const { return std::numeric_limits<T>::lowest(); }
    // can't use std::numeric_limits in device code, so some specializations are below...
};

template <>
__host__ __device__ inline char OperatorMax<char>::identity() const { return CHAR_MIN; }
template <>
__host__ __device__ inline unsigned char OperatorMax<unsigned char>::identity() const { return 0; }
template <>
__host__ __device__ inline short OperatorMax<short>::identity() const { return SHRT_MIN; }
template <>
__host__ __device__ inline unsigned short OperatorMax<unsigned short>::identity() const { return 0; }
template <>
__host__ __device__ inline int OperatorMax<int>::identity() const { return INT_MIN; }
template <>
__host__ __device__ inline unsigned int OperatorMax<unsigned int>::identity() const { return 0; }
template <>
__host__ __device__ inline float OperatorMax<float>::identity() const { return -FLT_MAX; }


template <typename T>
class OperatorMin
{
public:
    __host__ __device__ inline T operator() (const T a, const T b) const { return min(a, b); }
    __host__ __device__ inline T identity() const { return std::numeric_limits<T>::max(); }
    // can't use std::numeric_limits in device code, so some specializations are below...
};

template <>
__host__ __device__ inline char OperatorMin<char>::identity() const { return CHAR_MAX; }
template <>
__host__ __device__ inline unsigned char OperatorMin<unsigned char>::identity() const { return UCHAR_MAX; }
template <>
__host__ __device__ inline short OperatorMin<short>::identity() const { return SHRT_MAX; }
template <>
__host__ __device__ inline unsigned short OperatorMin<unsigned short>::identity() const { return USHRT_MAX; }
template <>
__host__ __device__ inline int OperatorMin<int>::identity() const { return INT_MAX; }
template <>
__host__ __device__ inline unsigned int OperatorMin<unsigned int>::identity() const { return UINT_MAX; }
template <>
__host__ __device__ inline float OperatorMin<float>::identity() const { return FLT_MAX; }


#endif
