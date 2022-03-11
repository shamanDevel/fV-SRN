#ifndef __functor_h__
#define __functor_h__


#include <cuda_runtime.h>


template <typename T>
class FunctorIdentity
{
public:
    __host__ __device__ inline T operator()(const T a) { return a; }
};


template <typename TIn, typename TOut>
class FunctorFlagTrue
{
public:
    __host__ __device__ inline TOut operator()(const TIn a) { return a ? 1 : 0; }
};


template <typename T>
class FunctorAbs
{
public:
    __host__ __device__ inline T operator()(const T a) { return abs(a); }
};



#endif
