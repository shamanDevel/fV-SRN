#pragma once

#include "renderer_commons.cuh"
#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
#include "renderer_sampler_curand.cuh"
#include "renderer_histogram.cuh"

/**
 * Defines:
 * VOLUME_INTERPOLATION_T
 */

#ifndef VOLUME_INTERPOLATION_T
//IntelliSense
#include "renderer_volume_grid.cuh"
#define VOLUME_INTERPOLATION_T kernel::VolumeInterpolationGrid
#endif

#ifndef FLT_MAX
#define FLT_MAX          3.402823466e+38F
#endif

/**
 * Extracts the minimal and maximal value, only one block!
 * The volume interpolation type must be VolumeInterpolationGrid
 */
__global__ void HistogramExtractKernel(
    kernel::VolumeHistogram* histogramOut)
{
    assert(gridDim.x == 1); //only one block!

    using VolumeInterpolation_t = VOLUME_INTERPOLATION_T;
    VolumeInterpolation_t volume;

    int3 resolution = volume.getResolution();
    ptrdiff_t numVoxels = ptrdiff_t(resolution.x) * resolution.y * resolution.z;

    //STAGE 1: FIND MIN+MAX
    //per thread reduction
    real_t threadMin = +FLT_MAX;
    real_t threadMax = -FLT_MAX;
    for (ptrdiff_t __i = threadIdx.x; __i<numVoxels; __i+=blockDim.x)
    {
        ptrdiff_t k = __i / (resolution.x * resolution.y);
        ptrdiff_t j = (__i - (k * resolution.x * resolution.y)) / resolution.x;
        ptrdiff_t i = __i - resolution.x * (j + resolution.y * k);
        //fetch the data
        real_t value = volume.sampleNearest(make_int3(i, j, k), 0);
        threadMin = fminf(threadMin, value);
        threadMax = fmaxf(threadMax, value);
    }
    //warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        int tmpMin = __shfl_down_sync(0xffffffff, threadMin, offset);
        if (tmpMin < threadMin) threadMin = tmpMin;
        
        int tmpMax = __shfl_down_sync(0xffffffff, threadMax, offset);
        if (tmpMax > threadMax) threadMax = tmpMax;
    }
    //block reduction
    static __shared__ float blockMin[32], blockMax[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    if (lane == 0)
    {
        blockMin[wid] = threadMin;
        blockMax[wid] = threadMax;
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        //first thread performs the final reduction
        int blocks = blockDim.x / warpSize;
        for (int i=1; i<blocks; ++i)
        {
            threadMin = fminf(threadMin, blockMin[i]);
            threadMax = fmaxf(threadMax, blockMax[i]);
        }
        //write out
        blockMin[0] = threadMin;
        blockMax[0] = threadMax;
        histogramOut->minDensity = threadMin;
        histogramOut->maxDensity = threadMax;
        printf("Histogram: min=%.4f, max=%.4f\n", threadMin, threadMax);
    }
    __syncthreads();
    const float globalMin = blockMin[0];
    const float globalMax = blockMax[0];
    const float denom = 1.0f / fmaxf(1e-6f, globalMax - globalMin);

    //STAGE 2: HISTOGRAM with atomics
    for (ptrdiff_t __i = threadIdx.x; __i < numVoxels; __i += blockDim.x)
    {
        ptrdiff_t k = __i / (resolution.x * resolution.y);
        ptrdiff_t j = (__i - (k * resolution.x * resolution.y)) / resolution.x;
        ptrdiff_t i = __i - resolution.x * (j + resolution.y * k);
        //fetch the data
        real_t value = volume.sampleNearest(make_int3(i, j, k), 0);
        //compute bucket
        static constexpr int numBinsMinus1 = kernel::VolumeHistogram::NUM_BINS - 1;
        int binIdx = static_cast<int>(numBinsMinus1 * (value - globalMin) * denom);
        //assert(binIdx >= 0 && binIdx < kernel::VolumeHistogram::NUM_BINS);
        binIdx = clamp(binIdx, 0, numBinsMinus1);
        //and increment
        ::atomicAdd(&histogramOut->bins[binIdx], 1);
    }

    //STAGE 3: Compute max bin value
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int maxBinValue = 0;
        for (int i = 0; i < kernel::VolumeHistogram::NUM_BINS; ++i)
        {
            maxBinValue = max(maxBinValue, histogramOut->bins[i]);
        }
        histogramOut->maxBinValue = maxBinValue;
    }
}