#pragma once

#include "renderer_commons.cuh"
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#ifndef SAMPLER_STATE_T
#define SAMPLER_STATE_T curandStateXORWOW_t
#endif

/*
TODO: cache sampler state per thread over the runs
*/

namespace kernel
{
	/**
	 * A sampler for random numbers.
	 * The sampler is special in that it is constructed once per thread only.
	 */
	struct Sampler
	{
		using State_t = SAMPLER_STATE_T;
		State_t state_;

		/**
		 * Constructs the sampler using the time.
		 * It also reads the current thread index for the sequence number.
		 */
		__device__ __inline__ Sampler(
			unsigned long long seed, unsigned long long time)
		{
			unsigned long long sequence =
				blockDim.x * gridDim.x * time +
				blockIdx.x * blockDim.x + threadIdx.x;
			curand_init(seed, sequence, 0, &state_);
		}

		//Samples a uniform value in (0,1]
		__device__ __inline__ real_t sampleUniform()
		{
#if KERNEL_DOUBLE_PRECISION==0
			return curand_uniform(&state_);
#else
			return curand_uniform_double(&state_);
#endif
		}

		//Samples a normally distributed value with mean=0 and standard deviation=1
		__device__ __inline__ real_t sampleNormal()
		{
#if KERNEL_DOUBLE_PRECISION==0
			return curand_normal(&state_);
#else
			return curand_normal_double(&state_);
#endif
		}
	};
}