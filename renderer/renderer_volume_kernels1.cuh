#pragma once

#include "renderer_commons.cuh"
#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
#include "renderer_sampler_curand.cuh"

/**
 * Defines:
 * VOLUME_INTERPOLATION_T
 * OUTPUT_CHANNELS
 * VOLUME_USE_DIRECTION
 */

__global__ void EvaluateNoBatches(
	dim3 virtual_size,
	kernel::Tensor2RW<real_t> positionsInput,
	kernel::Tensor2RW<real_t> directionsInput,
	kernel::Tensor2RW<real_t> densitiesOutput)
{
	using VolumeInterpolation_t = VOLUME_INTERPOLATION_T;
	VolumeInterpolation_t volume;

	//LOOP
#if KERNEL_SYNCHRONIZED_TRACING==1
	KERNEL_1D_LOOP_SYNC(b, valid, virtual_size)
#else
	KERNEL_1D_LOOP(b, virtual_size)
    static constexpr bool valid = true;
#endif
	{
		real3 position = valid ? make_real3(
			positionsInput[b][0],
			positionsInput[b][1],
			positionsInput[b][2]
		) : make_real3(0,0,0);
		constexpr const int batch = 0; //For now
#if VOLUME_USE_DIRECTION==1
		real3 direction = valid ? make_real3(
			directionsInput[b][0],
			directionsInput[b][1],
			directionsInput[b][2]
		) : make_real3(0, 0, 0);
#else
		const real3 direction = make_real3(0, 0, 0);
#endif
#if OUTPUT_CHANNELS==1
		real_t density = volume.eval<real_t>(position, direction, batch).value;
		if (valid)
		    densitiesOutput[b][0] = density;
#elif OUTPUT_CHANNELS==3
		real3 value = volume.eval<real3>(position, direction, batch).value;
		if (valid) {
			densitiesOutput[b][0] = value.x;
			densitiesOutput[b][1] = value.y;
			densitiesOutput[b][2] = value.z;
		}
#elif OUTPUT_CHANNELS==4
		real4 value = volume.eval<real4>(position, direction, batch).value;
		if (valid) {
			densitiesOutput[b][0] = value.x;
			densitiesOutput[b][1] = value.y;
			densitiesOutput[b][2] = value.z;
			densitiesOutput[b][3] = value.w;
		}
#else
#error "Unknown output channel count"
#endif
	}
	KERNEL_1D_LOOP_END
}
