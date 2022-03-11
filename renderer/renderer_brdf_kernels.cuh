#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

/**
 * Defines:
 * BRDF_T
 */

__global__ void EvaluateBRDF(
	dim3 virtual_size,
	kernel::Tensor2Read<real_t> rgbaInput,
	kernel::Tensor2Read<real_t> positionInput,
	kernel::Tensor2Read<real_t> gradientInput,
	kernel::Tensor2Read<real_t> rayDirInput,
	kernel::Tensor2RW<real_t> rgbaOutput)
{
	using BRDF_t = BRDF_T;
	BRDF_t brdf;

	//LOOP
	KERNEL_1D_LOOP(b, virtual_size)
	{
		real4 rgba = kernel::fetchReal4(rgbaInput, b);
		real3 position = kernel::fetchReal3(positionInput, b);
		real3 gradient = kernel::fetchReal3(gradientInput, b);
		real3 rayDir = kernel::fetchReal3(rayDirInput, b);

		int batch = 0;
		real4 color = brdf.eval(rgba, position, gradient, rayDir, batch);

		kernel::writeReal4(color, rgbaOutput, b);
	}
	KERNEL_3D_LOOP_END
}
