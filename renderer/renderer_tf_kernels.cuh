#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

/**
 * Defines:
 * TRANSFER_FUNCTION_T
 */

__global__ void EvaluateTF(
	dim3 virtual_size,
	kernel::Tensor2Read<real_t> densitiesInput,
	kernel::Tensor2Read<real_t> /*previousDensitiesInput*/,
	kernel::Tensor2Read<real_t> /*gradientsInput*/,
	kernel::Tensor2RW<real_t> colorsOutput,
	real_t densityMin, real_t densityMax, real_t /*stepsize*/)
{
	using TF_t = TRANSFER_FUNCTION_T;
	TF_t tf;
	const real_t divDensityRange = real_t(1) / (densityMax - densityMin);

	//LOOP
	KERNEL_1D_LOOP(b, virtual_size)
	{
		real_t density = densitiesInput[b][0];
		real4 color = make_real4(0, 0, 0, 0);
		if (density >= densityMin) {
			auto density2 = (density - densityMin) * divDensityRange;
			color = tf.eval(density2, real3{ 0,0,0 }, { -1 }, { 1 }, 0);
		}
		colorsOutput[b][0] = color.x;
		colorsOutput[b][1] = color.y;
		colorsOutput[b][2] = color.z;
		colorsOutput[b][3] = color.w;
	}
	KERNEL_3D_LOOP_END
}

__global__ void EvaluateTFWithPrevious(
	dim3 virtual_size,
	kernel::Tensor2Read<real_t> densitiesInput,
	kernel::Tensor2Read<real_t> previousDensitiesInput,
	kernel::Tensor2Read<real_t> /*gradientsInput*/,
	kernel::Tensor2RW<real_t> colorsOutput,
	real_t densityMin, real_t densityMax, real_t stepsize)
{
	using TF_t = TRANSFER_FUNCTION_T;
	TF_t tf;
	const real_t divDensityRange = real_t(1) / (densityMax - densityMin);

	//LOOP
	KERNEL_1D_LOOP(b, virtual_size)
	{
		real_t density = densitiesInput[b][0];
		real_t prevDensity = previousDensitiesInput[b][0];
		real4 color = make_real4(0, 0, 0, 0);
		if (density >= densityMin) {
			auto density2 = (density - densityMin) * divDensityRange;
			auto prevDensity2 = prevDensity>=0 ? ((prevDensity - densityMin) * divDensityRange) : -1;
			color = tf.eval(density2, real3{ 0,0,0 }, { prevDensity2 }, { stepsize }, 0);
		}
		colorsOutput[b][0] = color.x;
		colorsOutput[b][1] = color.y;
		colorsOutput[b][2] = color.z;
		colorsOutput[b][3] = color.w;
	}
	KERNEL_3D_LOOP_END
}

__global__ void EvaluateTFWithGradient(
	dim3 virtual_size,
	kernel::Tensor2Read<real_t> densitiesInput,
	kernel::Tensor2Read<real_t> /*previousDensitiesInput*/,
	kernel::Tensor2Read<real_t> gradientsInput,
	kernel::Tensor2RW<real_t> colorsOutput,
	real_t densityMin, real_t densityMax, real_t /*stepsize*/)
{
	using TF_t = TRANSFER_FUNCTION_T;
	TF_t tf;
	const real_t divDensityRange = real_t(1) / (densityMax - densityMin);

	//LOOP
	KERNEL_1D_LOOP(b, virtual_size)
	{
		real_t density = densitiesInput[b][0];
		real3 gradient = make_real3(
			gradientsInput[b][0],
			gradientsInput[b][1],
			gradientsInput[b][2]
		);
		real4 color = make_real4(0, 0, 0, 0);
		if (density >= densityMin) {
			auto density2 = (density - densityMin) * divDensityRange;
			color = tf.eval(density2, gradient, { -1 }, { 1 }, 0);
		}
		colorsOutput[b][0] = color.x;
		colorsOutput[b][1] = color.y;
		colorsOutput[b][2] = color.z;
		colorsOutput[b][3] = color.w;
	}
	KERNEL_3D_LOOP_END
}

__global__ void EvaluateTFWithPreviousWithGradient(
	dim3 virtual_size,
	kernel::Tensor2Read<real_t> densitiesInput,
	kernel::Tensor2Read<real_t> previousDensitiesInput,
	kernel::Tensor2Read<real_t> gradientsInput,
	kernel::Tensor2RW<real_t> colorsOutput,
	real_t densityMin, real_t densityMax, real_t stepsize)
{
	using TF_t = TRANSFER_FUNCTION_T;
	TF_t tf;
	const real_t divDensityRange = real_t(1) / (densityMax - densityMin);

	//LOOP
	KERNEL_1D_LOOP(b, virtual_size)
	{
		real_t density = densitiesInput[b][0];
		real_t prevDensity = previousDensitiesInput[b][0];
		real3 gradient = make_real3(
			gradientsInput[b][0],
			gradientsInput[b][1],
			gradientsInput[b][2]
		);
		real4 color = make_real4(0, 0, 0, 0);
		if (density >= densityMin) {
			auto density2 = (density - densityMin) * divDensityRange;
			auto prevDensity2 = prevDensity >= 0 ? ((prevDensity - densityMin) * divDensityRange) : -1;
			color = tf.eval(density2, gradient, { prevDensity2 }, { stepsize }, 0);
		}
		colorsOutput[b][0] = color.x;
		colorsOutput[b][1] = color.y;
		colorsOutput[b][2] = color.z;
		colorsOutput[b][3] = color.w;
	}
	KERNEL_3D_LOOP_END
}
