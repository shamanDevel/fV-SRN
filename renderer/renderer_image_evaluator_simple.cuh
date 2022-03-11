#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

#include <forward_vector.h>
#include "renderer_cudad_bridge.cuh"
#include "renderer_adjoint.cuh"

/**
 * Defines:
 * IMAGE_EVALUATOR__CAMERA_T
 * IMAGE_EVALUATOR__RAY_EVALUATOR_T
 * IMAGE_EVALUATOR__REQUIRES_SAMPLER
 * IMAGE_EVALUATOR__SUPERSAMPLING
 * IMAGE_EVALUATOR__HAS_BACKGROUND_IMAGE
 */

#ifdef IMAGE_EVALUATOR__REQUIRES_SAMPLER
#include "renderer_sampler_curand.cuh"
#endif

#ifndef FLT_MAX
#define FLT_MAX          3.402823466e+38F
#endif

/**
 * Evaluates the image.
 *
 * Output is a BCHW tensor with the channels
 * 0,1,2: rgb
 * 3: alpha
 * 4,5,6: normal
 * 7: depth
 */
__global__ void ImageEvaluatorSimpleKernel(
	dim3 virtual_size, kernel::Tensor4RW<real_t> output, int samples, unsigned int time,
	kernel::Tensor4RW<real_t> backgroundImage)
{
	using Camera_t = IMAGE_EVALUATOR__CAMERA_T;
	using RayEvaluator_t = IMAGE_EVALUATOR__RAY_EVALUATOR_T;

	Camera_t camera;
	RayEvaluator_t rayEvaluator;
#ifdef IMAGE_EVALUATOR__REQUIRES_SAMPLER
	::kernel::Sampler sampler(42, time);
#endif

	//LOOP
#if KERNEL_SYNCHRONIZED_TRACING==1
	assert(virtual_size.x * virtual_size.y * virtual_size.z % 32 == 0);
#endif
	KERNEL_3D_LOOP(x, y, b, virtual_size)
	{
		kernel::RayEvaluationOutput out{
			make_real4(0),
			make_real3(0),
			real_t(0)
		};

		real_t tmax = FLT_MAX;
#ifdef IMAGE_EVALUATOR__HAS_BACKGROUND_IMAGE
		real4 backgroundColor = make_real4(
			backgroundImage[0][0][y][x],
			backgroundImage[0][1][y][x],
			backgroundImage[0][2][y][x],
			backgroundImage[0][3][y][x]);
		tmax = backgroundColor.w>0 ? backgroundImage[0][4][y][x] : FLT_MAX;
		//debug
		//if (backgroundColor.w > 0) printf("[%04d, %04d] d=%f\n", int(x), int(y), tmax);
#endif

		for (int i = 0; i < samples; ++i)
		{
#ifdef IMAGE_EVALUATOR__SUPERSAMPLING
			//stupid random sampling. Replace by Plastic or stratified
			real_t nx = x + sampler.sampleUniform() - real_t(0.5);
			real_t ny = y + sampler.sampleUniform() - real_t(0.5);
#else
			real_t nx = x;
			real_t ny = y;
#endif
			//compute normalized device coordinates in [-1,+1]^2
			real2 ndc = make_real2(
				2 * (nx + real_t(0.5)) / output.size(3) - 1,
				2 * (ny + real_t(0.5)) / output.size(2) - 1);
			//camera + ray evaluation
			const auto [rayStart, rayDir] = camera.eval(ndc, b);

			//if (x == 720 && y == 430)
			//	printf("[%03d, %03d] ray=(%.4f,%.4f,%.4f)-(%.4f,%.4f,%.4f)\n",
			//		int(x), int(y), rayStart.x, rayStart.y, rayStart.z, rayDir.x, rayDir.y, rayDir.z);
			
#ifdef IMAGE_EVALUATOR__REQUIRES_SAMPLER
			kernel::RayEvaluationOutput nout = rayEvaluator.eval(
				rayStart, rayDir, tmax, b, sampler);
#else
			kernel::RayEvaluationOutput nout = rayEvaluator.eval(
				rayStart, rayDir, tmax, b);
#endif

			out.color += nout.color; //pre-multiplied alpha?
			out.normal += nout.normal * nout.color.w;
			out.depth += nout.depth * nout.color.w;
		}
		//write output, but first normalize
		out.depth /= out.color.w;
		out.color /= samples;
		out.normal /= samples;
		//out.normal = safeNormalize(out.normal);

		//mix with background
#ifdef IMAGE_EVALUATOR__HAS_BACKGROUND_IMAGE
		out.color = blend(out.color, backgroundColor);
#endif

		output[b][0][y][x] = out.color.x;
		output[b][1][y][x] = out.color.y;
		output[b][2][y][x] = out.color.z;
		output[b][3][y][x] = out.color.w;
		output[b][4][y][x] = out.normal.x;
		output[b][5][y][x] = out.normal.y;
		output[b][6][y][x] = out.normal.z;
		output[b][7][y][x] = out.depth;
	}
	KERNEL_3D_LOOP_END
}
