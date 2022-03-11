#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

#include <forward_vector.h>
#include "renderer_cudad_bridge.cuh"
#include "renderer_adjoint.cuh"

/**
 * Defines:
 * RAY_EVALUATION_STEPPING__VOLUME_INTERPOLATION_T
 * RAY_EVALUATION_STEPPING__TRANSFER_FUNCTION_T
 * RAY_EVALUATION_STEPPING__SKIP_TRANSFER_FUNCTION
 * RAY_EVALUATION_STEPPING__BLENDING_T
 * RAY_EVALUATION_STEPPING__BRDF_T
 * RAY_EVALUATION_STEPPING__ENABLE_EARLY_OUT
 */

namespace kernel
{
	struct RayEvaluationSteppingDvrParameters
	{
		//TODO: find a way to make this dependent on the position
		real_t stepsize;
		real_t alphaEarlyOut;
		real_t densityMin;
		real_t densityMax;
	};
}

__constant__::kernel::RayEvaluationSteppingDvrParameters rayEvaluationSteppingDvrParameters;

namespace kernel
{
	struct RayEvaluationSteppingDvr
	{
		using VolumeInterpolation_t = RAY_EVALUATION_STEPPING__VOLUME_INTERPOLATION_T;
		using TF_t = RAY_EVALUATION_STEPPING__TRANSFER_FUNCTION_T;
		using Blending_t = RAY_EVALUATION_STEPPING__BLENDING_T;
		using BRDF_t = RAY_EVALUATION_STEPPING__BRDF_T;

		VolumeInterpolation_t volume;
		TF_t tf;
		Blending_t blending;
		BRDF_t brdf;

		__device__ __inline__
		RayEvaluationOutput eval(real3 rayStart, real3 rayDir, real_t tmax, int batch)
		{
			real_t tmin, tmax1;
			intersectionRayAABB(
				rayStart, rayDir, volume.getBoxMin(), volume.getBoxSize(),
				tmin, tmax1);

			////DEBUG
			//if (blockIdx.x == 0 && threadIdx.x == 0)
			//{
			//	printf("box min=(%.3f, %.3f, %.3f), size=(%.3f, %.3f, %.3f), start=(%.3f, %.3f, %.3f), dir=(%.3f, %.3f, %.3f)\n",
			//		volume.getBoxMin().x, volume.getBoxMin().y, volume.getBoxMin().z,
			//		volume.getBoxSize().x, volume.getBoxSize().y, volume.getBoxSize().z,
			//		rayStart.x, rayStart.y, rayStart.z,
			//		rayDir.x, rayDir.y, rayDir.z);
			//}

			tmin = rmax(tmin, 0);
			tmax = rmin(tmax1, tmax);

			//stepping
			const real_t stepsize = rayEvaluationSteppingDvrParameters.stepsize;
			const real_t alphaEarlyOut = rayEvaluationSteppingDvrParameters.alphaEarlyOut;
			const real_t densityMin = rayEvaluationSteppingDvrParameters.densityMin;
			const real_t densityMax = rayEvaluationSteppingDvrParameters.densityMax;
			const real_t divDensityRange = real_t(1) / (densityMax - densityMin);

			RayEvaluationOutput output{
				make_real4(0),
				make_real3(0),
				0.0
			};
			VolumeInterpolation_t::density_t previousDensity = { -1 };

			int terminationIndex;
			for (terminationIndex = 0; ; ++terminationIndex)
			{
				real_t tcurrent = tmin + terminationIndex * stepsize;
#if RAY_EVALUATION_STEPPING__ENABLE_EARLY_OUT==1
				const int isValid = (tcurrent <= tmax) && (output.color.w < alphaEarlyOut);
#else
				const int isValid = (tcurrent <= tmax);
#endif

#if KERNEL_SYNCHRONIZED_TRACING==0
				if (!isValid) break;
#else
                //break only if all threads in the warp are done
				if (!__any_sync(0xffffffff, isValid))
					break;
#endif

				//sample volume
				real3 position = rayStart + rayDir * tcurrent;

#if RAY_EVALUATION_STEPPING__SKIP_TRANSFER_FUNCTION==1
				//Volume directly outputs the color
				const auto resultFromForward = volume.eval<real4>(position, rayDir, batch);
				auto color1 = resultFromForward.value;
				color1.w *= stepsize; //Usually done in the TF, for color-fields, do it manually
				auto n = volume.evalNormal(position, rayDir, batch);
#else
				//Volume stores the densities
				const auto resultFromForward = volume.eval<real_t>(position, rayDir, batch);
				const auto value = resultFromForward.value;
				const bool isInside = resultFromForward.isInside;
				auto density2 = (value - densityMin) * divDensityRange;
				auto color1 = make_real4(0);
				real3 n = make_real3(0);

				//compute normal
				const int requireNormal = isValid && (value >= densityMin);
#if KERNEL_SYNCHRONIZED_TRACING==0
				if (requireNormal)
					n = volume.evalNormal(position, rayDir, resultFromForward, batch);
#else
				//evaluate normal if *any* thread requests normals (TensorCores)
				if (__any_sync(0xffffffff, requireNormal))
					n = volume.evalNormal(position, rayDir, resultFromForward, batch);
#endif

				if (requireNormal)
				{
					//evaluate TF
					color1 = tf.eval(density2, n, previousDensity, stepsize, batch);
				}
				previousDensity = density2;
#endif

				//BRDF and blending
				if (color1.w > 0)
				{
					//BRDF
					auto color2 = brdf.eval(color1, position, n, rayDir, batch);

					//blending
					n = safeNormalize(n);
					RayEvaluationOutput newContribution = {
						color2,
						n,
						tcurrent
					};
					if (isValid)
					    output = blending.eval(output, newContribution);
				}
			}

			return output;
		}
	};
}
