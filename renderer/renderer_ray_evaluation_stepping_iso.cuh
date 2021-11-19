#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

#include <forward_vector.h>
#include "renderer_cudad_bridge.cuh"
#include "renderer_adjoint.cuh"

/**
 * Defines:
 * RAY_EVALUATION_STEPPING__VOLUME_INTERPOLATION_T
 * RAY_EVALUATION_STEPPING__ISOVALUE_BATCHED
 */

namespace kernel
{
	struct RayEvaluationSteppingIsoParameters
	{
		//TODO: find a way to make this dependent on the position
		real_t stepsize;

#ifdef RAY_EVALUATION_STEPPING__ISOVALUE_BATCHED
		//batch -> isovalue
		Tensor1Read<real_t> isovalue;
#else
		real_t isovalue;
#endif
	};
}

__constant__ ::kernel::RayEvaluationSteppingIsoParameters rayEvaluationSteppingIsoParameters;

namespace kernel
{
	
	struct RayEvaluationSteppingIso
	{
		using VolumeInterpolation_t = RAY_EVALUATION_STEPPING__VOLUME_INTERPOLATION_T;

		VolumeInterpolation_t volume;
		//TODO: Shading
		
		__device__ __inline__
		RayEvaluationOutput eval(real3 rayStart, real3 rayDir, real_t tmax, int batch)
		{
			real_t tmin, tmax1;
			intersectionRayAABB(
				rayStart, rayDir, volume.getBoxMin(), volume.getBoxSize(),
				tmin, tmax1);
			tmin = rmax(tmin, 0);
			tmax = rmin(tmax1, tmax);

			//if (blockIdx.x * blockDim.x + threadIdx.x == 21486) {
			//	printf("ray: (%.4f, %.4f, %.4f)->(%.4f, %.4f, %.4f)\n",
			//		rayStart.x, rayStart.y, rayStart.z,
			//		rayDir.x, rayDir.y, rayDir.z);
			//	printf("box: (%.4f, %.4f, %.4f)--(%.4f, %.4f, %.4f)\n",
			//		volume.getBoxMin().x, volume.getBoxMin().y, volume.getBoxMin().z,
			//		volume.getBoxSize().x, volume.getBoxSize().y, volume.getBoxSize().z);
			//	printf("tmin=%.4f, tmax=%.4f\n", tmin, tmax);
			//}
			
			//stepping
			real_t stepsize = rayEvaluationSteppingIsoParameters.stepsize;
#ifdef RAY_EVALUATION_STEPPING__ISOVALUE_BATCHED
			const real_t isovalue = rayEvaluationSteppingIsoParameters.isovalue[batch];
#else
			const real_t isovalue = rayEvaluationSteppingIsoParameters.isovalue;
#endif
			
			real4 color = make_real4(0);
			real3 normal = make_real3(0);
			real_t depth = 0;
			int terminationIndex;
			for (terminationIndex = 0; ; ++terminationIndex)
			{
				real_t tcurrent = tmin + terminationIndex * stepsize;
				if (tcurrent > tmax) break;

				real3 position = rayStart + rayDir * tcurrent;
				const auto [density, isInside] = volume.eval<real_t>(position, rayDir, batch);

				//if (blockIdx.x * blockDim.x + threadIdx.x == 21486)
				//	printf("tcurrent=%.4f, position=(%.4f, %.4f, %.4f) -> density=%.4f\n",
				//		tcurrent, position.x, position.y, position.z, density);

				if (density > isovalue)
				{
					normal = volume.evalNormal(position, rayDir, batch);
					normal = normalize(normal);
					//TODO: shading
					color = make_real4(make_real3(dot(normal, rayDir)), 1.0);
					depth = tcurrent;
					break;
				}
			}
			
			return { color, normal, depth};
		}
	};
	
}
