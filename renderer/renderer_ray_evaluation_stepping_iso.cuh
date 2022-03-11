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
 * RAY_EVALUATION_STEPPING__SURFACE_FEATURE (an option of SURFACE_FEATURE)
 */

#define SURFACE_FEATURE_OFF 0
#define SURFACE_FEATURE_FIRST_PRINCIPAL_CURVATURE 1
#define SURFACE_FEATURE_SECOND_PRINCIPAL_CURVATURE 2
#define SURFACE_FEATURE_MEAN_CURVATURE 3
#define SURFACE_FEATURE_GAUSSIAN_CURVATURE 4
#define SURFACE_FEATURE_CURVATURE_TEXTURE 5

namespace kernel
{
	struct RayEvaluationSteppingIsoParameters
	{
		//TODO: find a way to make this dependent on the position
		real_t stepsize;
		int binarySearchSteps;

#ifdef RAY_EVALUATION_STEPPING__ISOVALUE_BATCHED
		//batch -> isovalue
		Tensor1Read<real_t> isovalue;
#else
		real_t isovalue;
#endif
		real_t isocontourRange;
		cudaTextureObject_t isocontourTexture;
	};
}

__constant__ ::kernel::RayEvaluationSteppingIsoParameters rayEvaluationSteppingIsoParameters;

namespace kernel
{
	
	struct RayEvaluationSteppingIso
	{
		using VolumeInterpolation_t = RAY_EVALUATION_STEPPING__VOLUME_INTERPOLATION_T;

		VolumeInterpolation_t volume;

		__device__ __inline__
		void evalPoint(const real3& position, const real3& rayDir, int batch, real_t isovalue, int isValid,
			           bool& isInside, real4& color, real3& normal)
		{
			const auto resultFromForward = volume.eval<real_t>(position, rayDir, batch);
			const auto density = resultFromForward.value;
			const int requireNormal = isValid && (density > isovalue);
			isInside = requireNormal;
			real3 localNormal = make_real3(0);
			real2 localCurvature = make_real2(0);

#if KERNEL_SYNCHRONIZED_TRACING==0
			if (requireNormal) {
				localNormal = volume.evalNormal(position, rayDir, resultFromForward, batch);
#if RAY_EVALUATION_STEPPING__SURFACE_FEATURE != SURFACE_FEATURE_OFF
				localCurvature = volume.evalCurvature(position, rayDir, resultFromForward, batch);
#endif
			}
#else
			//evaluate normal if *any* thread requests normals (TensorCores)
			if (__any_sync(0xffffffff, requireNormal)) {
				localNormal = volume.evalNormal(position, rayDir, resultFromForward, batch);
#if RAY_EVALUATION_STEPPING__SURFACE_FEATURE != SURFACE_FEATURE_OFF
				localCurvature = volume.evalCurvature(position, rayDir, resultFromForward, batch);
#endif
			}
#endif

			if (requireNormal)
			{
				normal = safeNormalize(localNormal);
				//color
#if RAY_EVALUATION_STEPPING__SURFACE_FEATURE == SURFACE_FEATURE_OFF
				color = make_real4(1, 1, 1, 1);
#elif RAY_EVALUATION_STEPPING__SURFACE_FEATURE == SURFACE_FEATURE_CURVATURE_TEXTURE
				//convert from [-range, +range] to [0,1]
				float range = rayEvaluationSteppingIsoParameters.isocontourRange;
				float texX = (localCurvature.x + range) / (2 * range);
				float texY = (-localCurvature.y + range) / (2 * range);
				color = make_real4(tex2D<float4>(
					rayEvaluationSteppingIsoParameters.isocontourTexture,
					texX, texY));
#else
				//switch curvature metric
				float feature;
#if RAY_EVALUATION_STEPPING__SURFACE_FEATURE == SURFACE_FEATURE_FIRST_PRINCIPAL_CURVATURE
				feature = localCurvature.x;
#elif RAY_EVALUATION_STEPPING__SURFACE_FEATURE == SURFACE_FEATURE_SECOND_PRINCIPAL_CURVATURE
				feature = localCurvature.y;
#elif RAY_EVALUATION_STEPPING__SURFACE_FEATURE == SURFACE_FEATURE_MEAN_CURVATURE
				feature = 0.5 * (localCurvature.x + localCurvature.y);
#elif RAY_EVALUATION_STEPPING__SURFACE_FEATURE == SURFACE_FEATURE_GAUSSIAN_CURVATURE
				feature = localCurvature.x * localCurvature.y;
#endif
				//convert from [-range, +range] to [0,1]
				float range = rayEvaluationSteppingIsoParameters.isocontourRange;
				feature = (feature + range) / (2 * range);
				//query isocontour texture
				color = make_real4(tex1D<float4>(
					rayEvaluationSteppingIsoParameters.isocontourTexture,
					feature));
#endif
				//shading
				color *= dot(normal, rayDir);
				color.w = 1;
			}
		}
		
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

			//Find first intersection
			real4 color = make_real4(0);
			real3 normal = make_real3(0);
			real_t depth = 0;
			int terminationIndex;
			int isValid = 1;
			int foundHit = 0;
			for (terminationIndex = 0; ; ++terminationIndex)
			{
				real_t tcurrent = tmin + terminationIndex * stepsize;
				isValid = (tcurrent <= tmax) ? isValid : 0;
#if KERNEL_SYNCHRONIZED_TRACING==0
				if (!isValid) break;
#else
				//break only if all threads in the warp are done
				if (!__any_sync(0xffffffff, isValid))
					break;
#endif

				real3 position = rayStart + rayDir * tcurrent;
				bool isInside = false;
				evalPoint(position, rayDir, batch, isovalue,
					isValid, isInside, color, normal);
				if (isInside)
				{
					depth = tcurrent;
					isValid = 0;
					foundHit = 1;
				}
			}

			//refine using binary search
			int numBinarySteps = rayEvaluationSteppingIsoParameters.binarySearchSteps;
#if KERNEL_SYNCHRONIZED_TRACING==0
			if (!foundHit) numBinarySteps = 0;
#else
			//refine only if at least one thread found an intersection
			if (!__any_sync(0xffffffff, foundHit))
				numBinarySteps = 0;
#endif
			real_t depthOutside = depth - stepsize;
			real_t depthInside = depth;
			for (int i=0; i<numBinarySteps; ++i)
			{
				real_t depthTest = 0.5 * (depthOutside + depthInside);
				real3 position = rayStart + rayDir * depthTest;
				bool isInside = false;
				evalPoint(position, rayDir, batch, isovalue,
					foundHit, isInside, color, normal);
				if (isInside)
				{
					depth = depthTest;
					depthInside = depthTest;
				}
				else
				{
					depthOutside = depthTest;
				}
			}
			
			return { color, normal, depth};
		}
	};
	
}
