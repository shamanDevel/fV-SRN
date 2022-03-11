#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

/**
 * Defines:
 * RAY_EVALUATION_STEPPING__SURFACE_FEATURE (an option of SURFACE_FEATURE)
 */

#define SURFACE_FEATURE_OFF 0
#define SURFACE_FEATURE_FIRST_PRINCIPAL_CURVATURE 1
#define SURFACE_FEATURE_SECOND_PRINCIPAL_CURVATURE 2
#define SURFACE_FEATURE_MEAN_CURVATURE 3
#define SURFACE_FEATURE_GAUSSIAN_CURVATURE 4

__global__ void RayEvaluationIsoShadingCurvature(
    dim3 virtual_size,
    kernel::Tensor2Read<real_t> normals,
    kernel::Tensor2Read<real_t> curvature,
    kernel::Tensor2Read<real_t> rayDirection,
    kernel::Tensor2RW<real_t> outputColor,
    real_t isocontourRange,
    cudaTextureObject_t isocontourTexture)
{
    //Copied from renderer_ray_evaluation_stepping_iso.cuh
    KERNEL_1D_LOOP(i, virtual_size)
    {
		//read values
		real3 normal = safeNormalize(kernel::fetchReal3(normals, i));
		real2 localCurvature = kernel::fetchReal2(curvature, i);
		real3 rayDir = kernel::fetchReal3(rayDirection, i);

#if RAY_EVALUATION_STEPPING__SURFACE_FEATURE == SURFACE_FEATURE_OFF
		real4 color = make_real4(1, 1, 1, 1);
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
		feature = (feature + isocontourRange) / (2 * isocontourRange);
		//query isocontour texture
		real4 color = make_real4(tex1D<float4>(
			isocontourTexture,
			feature));
#endif
		//shading
		color *= dot(normal, rayDir);
		color.w = 1;

		kernel::writeReal4(color, outputColor, i);
    }
    KERNEL_1D_LOOP_END
}