#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

#include <forward_vector.h>
#include "renderer_cudad_bridge.cuh"
#include "renderer_adjoint.cuh"

#ifdef __CUDACC__
#include "cooperative_groups.cuh"
#endif
//#include <cooperative_groups/reduce.h>

/**
 * Defines:
 * TRANSFER_FUNCTION_TEXTURE__USE_TENSOR
 * TRANSFER_FUNCTION_TEXTURE__PREINTEGRATION_MODE
 */

namespace kernel
{
	struct TransferFunctionTextureParameters
	{
#ifdef TRANSFER_FUNCTION_TEXTURE__USE_TENSOR
		Tensor3Read<real_t> tensor;
#else
		cudaTextureObject_t tex;
		cudaTextureObject_t preintegrated;
#endif
	};
}

__constant__ kernel::TransferFunctionTextureParameters transferFunctionTextureParameters;

namespace kernel
{
	struct TransferFunctionTexture
	{

		__host__ __device__ __inline__ real4 eval(
			real_t density, real3 normal, real_t previousDensity, real_t stepsize, int batch) const
		{
			density = clamp(density, real_t(0), real_t(1));
#ifdef TRANSFER_FUNCTION_TEXTURE__USE_TENSOR
			const auto& tf = transferFunctionTextureParameters.tensor;
			const int R = tf.size(1);
			const real_t d = density * R - real_t(0.5);
			const int di = int(floorf(d));
			const real_t df = d - di;
			const real4 val0 = fetchReal4(tf, batch, clamp(di, 0, R - 1));
			const real4 val1 = fetchReal4(tf, batch, clamp(di + 1, 0, R - 1));
			real4 rgba = lerp(val0, val1, df);
			rgba.w *= stepsize;
#else
#if TRANSFER_FUNCTION_TEXTURE__PREINTEGRATION_MODE==0
			//normal stuff
			real4 rgba = make_real4(tex1D<float4>(
				transferFunctionTextureParameters.tex, static_cast<float>(density)));
			rgba.w *= stepsize;
#elif TRANSFER_FUNCTION_TEXTURE__PREINTEGRATION_MODE==1
			//1D-preintegration
			if (previousDensity < 0) previousDensity = density;
			real4 rgba;
			if (rabs(previousDensity-density)<1e-3)
			{
			    //fallback for constant density
				rgba = make_real4(tex1D<float4>(
					transferFunctionTextureParameters.tex, static_cast<float>(density)));
				rgba.w *= stepsize;
			}
			else {
				float4 Vsf = tex1D<float4>(transferFunctionTextureParameters.preintegrated, previousDensity);
				float4 Vsb = tex1D<float4>(transferFunctionTextureParameters.preintegrated, density);
				rgba = make_real4(
					make_real3(stepsize * (make_float3(Vsb) - make_float3(Vsf)) / (density - previousDensity)),
					static_cast<real_t>(1 - expf(-stepsize * (Vsb.w - Vsf.w) / (density - previousDensity)))
				);
				if (rgba.w > 1e-5) {
					rgba.x /= rgba.w; //premultiplication!
					rgba.y /= rgba.w;
					rgba.z /= rgba.w;
				}
			}
#else
			//2D-preintegration
			if (previousDensity < 0) previousDensity = density;
			real4 rgba = make_real4(tex2D<float4>(
				transferFunctionTextureParameters.preintegrated,
				static_cast<float>(previousDensity), static_cast<float>(density)));
			if (rgba.w > 1e-5) {
				rgba.x /= rgba.w; //premultiplication!
				rgba.y /= rgba.w;
				rgba.z /= rgba.w;
			}
#endif
#endif
			return rgba;
		}
		
	};
}
