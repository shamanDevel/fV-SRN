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
 * TRANSFER_FUNCTION_GAUSSIAN__SCALE_WITH_GRADIENT
 * TRANSFER_FUNCTION_GAUSSIAN__ANALYTIC
 */

namespace kernel
{
	struct TransferFunctionGaussianParameters
	{
		Tensor3Read<real_t> tensor;
	};
}

__constant__ kernel::TransferFunctionGaussianParameters transferFunctionGaussianParameters;

namespace kernel
{	
	struct TransferFunctionGaussian
	{
		template<typename D, typename M, typename S>
		static __host__ __device__ __inline__ auto gaussian(const D& d, const M& mu, const S& sigma)
		{
			using namespace cudAD;
			return exp(-(d - mu) * (d - mu) / (sigma * sigma));
		}

		static constexpr real_t SQRT_PI_2 = 0.8862269254527580136490837416705725913987747280611935641069038949; //sqrt(pi)/2
		static __host__ __device__ __inline__ real4 sampleTF(
			real_t density, real3 normal, const Tensor3Read<real_t>& tf, int batch, 
			real_t previousDensity, real_t stepsize)
		{
			const int R = tf.size(1);
			real4 c = make_real4(0);
			for (int i = 0; i < R; ++i)
			{
				real4 ci = fetchReal4(tf, batch, i);
				real_t mu = tf[batch][i][4];
				real_t sigma = tf[batch][i][5];
#ifdef TRANSFER_FUNCTION_GAUSSIAN__SCALE_WITH_GRADIENT
				sigma *= rmax(real_t(1e-5), length(normal)*real_t(0.1)); //some arbitrary scaling factor...
#endif
#if TRANSFER_FUNCTION_GAUSSIAN__ANALYTIC==0
				real_t ni = gaussian(density, mu, sigma);
#else
				real_t ni;
				if (previousDensity<0 || previousDensity==density)
				{
				    //special case of a homogeneous region
					ni = gaussian(density, mu, sigma);
				} else
				{
				    //piecewise analytic integration
					//assume constant color per segment
					ni = SQRT_PI_2 / (previousDensity - density) * sigma * (
						erf((previousDensity - mu) / sigma) + erf((mu - density) / sigma));
				}
#endif
				c += ci * ni;
			}
			c.w *= stepsize;
			return c;
		}

		__device__ __inline__ real4 eval(
			real_t density, real3 normal, real_t previousDensity, real_t stepsize, int batch) const
		{
			density = clamp(density, real_t(0), real_t(1));
			const auto& tf = transferFunctionGaussianParameters.tensor;
			real4 rgba = sampleTF(density, normal, tf, batch, previousDensity, stepsize);
			return rgba;
		}

	};
}
