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

namespace kernel
{
	struct TransferFunctionPiecewiseParameters
	{
		Tensor3Read<real_t> tensor;
	};
}

__constant__ kernel::TransferFunctionPiecewiseParameters transferFunctionPiecewiseParameters;

namespace kernel
{
	struct TransferFunctionPiecewise
	{
		template<typename T>
		static __host__ __device__ __inline__
			typename scalar_traits<T>::real4 sampleTF(
			T density, const Tensor3Read<T>& tf, int batch)
		{
			using v4 = typename scalar_traits<T>::real4;
			const int R = tf.size(1);
			//find control point interval
			int i;
			for (i = 0; i < R - 2; ++i)
				if (tf[batch][i + 1][4] > density) break;
			//fetch values
			const v4 val0 = fetchVector4<v4>(tf, batch, i);
			const v4 val1 = fetchVector4<v4>(tf, batch, i + 1);
			const T pos0 = tf[batch][i][4];
			const T pos1 = tf[batch][i + 1][4];
			//linear interpolation
			//density<=pos0 -> pos0 ELSE density>pos1 -> pos1 ELSE density
			density = clamp(density, pos0, pos1);
			const T frac = (density - pos0) / (pos1 - pos0);
			//val0 + frac * (val1 - val0) = (1-frac)*val0 + frac*val1
			v4 rgba = lerp(val0, val1, frac);
			return rgba;
		}

		__device__ __inline__ real4 eval(
			real_t density, real3 normal, real_t previousDensity, real_t stepsize, int batch) const
		{
			density = clamp(density, real_t(0), real_t(1));
			const auto& tf = transferFunctionPiecewiseParameters.tensor;
			real4 rgba = sampleTF(density, tf, batch);
			rgba.w *= stepsize;
			return rgba;
		}

	};
}
