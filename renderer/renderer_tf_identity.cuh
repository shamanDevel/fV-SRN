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
	struct TransferFunctionIdentityParameters
	{
#ifdef TRANSFER_FUNCTION_IDENTITY__BATCHED
		//batch, 0, C; C=0->absorption, C=1->emission
		Tensor3Read<real_t> scaleAbsorptionEmission;
#else
		real_t scaleAbsorption;
		real_t scaleEmission;
#endif
	};
}

__constant__ kernel::TransferFunctionIdentityParameters transferFunctionIdentityParameters;

namespace kernel
{
	struct TransferFunctionIdentity
	{

		__host__ __device__ __inline__ real4 eval(
			real_t density, real3 normal, real_t previousDensity, real_t stepsize, int batch) const
		{
			density = clamp(density, real_t(0), real_t(1));
#ifdef TRANSFER_FUNCTION_IDENTITY__BATCHED
			const real_t scaleAbsorption = 
				transferFunctionIdentityParameters.scaleAbsorptionEmission[batch][0][0];
			const real_t scaleColor = 
				transferFunctionIdentityParameters.scaleAbsorptionEmission[batch][0][1];
#else
			const real_t scaleAbsorption = transferFunctionIdentityParameters.scaleAbsorption;
			const real_t scaleEmission = transferFunctionIdentityParameters.scaleEmission;
#endif
			return make_real4(
				density * scaleEmission, //red
				density * scaleEmission, //green
				density * scaleEmission, //blue
				density * scaleAbsorption * stepsize); //absorption
		}

#if 0
		__host__ __device__ __inline__ void adjoint(
			real_t density, real3 normal, int batch
			const Tensor3Read<T>& tf, int batch, real_t density,
			const real4& adj_color, real_t& adj_density,
			BTensor3RW<T>& adj_tf, real_t* sharedData) const
		{
			real_t densityClamped = clamp(density, real_t(0), real_t(1));
			const real_t scaleAbsorption = tf[batch][0][0];
			const real_t scaleColor = tf[batch][0][1];

			real_t adj_densityClamped =
				dot(make_real3<T>(adj_color), make_real3<T>(scaleColor)) +
				adj_color.w * scaleAbsorption;
			if constexpr (HasTFDerivative)
			{
				real_t adj_scaleColor = dot(make_real3<T>(adj_color), make_real3<T>(densityClamped));
				real_t adj_scaleAbsorption = adj_color.w * densityClamped;
				if constexpr (DelayedAccumulation)
				{
					sharedData[0] += adj_scaleAbsorption;
					sharedData[1] += adj_scaleColor;
				}
				else {
					kernel::atomicAdd(&adj_tf[batch][0][0], adj_scaleAbsorption);
					kernel::atomicAdd(&adj_tf[batch][0][1], adj_scaleColor);
				}
			}

			adj_density = (density > 0 && density < 1) ? adj_densityClamped : 0;
		}
		__host__ __device__ __inline__ void adjointAccumulate(
			int batch, BTensor3RW<T>& adj_tf, real_t* sharedData) const
		{
			//TODO: optimize via warp reductions?
			kernel::atomicAdd(&adj_tf[batch][0][0], sharedData[0]);
			kernel::atomicAdd(&adj_tf[batch][0][1], sharedData[1]);
		}
		__host__ __device__ __inline__ void adjointInit(
			BTensor3RW<T>& adj_tf, real_t* sharedData) const
		{
			const int R = adj_tf.size(1);
			const int C = 2;
			for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
				sharedData[r * C + c] = 0;
		}

		template<int D, typename density_t>
		__host__ __device__ __inline__ auto evalForwardGradients(
			const Tensor3Read<T>& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, false> /*hasTFDerivative*/) const
		{
			using namespace cudAD;
			density = clamp(density, real_t(0), real_t(1));
			const real_t scaleAbsorption = tf[batch][0][0];
			const real_t scaleColor = tf[batch][0][1];
			return make_real4<T>(
				density * scaleColor, //red
				density * scaleColor, //green
				density * scaleColor, //blue
				density * scaleAbsorption); //absorption
		}
		template<int D, typename density_t>
		__host__ __device__ __inline__ cudAD::fvar<real4, D> evalForwardGradients(
			const Tensor3Read<T>& tf, int batch, density_t density,
			const ITensor3Read& d_tf,
			integral_constant<bool, true> /*hasTFDerivative*/) const
		{
			using namespace cudAD;
			density = clamp(density, real_t(0), real_t(1));
			const fvar<real_t, D> scaleAbsorption = fvar<real_t, D>::input(
				tf[batch][0][0], d_tf[batch][0][0]);
			const fvar<real_t, D> scaleColor = fvar<real_t, D>::input(
				tf[batch][0][1], d_tf[batch][0][1]);
			return make_real4<T>(
				density * scaleColor, //red
				density * scaleColor, //green
				density * scaleColor, //blue
				density * scaleAbsorption); //absorption
		}
#endif
	};
}
