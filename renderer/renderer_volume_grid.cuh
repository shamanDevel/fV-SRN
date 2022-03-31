#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
#include "helper_matrixmath.cuh"

#include <forward_vector.h>
#include "renderer_cudad_bridge.cuh"
#include "renderer_adjoint.cuh"

/*
 * Defines by the host:
 * VOLUME_INTERPOLATION_GRID__REQUIRES_NORMAL
 * VOLUME_INTERPOLATION_GRID__USE_TENSOR
 * VOLUME_INTERPOLATION_GRID__TENSOR_TYPE float or double
 * VOLUME_INTERPOLATION_GRID__INTERPOLATION = 0->nearest, 1->trilinear, 2->tricubic
 * VOLUME_INTERPOLATION_GRID__OBJECT_SPACE
 * VOLUME_INTERPOLATION_GRID__TEXTURE_TYPE
 * VOLUME_INTERPOLATION_GRID__TEXTURE_EXTRACTOR
 * VOLUME_INTERPOLATION_GRID__GRID_RESOLUTION_OLD_BEHAVIOR
 * VOLUME_INTERPOLATION_GRID__CURVATURE_FROM_GRID
 */

namespace kernel
{
	struct VolumeInterpolationGridParameters
	{
#ifdef VOLUME_INTERPOLATION_GRID__USE_TENSOR
		using input_t = VOLUME_INTERPOLATION_GRID__TENSOR_TYPE;
		Tensor4Read<input_t> tensor;
#else
		cudaTextureObject_t tex;
#endif
		int3 resolutionMinusOne; //resolution-1
		real3 boxMin;
		real3 boxSize;
		real3 normalStep;
		real3 normalScale;
	};
}

__constant__ kernel::VolumeInterpolationGridParameters volumeInterpolationGridParameters;

namespace kernel
{
	template<typename Value_t = real_t>
	struct VolumeGridOutput
	{
		Value_t value;
		bool isInside;
#ifdef VOLUME_INTERPOLATION_GRID__CURVATURE_FROM_GRID
		real2 curvature;
#endif
	};

	struct VolumeInterpolationGrid
	{
		__host__ __device__ __forceinline__
		real3 getBoxMin() const
		{
			return volumeInterpolationGridParameters.boxMin;
		}
		__host__ __device__ __forceinline__
		real3 getBoxSize() const
		{
			return volumeInterpolationGridParameters.boxSize;
		}
		__host__ __device__ __forceinline__
	    int3 getResolution() const
		{
			return volumeInterpolationGridParameters.resolutionMinusOne + make_int3(1);
		}
		using density_t = real_t;

		//Component accessors
		__host__ __device__ __forceinline__ real_t getDirect(float v) { return v; }
		__host__ __device__ __forceinline__ real_t getX(float4 v) { return v.x; }
		__host__ __device__ __forceinline__ real_t getY(float4 v) { return v.y; }
		__host__ __device__ __forceinline__ real_t getZ(float4 v) { return v.z; }
		__host__ __device__ __forceinline__ real_t getW(float4 v) { return v.w; }
		__host__ __device__ __forceinline__ real_t getMagnitude(float4 v) { return length(v); }
		__host__ __device__ __forceinline__ real3 getVelocity(float4 v)
	        { return make_real3(v.x, v.y, v.z); }
		__host__ __device__ __forceinline__ real4 getColor(float4 v) { return make_real4(v); }
		
		__device__ __inline__ auto sampleNearest(int3 posObject, int batch)
		{
#ifdef VOLUME_INTERPOLATION_GRID__USE_TENSOR
			const auto& t = volumeInterpolationGridParameters.tensor;
			const int3& volumeSize2 = volumeInterpolationGridParameters.resolutionMinusOne;
			int3 ipos = clamp(posObject, make_int3(0), volumeSize2);
			auto value = t[batch][ipos.x][ipos.y][ipos.z];
			//TODO: feature extractor, currently only scalar densities supported
		    return value; 
#else
			auto v = tex3D<VOLUME_INTERPOLATION_GRID__TEXTURE_TYPE>(volumeInterpolationGridParameters.tex, posObject.x, posObject.y, posObject.z);
			return VOLUME_INTERPOLATION_GRID__TEXTURE_EXTRACTOR(v);
#endif
		}
		__device__ __inline__ auto sampleLinear(real3 posObject, int batch)
		{
#ifdef VOLUME_INTERPOLATION_GRID__USE_TENSOR
			const auto& t = volumeInterpolationGridParameters.tensor;
			const int3& volumeSize2 = volumeInterpolationGridParameters.resolutionMinusOne;

			int3 ipos = make_int3(posObject);
			real_t densities[8];
			int3 iposL = clamp(ipos, make_int3(0), volumeSize2);
			int3 iposH = clamp(ipos + make_int3(1), make_int3(0), volumeSize2);
			densities[0b000] = t[batch][iposL.x][iposL.y][iposL.z];
			densities[0b001] = t[batch][iposL.x][iposL.y][iposH.z];
			densities[0b010] = t[batch][iposL.x][iposH.y][iposL.z];
			densities[0b011] = t[batch][iposL.x][iposH.y][iposH.z];
			densities[0b100] = t[batch][iposH.x][iposL.y][iposL.z];
			densities[0b101] = t[batch][iposH.x][iposL.y][iposH.z];
			densities[0b110] = t[batch][iposH.x][iposH.y][iposL.z];
			densities[0b111] = t[batch][iposH.x][iposH.y][iposH.z];
			auto fpos = posObject - make_real3(ipos);
			real_t value = lerp(
				lerp(
					lerp(densities[0b000], densities[0b100], fpos.x),
					lerp(densities[0b010], densities[0b110], fpos.x),
					fpos.y),
				lerp(
					lerp(densities[0b001], densities[0b101], fpos.x),
					lerp(densities[0b011], densities[0b111], fpos.x),
					fpos.y),
				fpos.z);
			//TODO: feature extractor, currently only scalar densities supported
			return value;
#else
			auto v = tex3D<VOLUME_INTERPOLATION_GRID__TEXTURE_TYPE>(volumeInterpolationGridParameters.tex, posObject.x, posObject.y, posObject.z);
			return VOLUME_INTERPOLATION_GRID__TEXTURE_EXTRACTOR(v);
#endif
		}

		//Source: https://github.com/DannyRuijters/CubicInterpolationCUDA
		// Inline calculation of the bspline convolution weights, without conditional statements
		template<class T> __device__ __inline__ void bspline_weights(T fraction, T& w0, T& w1, T& w2, T& w3)
		{
			const T one_frac = 1.0f - fraction;
			const T squared = fraction * fraction;
			const T one_sqd = one_frac * one_frac;

			w0 = 1.0f / 6.0f * one_sqd * one_frac;
			w1 = 2.0f / 3.0f - 0.5f * squared * (2.0f - fraction);
			w2 = 2.0f / 3.0f - 0.5f * one_sqd * (2.0f - one_frac);
			w3 = 1.0f / 6.0f * squared * fraction;
		}
		//Source: https://github.com/DannyRuijters/CubicInterpolationCUDA
		__device__ __inline__ auto sampleCubic(real3 coord, int batch)
		{
			// shift the coordinate from [0,extent] to [-0.5, extent-0.5]
			const real3 coord_grid = coord - 0.5f;
			const real3 index = floor(coord_grid);
			const real3 fraction = coord_grid - index;
			float3 w0, w1, w2, w3;
			bspline_weights(fraction, w0, w1, w2, w3);

			const float3 g0 = w0 + w1;
			const float3 g1 = w2 + w3;
			const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
			const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

			// fetch the eight linear interpolations
			// weighting and fetching is interleaved for performance and stability reasons
			auto tex000 = sampleLinear(make_real3(h0.x, h0.y, h0.z), batch);
			auto tex100 = sampleLinear(make_real3(h1.x, h0.y, h0.z), batch);
			tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
			auto tex010 = sampleLinear(make_real3(h0.x, h1.y, h0.z), batch);
			auto tex110 = sampleLinear(make_real3(h1.x, h1.y, h0.z), batch);
			tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
			tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
			auto tex001 = sampleLinear(make_real3(h0.x, h0.y, h1.z), batch);
			auto tex101 = sampleLinear(make_real3(h1.x, h0.y, h1.z), batch);
			tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
			auto tex011 = sampleLinear(make_real3(h0.x, h1.y, h1.z), batch);
			auto tex111 = sampleLinear(make_real3(h1.x, h1.y, h1.z), batch);
			tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
			tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

			return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
		}
		
		__device__ __inline__ auto sample(real3 posObject, int batch)
		{
#if VOLUME_INTERPOLATION_GRID__INTERPOLATION==0
			return sampleNearest(make_int3(round(posObject)), batch);
#elif VOLUME_INTERPOLATION_GRID__INTERPOLATION==1
			return sampleLinear(posObject, batch);
#else
			return sampleCubic(posObject, batch);
#endif
		}

		template<typename Value_t>
		__device__ __inline__ VolumeGridOutput<Value_t> eval(real3 position, real3 direction, int batch)
		{
			//the texture is adressed in unnormalized coordinates
#ifndef VOLUME_INTERPOLATION_GRID__OBJECT_SPACE
			//transform from [boxMin, boxMax] to [0, res]
#if VOLUME_INTERPOLATION_GRID__GRID_RESOLUTION_OLD_BEHAVIOR==1
			auto scale = volumeInterpolationGridParameters.resolutionMinusOne;
#else
			auto scale = volumeInterpolationGridParameters.resolutionMinusOne + 1;
#endif
			position = (position - volumeInterpolationGridParameters.boxMin) /
				volumeInterpolationGridParameters.boxSize * make_real3(scale);
#endif

			////DEBUG
			//if (blockIdx.x == 0 && threadIdx.x < 4)
			//{
			//	printf("[%d] position: %.3f, %.3f, %.3f\n", threadIdx.x, position.x, position.y, position.z);
			//}

			bool isInside = all(position >= make_real3(0)) &&
				all(position <= make_real3(volumeInterpolationGridParameters.resolutionMinusOne));

			//now perform the interpolation
			auto value = sample(position, batch);
#ifdef VOLUME_INTERPOLATION_GRID__CURVATURE_FROM_GRID
			real_t density = value.x;
			real2 curvature = make_real2(value.y, value.z);
			return VolumeGridOutput<Value_t>{ density, isInside, curvature };
#else
			static_assert(is_same_v<decltype(value), Value_t>,
				"Requested a different return type than the one the TextureExtractur returned");

			return VolumeGridOutput<Value_t>{ value, isInside };
#endif
		}

		__device__ __inline__ real3 evalNormalImpl(real3 position, int batch)
		{
			//the texture is adressed in unnormalized coordinates
#ifndef VOLUME_INTERPOLATION_GRID__OBJECT_SPACE
			//transform from [boxMin, boxMax] to [0, res]
#if VOLUME_INTERPOLATION_GRID__GRID_RESOLUTION_OLD_BEHAVIOR==1
			auto scale = volumeInterpolationGridParameters.resolutionMinusOne;
#else
			auto scale = volumeInterpolationGridParameters.resolutionMinusOne + 1;
#endif
			position = (position - volumeInterpolationGridParameters.boxMin) /
				volumeInterpolationGridParameters.boxSize * make_real3(scale);
#endif

			//compute normal
			const real3 normalScale = volumeInterpolationGridParameters.normalScale;
			const real3 normalStep = volumeInterpolationGridParameters.normalStep;
			real3 normal = make_real3(0);
#ifdef VOLUME_INTERPOLATION_GRID__REQUIRES_NORMAL
#ifdef VOLUME_INTERPOLATION_GRID__CURVATURE_FROM_GRID
			normal.x = normalScale.x * (
				sample(position + make_real3(normalStep.x, 0, 0), batch).x -
				sample(position - make_real3(normalStep.x, 0, 0), batch).x
				);
			normal.y = normalScale.y * (
				sample(position + make_real3(0, normalStep.y, 0), batch).x -
				sample(position - make_real3(0, normalStep.y, 0), batch).x
				);
			normal.z = normalScale.z * (
				sample(position + make_real3(0, 0, normalStep.z), batch).x -
				sample(position - make_real3(0, 0, normalStep.z), batch).x
				);
#else
			normal.x = normalScale.x * (
				sample(position + make_real3(normalStep.x, 0, 0), batch) -
				sample(position - make_real3(normalStep.x, 0, 0), batch)
				);
			normal.y = normalScale.y * (
				sample(position + make_real3(0, normalStep.y, 0), batch) -
				sample(position - make_real3(0, normalStep.y, 0), batch)
				);
			normal.z = normalScale.z * (
				sample(position + make_real3(0, 0, normalStep.z), batch) -
				sample(position - make_real3(0, 0, normalStep.z), batch)
				);
#endif
#endif

			//done
			return normal;
		}

	    template<typename Value_t>
	    __device__ __inline__ real3 evalNormal(real3 position, real3 direction,
		    const VolumeGridOutput<Value_t>& resultFromEval, int batch)
	    {
		    return evalNormalImpl(position, batch);
	    }

	    template<typename Value_t>
	    __device__ __inline__ real2 evalCurvature(real3 position, real3 direction,
		    const VolumeGridOutput<Value_t>& resultFromEval, int batch)
	    {
#ifdef VOLUME_INTERPOLATION_GRID__CURVATURE_FROM_GRID
			real2 curvature = resultFromEval.curvature;
			return curvature; //TODO: scale?
#else
		    // 1. measure gradient
		    real3 g = evalNormalImpl(position, batch);
		    real_t gNorm = rmax(length(g), real_t(1e-7));
		    real3 n = -g / gNorm;
		    real3x3 P = real3x3::Identity() - real3x3::OuterProduct(n, n);

		    // 2. measure Hessian matrix
#if VOLUME_INTERPOLATION_GRID__GRID_RESOLUTION_OLD_BEHAVIOR==1
			auto scale = volumeInterpolationGridParameters.resolutionMinusOne;
#else
			auto scale = volumeInterpolationGridParameters.resolutionMinusOne + 1;
#endif
			real3 h = volumeInterpolationGridParameters.normalStep * 
				volumeInterpolationGridParameters.boxSize / make_real3(scale);

		    real3 denom = 1 / (2 * h);
		    real3x3 Hprime = real3x3::FromColumns(
			    denom.x * (evalNormalImpl(position + make_real3(h.x, 0, 0), batch) - evalNormalImpl(position - make_real3(h.x, 0, 0), batch)),
			    denom.y * (evalNormalImpl(position + make_real3(0, h.y, 0), batch) - evalNormalImpl(position - make_real3(0, h.y, 0), batch)),
			    denom.z * (evalNormalImpl(position + make_real3(0, 0, h.z), batch) - evalNormalImpl(position - make_real3(0, 0, h.z), batch))
		    );
		    real3x3 H = 0.5 * (Hprime + Hprime.transpose()); //make symmetric
		    real3x3 G = (-1 / gNorm) * P.matmul(H.matmul(P));

		    // 3. extract curvature 
		    real_t T = G.trace();
		    real_t F = G.frobenius();
		    real_t discr = sqrtr(2 * F * F - T * T);
		    real_t k1 = 0.5 * (T + discr);
		    real_t k2 = 0.5 * (T - discr);
		    if (isnan(k1) || isnan(k2))
			    k1 = k2 = real_t(0); //degenerated point
		    return make_real2(k1, k2);
#endif
	    }
	};
}