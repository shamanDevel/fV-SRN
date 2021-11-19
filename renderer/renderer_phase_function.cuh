#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
#include "helper_math.cuh"

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559 // 2*pi
#endif

#ifndef M_1_4PI
#define M_1_4PI 0.079577471545947667884441881686257 // 1/4pi
#endif

/*
 * Defines:
 * PHASE_FUNCTION_HENYEY_GREENSTEIN_BATCHED
 */

namespace kernel
{
	struct PhaseFunctionHenyeyGreensteinParameters
	{
#ifdef PHASE_FUNCTION_HENYEY_GREENSTEIN_BATCHED
		Tensor1Read<real_t> g;
#else
		real_t g;
#endif
	};
}

__constant__ kernel::PhaseFunctionHenyeyGreensteinParameters phaseFunctionHenyeyGreensteinParameters;

#ifdef PHASE_FUNCTION_HENYEY_GREENSTEIN_BATCHED
#define FETCH(parameter) (phaseFunctionHenyeyGreensteinParameters.parameter[batch])
#else
#define FETCH(parameter) (phaseFunctionHenyeyGreensteinParameters.parameter)
#endif

namespace kernel
{
	struct PhaseFunctionHelpers
	{
		static __host__ __device__ __inline__
		real_t getCosAngle(const real3& dirIn, const real3& dirOut)
		{
			return dot(-dirIn, dirOut);
		}

		template<typename Sampler>
		static __host__ __device__ __inline__
		real3 directionFromAngle(const real3& dirIn, real_t sampledCosAngle, Sampler& sampler)
		{
			const real_t cosTheta = sampledCosAngle;
			const real_t sinTheta = sqrtr(rmax(real_t(0), 1 - cosTheta * cosTheta));
			const real_t phi = M_2PI* sampler.sampleUniform();

			//coordinate system from vector
			//https://www.pbr-book.org/3ed-2018/Geometry_and_Transformations/Vectors#CoordinateSystem
			const real3 v1 = -dirIn;
			real3 v2, v3;
			if (rabs(v1.x) > rabs(v1.y))
				v2 = make_real3(-v1.z, 0, v1.x) / sqrtr(v1.x * v1.x + v1.z * v1.z);
			else
				v2 = make_real3(0, v1.z, -v1.y) / sqrtr(v1.y * v1.y + v1.z * v1.z);
			v3 = cross(v1, v2);

			//https://www.pbr-book.org/3ed-2018/Color_and_Radiometry/Working_with_Radiometric_Integrals#SphericalDirection
			return sinTheta * std::cos(phi) * v2 +
				sinTheta * std::sin(phi) * v3 + cosTheta * v1;
		}
	};
	
	struct PhaseFunctionHenyeyGreenstein
	{
		//for unit tests:
		__host__ __device__ __inline__
		real_t probAngle(real_t cosTheta, real3 pos, int batch)
		{
			real_t g = FETCH(g);
			real_t denom = 1 + g * g + 2 * g * cosTheta;
			return M_1_4PI * (1 - g * g) / (denom * sqrtr(denom));
		}

		__host__ __device__ __inline__
		real_t prob(real3 dirIn, real3 dirOut, real3 pos, int batch)
		{
			real_t cosTheta = PhaseFunctionHelpers::getCosAngle(dirIn, dirOut);
			return probAngle(cosTheta, pos, batch);
		}

		//for unit tests:
		template<typename Sampler>
		__host__ __device__ __inline__
		real_t sampleAngle(real3 dirIn, real3 pos, Sampler& sampler, int batch)
		{
			real_t g = FETCH(g);
			real_t u = sampler.sampleUniform();
			real_t cosTheta;
			if (rabs(g) < 1e-3f)
				cosTheta = 1 - 2 * u;
			else {
				real_t sqrTerm = (1 - g * g) /
					(1 - g + 2 * g * u);
				cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
			}
			return -cosTheta;
		}
		
		template<typename Sampler>
		__host__ __device__ __inline__
		real3 sample(real3 dirIn, real3 pos, Sampler& sampler, int batch)
		{
			real_t cosTheta = sampleAngle(dirIn, pos, sampler, batch);
			return PhaseFunctionHelpers::directionFromAngle(dirIn, cosTheta, sampler);
		}
	};

	struct PhaseFunctionRayleigh
	{
		//for unit tests:
		__host__ __device__ __inline__
		real_t probAngle(real_t cosTheta, real3 pos, int batch)
		{
			return real_t(M_1_4PI * 0.75) * (1 + cosTheta * cosTheta);
		}
		
		__host__ __device__ __inline__
		real_t prob(real3 dirIn, real3 dirOut, real3 pos, int batch)
		{
			real_t cosTheta = PhaseFunctionHelpers::getCosAngle(dirIn, dirOut);
			return probAngle(cosTheta, pos, batch);
		}

		//for unit tests:
		template<typename Sampler>
		__host__ __device__ __inline__
		real_t sampleAngle(real3 dirIn, real3 pos, Sampler& sampler, int batch)
		{
			real_t z = 4 * sampler.sampleUniform() - 2;
			real_t z2 = sqrtr(z * z + 1);
			real_t A = cbrtr(z + z2);
			real_t B = cbrtr(z - z2);
			real_t cosTheta = A + B;
			return cosTheta;
		}

		template<typename Sampler>
		__host__ __device__ __inline__
		real3 sample(real3 dirIn, real3 pos, Sampler& sampler, int batch)
		{
			real_t cosTheta = sampleAngle(dirIn, pos, sampler, batch);
			return PhaseFunctionHelpers::directionFromAngle(dirIn, cosTheta, sampler);
		}
	};
}

#undef FETCH
