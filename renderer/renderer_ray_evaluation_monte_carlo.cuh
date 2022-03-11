#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

// for intelli-sense

#ifndef RAY_EVALUATION_MONTE_CARLO__SAMPLING_T
#include "renderer_sampler_curand.cuh"
#define RAY_EVALUATION_MONTE_CARLO__SAMPLING_T ::kernel::Sampler
#endif

#ifndef RAY_EVALUATION_MONTE_CARLO__VOLUME_INTERPOLATION_T
#include "renderer_volume_grid.cuh"
#define RAY_EVALUATION_MONTE_CARLO__VOLUME_INTERPOLATION_T ::kernel::VolumeInterpolationGrid
#endif

#ifndef RAY_EVALUATION_MONTE_CARLO__TRANSFER_FUNCTION_T
#include "renderer_tf_gaussian.cuh"
#define RAY_EVALUATION_MONTE_CARLO__TRANSFER_FUNCTION_T ::kernel::TransferFunctionGaussian
#endif

#ifndef RAY_EVALUATION_MONTE_CARLO__BRDF_T
#include "renderer_brdf_lambert.cuh"
#define RAY_EVALUATION_MONTE_CARLO__BRDF_T ::kernel::BRDFLambert
#endif

#ifndef RAY_EVALUATION_MONTE_CARLO__PHASE_FUNCTIION_T
#include "renderer_phase_function.cuh"
#define RAY_EVALUATION_MONTE_CARLO__PHASE_FUNCTIION_T ::kernel::PhaseFunctionHenyeyGreenstein
#endif

//Further defines:
//RAY_EVALUATION_MONTE_CARLO__SKIP_TRANSFER_FUNCTION

namespace kernel
{
	struct RayEvaluationMonteCarloParameters
	{
		real_t maxAbsorption; //max absorption for delta tracking
		real_t densityMin;
		real_t densityMax;
		//fraction of the absorption in the transmittance,
		//that is responsible for scattering. 1-scatteringFactor
		//is then the emission
		//real_t scatteringFactor;
		int numBounces;

		real4 lightPositionAndRadius;
		real_t lightIntensity;
		real_t colorScaling;
	};
}

__constant__ ::kernel::RayEvaluationMonteCarloParameters rayEvaluationMonteCarloParameters;

namespace kernel
{
	struct RayEvaluationMonteCarlo
	{
		using VolumeInterpolation_t = RAY_EVALUATION_MONTE_CARLO__VOLUME_INTERPOLATION_T;
		using TF_t = RAY_EVALUATION_MONTE_CARLO__TRANSFER_FUNCTION_T;
		//using BRDF_t = RAY_EVALUATION_MONTE_CARLO__BRDF_T;
		using PhaseFunction_t = RAY_EVALUATION_MONTE_CARLO__PHASE_FUNCTIION_T;
		using Sampler_t = RAY_EVALUATION_MONTE_CARLO__SAMPLING_T;

		VolumeInterpolation_t volume;
		TF_t tf;
		//BRDF_t brdf;
		PhaseFunction_t phase;

		/**
		 * Delta tracking through the volume. Returns >0 iff a medium interaction was sampled
		 */
		template<typename Sampler = Sampler_t>
		__device__ __inline__
		float deltaTracking(real3 rayStart, real3 rayDir, int batch, Sampler& sampler, 
			real3& hitPosition, real4& hitColorFromTf, real3& hitNormal)
		{
			//stepping
			const real_t maxAbsorption = rayEvaluationMonteCarloParameters.maxAbsorption;
			const real_t divMaxAbsorption = 1 / maxAbsorption;
			const real_t densityMin = rayEvaluationMonteCarloParameters.densityMin;
			const real_t densityMax = rayEvaluationMonteCarloParameters.densityMax;
			const real_t divDensityRange = real_t(1) / (densityMax - densityMin);

			real_t tout = 0;
			real_t tcurrent = 0;
			bool isValid = true;
			while (true) {
				//sample in homogeneous medium
				real_t negStepsize = rlog(sampler.sampleUniform()) * divMaxAbsorption;
				tcurrent -= negStepsize;

				//evaluate current density
				real3 position = rayStart + rayDir * tcurrent;
				const auto resultFromForward = volume.eval<real_t>(position, rayDir, batch);
				const auto density = resultFromForward.value;
				const bool isInside = resultFromForward.isInside;
				if (isValid && !isInside) {
					hitPosition = position;
					tout = 0;
					isValid = false;
				}

				//compute normal
				real3 normal = make_real3(0);
				const int requireNormal = isValid && density >= densityMin;
#if KERNEL_SYNCHRONIZED_TRACING==0
				if (requireNormal)
					normal = volume.evalNormal(position, rayDir, resultFromForward, batch);
#else
				//evaluate normal if *any* thread requests normals (TensorCores)
				if (__any_sync(0xffffffff, requireNormal))
					normal = volume.evalNormal(position, rayDir, resultFromForward, batch);
#endif

				if (requireNormal)
				{
					auto density2 = (density - densityMin) * divDensityRange;

					//evaluate TF
					VolumeInterpolation_t::density_t previousDensity = { 0 };
					auto color1 = tf.eval(density2, normal, previousDensity, 1, batch);
					auto currentAbsorption = color1.w;

					//if (threadIdx.x == 0 && blockIdx.x == 0)
					//	printf("color: %.3f,%.3f,%.3f,%.3f\n",
					//		color1.x, color1.y, color1.z, color1.w);

					//check if this was a virtual particle or real particle
					if (currentAbsorption * divMaxAbsorption > sampler.sampleUniform())
					{
						//real particle
						hitPosition = position;
						hitColorFromTf = color1;
						hitNormal = normal;
						tout = tcurrent;
						isValid = false;
					}
				}
#if KERNEL_SYNCHRONIZED_TRACING==0
				if (!isValid) break;
#else
				//break only if all threads in the warp are done
				if (!__any_sync(0xffffffff, isValid))
					break;
#endif
			}

			return tout;
		}
		
		/**
		 * Samples a position on the light
		 */
		template<typename Sampler = Sampler_t>
		__device__ __inline__
		real3 sampleLightPosition(Sampler& sampler)
		{
			//sample unit sphere
			real3 pos;
			do
			{
				pos = make_real3(
					sampler.sampleUniform() * 2 - 1,
					sampler.sampleUniform() * 2 - 1,
					sampler.sampleUniform() * 2 - 1
				);
			} while (lengthSquared(pos) > 1);
			//normalize, transform
			real3 center = make_real3(rayEvaluationMonteCarloParameters.lightPositionAndRadius);
			real_t radius = rayEvaluationMonteCarloParameters.lightPositionAndRadius.w;
			return normalize(pos) * radius + center;
		}
		
		/**
		 * Background intersection. No scattering in the medium happened,
		 * now check for area lights.
		 */
		template<typename Sampler = Sampler_t>
		__device__ __inline__
		RayEvaluationOutput evalBackground(
			real3 rayStart, real3 rayDir, int batch, Sampler& sample)
		{
			bool isLight;
			//ray sphere intersection
			real3 center = make_real3(rayEvaluationMonteCarloParameters.lightPositionAndRadius);
			real_t radius = rayEvaluationMonteCarloParameters.lightPositionAndRadius.w;
			real_t a = lengthSquared(rayDir);
			real_t b = 2 * dot(rayDir, rayStart - center);
			real_t c = lengthSquared(rayStart - center) - radius * radius;
			real_t D = b * b - 4 * a * c;
			isLight = D > 0;

			real_t I = isLight ? rayEvaluationMonteCarloParameters.lightIntensity : 0;
			real_t alpha = isLight ? 1 : 0;
			return RayEvaluationOutput{
				make_real4(I,I,I,alpha),
				make_real3(0,0,0),
				0
			};
		}

		template<typename Sampler = Sampler_t>
		__device__ __inline__
		RayEvaluationOutput eval(real3 rayStart, real3 rayDir, real_t tmax, int batch, Sampler& sampler)
		{
			real_t tmin, tmax1;
			intersectionRayAABB(
				rayStart, rayDir, volume.getBoxMin(), volume.getBoxSize(),
				tmin, tmax1);
			tmin = rmax(tmin, 0);
			tmax = rmin(tmax, tmax1);

			const int numBounces = rayEvaluationMonteCarloParameters.numBounces;
			const real_t colorScaling = rayEvaluationMonteCarloParameters.colorScaling;
			
			real3 emission = make_real3(0);
			real3 beta = make_real3(1);
			real_t outAlpha = 0;
			real_t outDepth = 0;
			real3 outNormal = make_real3(0);
			real3 position = rayStart + tmin * rayDir;
			bool isValid = true;
			for (int bounces = 0; bounces <= numBounces; ++bounces)
			{
				//find next intersection
				real3 nextPosition;
				real4 tfColor;
				real3 normal;
				float thit = deltaTracking(position, rayDir, batch,
					sampler, nextPosition, tfColor, normal);
				if (isValid && bounces == 0)
				{
					outAlpha = thit > 0;
					outDepth = thit;
					outNormal = normal;
				}

#if KERNEL_SYNCHRONIZED_TRACING==0
				bool any_hit = thit > 0;
#else
				bool any_hit = __any_sync(0xffffffff, thit > 0);
#endif

				if (any_hit)
				{
					// NOT NEEDED
					//if (thit > 0 && isValid && bounces==0 && thit+tmin>tmax)
					//{
					//	//background hit
					//	outAlpha = 0;
					//	emission = make_real3(0, 0, 0);
					//	isValid = false;
					//}

					//medium intersection
					if (thit>0)
					    beta *= make_real3(tfColor) * (tfColor.w * colorScaling);
					
					//1. direct illumination
					real3 hitPosition;
					real4 hitColor;
					real3 hitNormal;
					real3 lightPos = sampleLightPosition(sampler);
					real3 lightDir = normalize(lightPos - nextPosition);
					real_t p = phase.prob(rayDir, lightDir, nextPosition, batch);
					if (deltaTracking(nextPosition, lightDir, batch, sampler, hitPosition, hitColor, hitNormal) <= 0)
					{
						if (thit > 0 && isValid) {
							real_t I = rayEvaluationMonteCarloParameters.lightIntensity;
							emission += beta * make_real3(p * I);
						}
					}
					
					//2. next ray
					if (thit > 0 && isValid) {
						real3 nextDir = phase.sample(rayDir, nextPosition, sampler, batch);
						beta *= phase.prob(rayDir, nextDir, nextPosition, batch);
						position = nextPosition;
						rayDir = nextDir;
					}
				}
				if (isValid && thit <= 0)
					isValid = false;
//				else
//				{
//					//background
//					if (isValid) {
//						real4 c = evalBackground(position, rayDir, batch, sampler).color;
//#ifdef RAY_EVALUATION_MONTE_CARLO__HIDE_LIGHT
//						if (bounces > 0) emission += beta * make_real3(c);
//#else
//						emission += beta * make_real3(c);
//						if (bounces == 0) outAlpha = c.w;
//#endif
//						isValid = false;
//					}
//				}

#if KERNEL_SYNCHRONIZED_TRACING==0
				if (!isValid) break;
#else
				//break early only if all threads in the warp are done
				if (!__any_sync(0xffffffff, isValid))
					break;
#endif
			}
			
			return RayEvaluationOutput{
					make_real4(emission, outAlpha),
					normalize(outNormal),
					outDepth
			};
		}
	};
}