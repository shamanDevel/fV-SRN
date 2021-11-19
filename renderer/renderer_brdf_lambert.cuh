#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

#define BRDF_LIGHT_POINT 0
#define BRDF_LIGHT_DIRECTION 1

/*
 * Defines:
 * BRDF_LAMBERT__BATCHED
 * BRDF_LAMBERT_ENABLE_MAGNITUDE_SCALING
 * BRDF_LAMBERT_ENABLE_PHONG
 * BRDF_LAMBERT_LIGHT_TYPE 0/1
 */

namespace kernel
{
	struct BRDFLambertParameters
	{
#ifdef BRDF_LAMBERT__BATCHED
		//batch, 0, C; C=0->absorption, C=1->emission
		Tensor1Read<real_t> magnitudeScaling;
		Tensor1Read<real_t> ambient;
		Tensor1Read<real_t> specular;
		Tensor1Read<real_t> magnitudeCenter;
		Tensor1Read<real_t> magnitudeRadius;
		Tensor1Read<int>    specularExponent;
		Tensor2Read<real_t> lightParameter;
#else
		real_t magnitudeScaling;
		real_t ambient;
		real_t specular;
		real_t magnitudeCenter;
		real_t magnitudeRadius;
		int    specularExponent;
		real3  lightParameter;
#endif
	};
}

__constant__ kernel::BRDFLambertParameters brdfLambertParameters;

#ifdef BRDF_LAMBERT__BATCHED
#define FETCH(parameter) (brdfLambertParameters.parameter[batch])
#define FETCH3(parameter) fetchReal3(brdfLambertParameters.parameter, batch)
#else
#define FETCH(parameter) (brdfLambertParameters.parameter)
#define FETCH3(parameter) (brdfLambertParameters.parameter)
#endif

namespace kernel
{
	struct BRDFLambert
	{
		__host__ __device__ __inline__
			real4 eval(
				const real4& rgbAbsorption, const real3& position,
				const real3& gradient, const real3& rayDir, int batch)
		{
#if defined(BRDF_LAMBERT_ENABLE_MAGNITUDE_SCALING) || defined(BRDF_LAMBERT_ENABLE_PHONG)
			const real_t gradientNormSqr = lengthSquared(gradient);
			const real_t gradientNorm = rsqrt(gradientNormSqr);
			const real3 normal = safeNormalize(gradient);
#endif

			real3 rgb = make_real3(rgbAbsorption);
			real_t absorption = rgbAbsorption.w;

			//magnitude scaling
#ifdef BRDF_LAMBERT_ENABLE_MAGNITUDE_SCALING
			real_t magnitudeScaling = FETCH(magnitudeScaling);
			absorption *= (1 - rexp(-magnitudeScaling * gradientNormSqr));
#endif

			//phong
#ifdef BRDF_LAMBERT_ENABLE_PHONG
#if BRDF_LAMBERT_LIGHT_TYPE == BRDF_LIGHT_DIRECTION
			real3 lightDirection = normalize(-FETCH3(lightParameter));
#elif BRDF_LAMBERT_LIGHT_TYPE == BRDF_LIGHT_POINT
			real3 lightDirection = normalize(FETCH3(lightParameter) - position);
#else
#error "Unknown light type"
#endif
			real_t magnitudeCenter = FETCH(magnitudeCenter);
			real_t magnitudeRadius = FETCH(magnitudeRadius);
			real_t phongStrength = smoothstep(
				magnitudeCenter - magnitudeRadius,
				magnitudeCenter + magnitudeRadius,
				gradientNorm);
			real_t ambientStrength = lerp(1, FETCH(ambient), phongStrength);
			real3 diffuseLight = rabs(dot(normal, lightDirection)) * rgb;
			real_t specularExponent = FETCH(specularExponent);
			real3 specularLight = make_real3((specularExponent+2)*0.159155) * rpow(
				rmax(real_t(0), dot(rayDir, reflect(lightDirection, -normal))),
				specularExponent
			);
			rgb = ambientStrength * rgb +
				(1 - ambientStrength) * (diffuseLight + FETCH(specular) * specularLight);
#endif
			
			return make_real4(rgb, absorption);
		}
	};
}

#undef FETCH
#undef FETCH3
