#pragma once

#include "renderer_commons.cuh"
#include <forward_vector.h>
#include "renderer_cudad_bridge.cuh"

#define BLENDING_MODE_ALPHA 0
#define BLENDING_MODE_BEER_LAMBERT 1
//#define BLENDING_MODE
/*
 * No parameters,
 * but the host defines the blending mode
 * #define BLENDING_MODE
 */

namespace kernel
{
	struct Blending
	{
		__host__ __device__ __inline__
		real4 eval(const real4& rgb_alpha_previous, const real4& rgb_absorption_new_contribution) const
		{
			const real4& acc = rgb_alpha_previous;
			const real4& current = rgb_absorption_new_contribution;
#if BLENDING_MODE == BLENDING_MODE_BEER_LAMBERT
			real_t currentAlpha = 1 - real_t(std::exp(-current.w));
#elif BLENDING_MODE == BLENDING_MODE_ALPHA
			real_t currentAlpha = rmin(real_t(1), current.w);
#endif
			real3 colorOut = make_real3(acc) + (1 - acc.w) * make_real3(current) * currentAlpha;
			real_t alphaOut = acc.w + (1 - acc.w) * currentAlpha;
			return make_real4(colorOut, alphaOut);
		}

		__host__ __device__ __inline__
		RayEvaluationOutput eval(const RayEvaluationOutput& rgb_alpha_previous,
			const RayEvaluationOutput& rgb_absorption_new_contribution) const
		{
			const real4& acc = rgb_alpha_previous.color;
			const real4& current = rgb_absorption_new_contribution.color;
#if BLENDING_MODE == BLENDING_MODE_BEER_LAMBERT
			real_t currentAlpha = 1 - real_t(std::exp(-current.w));
#elif BLENDING_MODE == BLENDING_MODE_ALPHA
			real_t currentAlpha = rmin(real_t(1), current.w);
#endif
			real3 colorOut = make_real3(acc) + (1 - acc.w) * make_real3(current) * currentAlpha;
			real_t alphaOut = acc.w + (1 - acc.w) * currentAlpha;
			real3 normalOut = rgb_alpha_previous.normal + (1 - acc.w) * rgb_absorption_new_contribution.normal * currentAlpha;
			real_t depthOut = rgb_alpha_previous.depth + (1 - acc.w) * rgb_absorption_new_contribution.depth * currentAlpha;
			return { make_real4(colorOut, alphaOut), normalOut, depthOut };
		}
	};
}
