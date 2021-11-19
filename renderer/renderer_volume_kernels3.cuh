#pragma once

#include "renderer_commons.cuh"
#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
#include "renderer_sampler_curand.cuh"

/**
 * Defines:
 * VOLUME_INTERPOLATION_T
 * HAS_TRANSFER_FUNCTION 0 or 1
 * TRANSFER_FUNCTION_T
 */

__global__ void ImportanceSamplingWithProbabilityGrid(
	unsigned int numSamples,
	kernel::Tensor3Read<real_t> probabilityGrid,
	kernel::Tensor2RW<real_t> positionsOutput,
	kernel::Tensor2RW<real_t> densitiesOutput,
	kernel::Tensor2RW<real_t> colorsOutput,
	real_t maxValue, real_t minProb,
	unsigned int* counter, int seed, int time,
	real_t densityMin, real_t densityMax)
{
	using VolumeInterpolation_t = VOLUME_INTERPOLATION_T;
	VolumeInterpolation_t volume;

	int3 volumeSize = make_int3(probabilityGrid.size(0), probabilityGrid.size(1), probabilityGrid.size(2));
	const int3 volumeSize2 = volumeSize - make_int3(1);

#if HAS_TRANSFER_FUNCTION==1
	using TF_t = TRANSFER_FUNCTION_T;
	TF_t tf;
	const real_t divDensityRange = real_t(1) / (densityMax - densityMin);
#endif

	//init sampler
	::kernel::Sampler sampler(seed, time);

	//Importance sampling has a lot of divergence between threads
	assert(KERNEL_SYNCHRONIZED_TRACING == 0);

	//loop until enough samples were drawn
	while (1)
	{
		//find sample location
		real3 position;
		real_t density;
		real4 color;
		while (1)
		{
			position.x = sampler.sampleUniform();
			position.y = sampler.sampleUniform();
			position.z = sampler.sampleUniform();
			//trilinear interpolation
			real3 gridPosition = position * make_real3(volumeSize2);
			int3 ipos = make_int3(gridPosition);
			real_t densities[8];
			int3 iposL = clamp(ipos, make_int3(0), volumeSize2);
			int3 iposH = clamp(ipos + make_int3(1), make_int3(0), volumeSize2);
			densities[0b000] = probabilityGrid[iposL.x][iposL.y][iposL.z];
			densities[0b001] = probabilityGrid[iposL.x][iposL.y][iposH.z];
			densities[0b010] = probabilityGrid[iposL.x][iposH.y][iposL.z];
			densities[0b011] = probabilityGrid[iposL.x][iposH.y][iposH.z];
			densities[0b100] = probabilityGrid[iposH.x][iposL.y][iposL.z];
			densities[0b101] = probabilityGrid[iposH.x][iposL.y][iposH.z];
			densities[0b110] = probabilityGrid[iposH.x][iposH.y][iposL.z];
			densities[0b111] = probabilityGrid[iposH.x][iposH.y][iposH.z];
			auto fpos = gridPosition - make_real3(ipos);
			real_t prob = lerp(
				lerp(
					lerp(densities[0b000], densities[0b100], fpos.x),
					lerp(densities[0b010], densities[0b110], fpos.x),
					fpos.y),
				lerp(
					lerp(densities[0b001], densities[0b101], fpos.x),
					lerp(densities[0b011], densities[0b111], fpos.x),
					fpos.y),
				fpos.z);
			//compute importance and sample
			prob = rmax(prob / maxValue, minProb);
			real_t query = sampler.sampleUniform();
			if (prob <= query)
				continue; //next try
			//take that sample!
			//fetch density
			constexpr int batch = 0; //For now
			const real3 direction = make_real3(0, 0, 0);
			density = volume.eval<real_t>(position, direction, batch).value;
			//compute color if requested
#if HAS_TRANSFER_FUNCTION==1
			color = make_real4(0, 0, 0, 0);
			if (density >= densityMin) {
				auto density2 = (density - densityMin) * divDensityRange;
				color = tf.eval(density2, real3{ 0,0,0 }, { 0 }, { 1 }, 0);
			}
#endif
			break;
		}
		//find empty spot in the output array
		unsigned int i = atomicInc(counter, 0xffffffffu);
		if (i >= numSamples)
			break; //no more space left
		//write output
		positionsOutput[i][0] = position.x;
		positionsOutput[i][1] = position.y;
		positionsOutput[i][2] = position.z;
		densitiesOutput[i][0] = density;
#if HAS_TRANSFER_FUNCTION==1
		colorsOutput[i][0] = color.x;
		colorsOutput[i][1] = color.y;
		colorsOutput[i][2] = color.z;
		colorsOutput[i][3] = color.w;
#endif
	}
}
