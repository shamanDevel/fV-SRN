#pragma once

#include "renderer_tensor.cuh"
#include "renderer_commons.cuh"
#include "renderer_utils.cuh"
#include "helper_math.cuh"
#include "renderer_activations.cuh"

#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include <mma.h>

/**
* For the brave souls who get this far and untangle the code below,
* you are the chosen ones, the saviors of the art of programming,
* the gods among men who can move the world on their fingertips.
* To you, true saviors, kings of men, I say this:
* never gonna give you up, never gonna let you down,
* never gonna run around and desert you. Never gonna make you cry,
* never gonna say goodbye. Never gonna tell a lie and hurt you.
*/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#error "TensorCores require a CUDA architecture >= 700"
#endif

/* Defined by the host
 * Filled with defaults for intellisense
 *
 * BLOCK_SIZE
 * NUM_HIDDEN_LAYERS
 * HIDDEN_CHANNELS_DIV16
 * HAS_FOURIER_FEATURES
 * NUM_FOURIER_FEATURES
 * ACTIVATION
 * FIRST_AND_LAST_IN_SHARED_MEMORY
 */
#define TO_BE_FILLED_BY_THE_HOST -1

//The block size, maximized for register usage
//but small enough to fit all shared memory
#ifndef BLOCK_SIZE 
#define BLOCK_SIZE 256 //TO_BE_FILLED_BY_THE_HOST
#endif

//the number of hidden layers
#ifndef NUM_HIDDEN_LAYERS
#define NUM_HIDDEN_LAYERS 1 //TO_BE_FILLED_BY_THE_HOST
#endif

//the number of channels in the hidden layers divided by 16
#ifndef HIDDEN_CHANNELS_DIV16
#define HIDDEN_CHANNELS_DIV16 2 // TO_BE_FILLED_BY_THE_HOST
#endif
#define HIDDEN_CHANNELS (16*(HIDDEN_CHANNELS_DIV16))

//0 or 1 depending on if the first layer represents the fourier feature mapping.
//If false (0), cWeightsFirst and cBiasFirst store the weights and bias of the
//  first layer with 3 input channels and FIRST_CHANNELS output channels.
//  Usually in this case, FIRST_CHANNELS==HIDDEN_CHANNELS
//If true (1), NUM_FOURIER_FEATURES contains the number of fourier features
//  and cWeightsFourier store that many weights and features.
//  Given an input position 'real3 p', the outputs of the fourier features are assembled as
//  [p.x, p.y, p.z, 0,
//    cos(cWeightsFourier * p),
//    sin(cWeightsFourier * p)]
//  Note that the multiplication with 2*pi is assumed to be included in cWeightsFourier.
// It must hold that (4+2*NUM_FOURIER_FEATURES) == FIRST_CHANNELS
#ifndef HAS_FOURIER_FEATURES
#define HAS_FOURIER_FEATURES 1 //TO_BE_FILLED_BY_THE_HOST
#endif
#ifndef NUM_FOURIER_FEATURES
#define NUM_FOURIER_FEATURES 14 //TO_BE_FILLED_BY_THE_HOST
#endif

/**
 * Specifies how the direction shall be used:
 * 0: no direction
 * 1: direction as additional input, but not in the fourier matrix
 * 2: direction as input + fourier matrix
 */
#ifndef USE_DIRECTION
#define USE_DIRECTION 0 //TO_BE_FILLED_BY_THE_HOST
#endif

//The activation function. Must be a valid function name in the ::kernel::activation
//namespace. Currently: ReLU, Sine
#ifndef ACTIVATION
#define ACTIVATION SnakeAlt
#endif

#define OUTPUT_MODE_DENSITY 0 //real_t output
#define OUTPUT_MODE_DENSITY_DIRECT 1 //real_t output
#define OUTPUT_MODE_RGBO 2   //real4 output
#define OUTPUT_MODE_RGBO_DIRECT 3   //real4 output
#define OUTPUT_MODE_DENSITY_GRADIENT 4   //real_t + gradient output
#define OUTPUT_MODE_DENSITY_GRADIENT_DIRECT 5   //real_t + gradient output
#define OUTPUT_MODE_DENSITY_GRADIENT_CUBIC 6   //real_t + gradient output, but gradients are cubed
#define OUTPUT_MODE_DENSITY_CURVATURE 7 //density + gradient + curvature
#define OUTPUT_MODE_DENSITY_CURVATURE_DIRECT 8 //density + gradient + curvature
#ifndef OUTPUT_MODE
#define OUTPUT_MODE OUTPUT_MODE_DENSITY_CURVATURE //TO_BE_FILLED_BY_THE_HOST
#endif

//Bias, the last weights and the shared memory can be accessed
//directly from either shared memory or constant memory.
//Define as 1 for shared memory, as 0 for constant memory
#ifndef FIRST_AND_LAST_IN_SHARED_MEMORY
#define FIRST_AND_LAST_IN_SHARED_MEMORY 1
#endif

//The number of channels in the latent grid (or 0 if no latent grid)
//The channels are grouped by 4 as the 3D textures store four channels each
#ifndef LATENT_GRID_CHANNELS_DIV16
#define LATENT_GRID_CHANNELS_DIV16 1 //TO_BE_FILLED_BY_THE_HOST
#endif

#define LATENT_GRID_ENCODING_FLOAT 0 //direct float storage
#define LATENT_GRID_ENCODING_BYTE_LINEAR 1 //bytes, linear storage
#define LATENT_GRID_ENCODING_BYTE_GAUSSIAN 2 //bytes, gaussian mapping
//The encoding for the latent grid
#ifndef LATENT_GRID_ENCODING
#define LATENT_GRID_ENCODING LATENT_GRID_ENCODING_FLOAT //TO_BE_FILLED_BY_THE_HOST
#endif

//Pass the time as additional input after the position to the network.
//Requires a latent grid
#ifndef PASS_TIME_TO_NETWORK
#define PASS_TIME_TO_NETWORK 1 //TO_BE_FILLED_BY_THE_HOST
#endif

#define GRADIENT_MODE_OFF_OR_DIRECT 0
#define GRADIENT_MODE_FINITE_DIFFERENCES 1
#define GRADIENT_MODE_ADJOINT_METHOD 2
//the gradient computation mode
#ifndef GRADIENT_MODE
#define GRADIENT_MODE GRADIENT_MODE_OFF_OR_DIRECT
#endif

//further macros

#if HAS_FOURIER_FEATURES==1
#if USE_DIRECTION>=1
#if (8+2*NUM_FOURIER_FEATURES) != HIDDEN_CHANNELS
#error "(8+2*NUM_FOURIER_FEATURES) == HIDDEN_CHANNELS not satisfied"
#endif
#else
#if (4+2*NUM_FOURIER_FEATURES) != HIDDEN_CHANNELS
#error "(4+2*NUM_FOURIER_FEATURES) == HIDDEN_CHANNELS not satisfied"
#endif
#endif
#endif

#if USE_DIRECTION==2
#define FOURIER_CHANNELS 6
#else
#define FOURIER_CHANNELS 3
#endif
#if USE_DIRECTION>=1
#define FIRST_CHANNELS 6
#else
#define FIRST_CHANNELS 3
#endif

#define LATENT_GRID_CHANNELS_DIV4 (LATENT_GRID_CHANNELS_DIV16*4)
#if LATENT_GRID_CHANNELS_DIV4>0

#if HAS_FOURIER_FEATURES==0
#error "For now, LatentGrid requires FourierFeatures to simplify the implementation"
#endif
#define HAS_LATENT_GRID 1
#define LATENT_GRID_LAYER_TOTAL_INPUTS (16*(LATENT_GRID_CHANNELS_DIV16) + (HIDDEN_CHANNELS))
#define LATENT_GRID_LAYER_TOTAL_INPUTS_DIV16 ((LATENT_GRID_CHANNELS_DIV16) + (HIDDEN_CHANNELS_DIV16))

#else
#define HAS_LATENT_GRID 0
#if PASS_TIME_TO_NETWORK==1
#error "PASS_TIME_TO_NETWORK requires a Latent Gird"
#endif
#endif

#define NUM_HIDDEN_LAYERS_WITH_LATENT_GRID (NUM_HIDDEN_LAYERS + HAS_LATENT_GRID)

#if LATENT_GRID_ENCODING>0
#define HAS_LATENT_GRID_OFFSET_SCALE 1
#else
#define HAS_LATENT_GRID_OFFSET_SCALE 0
#endif

#define MIN_ONE(val) (((val)>0)?(val):(1))

//Parameters
namespace kernel
{
	struct alignas(32) VolumeInterpolationTensorcoresParameters
	{

#if HAS_FOURIER_FEATURES==1
		alignas(32) half cWeightsFourier[FOURIER_CHANNELS * NUM_FOURIER_FEATURES]; //column-major (output channels (NUM_FOURIER_FEATURES) are fastest)
#else
		alignas(32) half cWeightsFirst[FIRST_CHANNELS * HIDDEN_CHANNELS]; //column-major (output channels are fastest)
		alignas(32) half cBiasFirst[HIDDEN_CHANNELS];
#endif

#if HAS_LATENT_GRID==1
		alignas(32) cudaTextureObject_t cLatentGridA[LATENT_GRID_CHANNELS_DIV4];
		alignas(32) cudaTextureObject_t cLatentGridB[LATENT_GRID_CHANNELS_DIV4];
#if HAS_LATENT_GRID_OFFSET_SCALE==1
		alignas(32) float4 cLatentGridOffsetA[LATENT_GRID_CHANNELS_DIV4];
		alignas(32) float4 cLatentGridOffsetB[LATENT_GRID_CHANNELS_DIV4];
		alignas(32) float4 cLatentGridScaleA[LATENT_GRID_CHANNELS_DIV4];
		alignas(32) float4 cLatentGridScaleB[LATENT_GRID_CHANNELS_DIV4];
#endif
		//interpolation
		//Note: this also serves as the time that is passed to the network,
		// the first entry from here is used for that.
		alignas(32) float4 cLatentGridInterpolation[LATENT_GRID_CHANNELS_DIV4];
		//weights for the first layer that interprets the fourier features + latent grid
		alignas(32) half cWeightsLatentGrid[LATENT_GRID_LAYER_TOTAL_INPUTS * HIDDEN_CHANNELS]; //row-major (input channels are fastest)
		alignas(32) half cBiasLatentGrid[HIDDEN_CHANNELS];
#endif

		alignas(32) half cWeightsHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS * HIDDEN_CHANNELS)]; //row-major (input channels are fastest)
		alignas(32) half cBiasHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS)];

#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_CUBIC)
		alignas(32) half2 cWeightsLast[HIDDEN_CHANNELS * 2]; //col-major (output channels are fastest)
		alignas(32) half2 cBiasLast[2];
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_DIRECT)
		alignas(32) half cWeightsLast[HIDDEN_CHANNELS];
		alignas(32) half cBiasLast;
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT)
		alignas(32) half2 cWeightsLast[HIDDEN_CHANNELS * 3]; //col-major (output channels are fastest)
		alignas(32) half2 cBiasLast[3];
#else
#error "Unknown output mode, forgot to handle the new case here?"
#endif

		alignas(32) float3 boxMin;
		alignas(32) float3 boxSize;
		alignas(32) float activationParam;
		alignas(32) float finiteDifferencesStepsize;
		alignas(32) float latentGridDifferencesStepSize;
	};
}

__constant__ kernel::VolumeInterpolationTensorcoresParameters volumeInterpolationTensorcoresParameters;

//WARNING: all memory involved in tensor-core operations (wmma)
// must be aligned to 32 bytes.
#if HAS_LATENT_GRID==1
__shared__ alignas(32) half sWeightsLatentGrid[LATENT_GRID_LAYER_TOTAL_INPUTS * HIDDEN_CHANNELS];
__shared__ alignas(32) half sBiasLatentGrid[HIDDEN_CHANNELS];
#endif
__shared__ alignas(32) half sWeightsHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS * HIDDEN_CHANNELS)];
__shared__ alignas(32) half sBiasHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS)];
__shared__ alignas(32) half sIntermediateResults[BLOCK_SIZE * HIDDEN_CHANNELS];
#if GRADIENT_MODE == GRADIENT_MODE_ADJOINT_METHOD
__shared__ alignas(32) half sStorageForAdjointHidden[MIN_ONE(NUM_HIDDEN_LAYERS_WITH_LATENT_GRID * BLOCK_SIZE * HIDDEN_CHANNELS)];
__shared__ alignas(32) half sStorageForAdjointLast[BLOCK_SIZE];
#endif

#if FIRST_AND_LAST_IN_SHARED_MEMORY==1

#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_CUBIC)
__shared__ alignas(32) half2 sWeightsLast[HIDDEN_CHANNELS * 2];
__shared__ half2 sBiasLast[2];
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_DIRECT)
__shared__ half sWeightsLast[HIDDEN_CHANNELS];
__shared__ half sBiasLast;
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT)
__shared__ half2 sWeightsLast[HIDDEN_CHANNELS * 3]; //col-major (output channels are fastest)
__shared__ half2 sBiasLast[3];
#else
#error "Unknown output mode, forgot to handle the new case here?"
#endif

#if HAS_FOURIER_FEATURES==1
__shared__ half sWeightsFourier[FOURIER_CHANNELS * NUM_FOURIER_FEATURES];
#else
__shared__ half sWeightsFirst[FIRST_CHANNELS * HIDDEN_CHANNELS];
__shared__ half sBiasFirst[HIDDEN_CHANNELS];
#endif

#else

//aliases, so the code does not care about the shared<->constant memory switch
#define sWeightsLast volumeInterpolationTensorcoresParameters.cWeightsLast
#define sBiasLast volumeInterpolationTensorcoresParameters.cBiasLast
#if HAS_FOURIER_FEATURES==1
#define sWeightsFourier volumeInterpolationTensorcoresParameters.cWeightsFourier
#else
#define sWeightsFirst volumeInterpolationTensorcoresParameters.cWeightsFirst
#define sBiasFirst volumeInterpolationTensorcoresParameters.cBiasFirst
#endif

#endif

namespace kernel
{

	template<typename Value_t = real_t>
	struct VolumeNetworkOutput
	{
		Value_t value;
		Value_t valueNoClamping;
		bool isInside;
#if (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT) || \
    (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_CUBIC) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT)
		real3 normal;
#endif
#if (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT)
		real2 curvature;
#endif
	};

	namespace encoding
	{
		template<int GridEncoding, bool IsB>
		__device__ __inline__ float4 EncodeGridValue(float4 v, int channel);

		template<>
		__device__ __inline__ float4 EncodeGridValue<LATENT_GRID_ENCODING_FLOAT, false>(float4 v, int channel)
		{
			return v;
		}
		template<>
		__device__ __inline__ float4 EncodeGridValue<LATENT_GRID_ENCODING_FLOAT, true>(float4 v, int channel)
		{
			return v;
		}

		//value(x) = offset + x * scale
		template<>
		__device__ __inline__ float4 EncodeGridValue<LATENT_GRID_ENCODING_BYTE_LINEAR, false>(float4 v, int channel)
		{
#if HAS_LATENT_GRID_OFFSET_SCALE==1
			return volumeInterpolationTensorcoresParameters.cLatentGridOffsetA[channel] +
				v * volumeInterpolationTensorcoresParameters.cLatentGridScaleA[channel];
#else
			__trap();
			return v; //never called
#endif
		}
		template<>
		__device__ __inline__ float4 EncodeGridValue<LATENT_GRID_ENCODING_BYTE_LINEAR, true>(float4 v, int channel)
		{
#if HAS_LATENT_GRID_OFFSET_SCALE==1
			return volumeInterpolationTensorcoresParameters.cLatentGridOffsetB[channel] +
				v * volumeInterpolationTensorcoresParameters.cLatentGridScaleB[channel];
#else
			__trap();
			return v; //never called
#endif
		}

		constexpr float ENCODING_GAUSSIAN_EPSILON = 1e-4f;
		constexpr float ENCODING_GAUSSIAN_2_MINUS_EPSILON = 2 - 1e-4f;
		constexpr float ENCODING_GAUSSIAN_SQRT2 = 1.4142135623730950488016887242096980f;
		//value(x) = mean + sigma * \Theta^-1(x)
		//\Theta^-1(x) = sqrt(2) * ErfInvf((2-epsilon)*(v-0.5))
		template<>
		__device__ __inline__ float4 EncodeGridValue<LATENT_GRID_ENCODING_BYTE_GAUSSIAN, false>(float4 v, int channel)
		{
			return EncodeGridValue<LATENT_GRID_ENCODING_BYTE_LINEAR, false>(
				ENCODING_GAUSSIAN_SQRT2 * erfinvf(ENCODING_GAUSSIAN_2_MINUS_EPSILON * (v - 0.5f)),
				channel);
		}
		template<>
		__device__ __inline__ float4 EncodeGridValue<LATENT_GRID_ENCODING_BYTE_GAUSSIAN, true>(float4 v, int channel)
		{
			return EncodeGridValue<LATENT_GRID_ENCODING_BYTE_LINEAR, true>(
				ENCODING_GAUSSIAN_SQRT2 * erfinvf(ENCODING_GAUSSIAN_2_MINUS_EPSILON * (v - 0.5f)),
				channel);
		}
	}

	struct VolumeInterpolationTensorcores
	{
		__host__ __device__ __forceinline__
			real3 getBoxMin() const
		{
			return make_real3(volumeInterpolationTensorcoresParameters.boxMin);
		}
		__host__ __device__ __forceinline__
			real3 getBoxSize() const
		{
			return make_real3(volumeInterpolationTensorcoresParameters.boxSize);
		}

		using density_t = real_t;

		__device__ __inline__ VolumeInterpolationTensorcores()
		{
			//make use of the fact that the kernel structs are
			// instantiated once at the beginning of the kernel.
			//Load the weights here!

			assert(blockDim.x == BLOCK_SIZE);
			const int warpID = threadIdx.x / 32;
			const int lineID = threadIdx.x % 32;
			const int numWarps = blockDim.x / 32;

			//first layer / fourier
			if (warpID == 0)
			{
#if FIRST_AND_LAST_IN_SHARED_MEMORY==1
#if HAS_FOURIER_FEATURES==1
				static constexpr int numFourierDiv32 = CUMAT_DIV_UP(NUM_FOURIER_FEATURES, 32);
#pragma unroll
				for (int c = 0; c < numFourierDiv32; ++c) {
					const int i = lineID + 32 * c;
					if (i < NUM_FOURIER_FEATURES) {
#pragma unroll
						for (int j=0; j< FOURIER_CHANNELS; ++j)
							sWeightsFourier[i + NUM_FOURIER_FEATURES * j] = volumeInterpolationTensorcoresParameters.cWeightsFourier[i + NUM_FOURIER_FEATURES * j];
					}
				}
#else
				static constexpr int cinDiv32 = CUMAT_DIV_UP(HIDDEN_CHANNELS, 32);
#pragma unroll
				for (int c = 0; c < cinDiv32; ++c) {
					const int i = lineID + 32 * c;
					if (i < HIDDEN_CHANNELS) {
#pragma unroll
						for (int j = 0; j < FIRST_CHANNELS; ++j)
							sWeightsFirst[i + HIDDEN_CHANNELS * j] = volumeInterpolationTensorcoresParameters.cWeightsFirst[i + HIDDEN_CHANNELS * j];
						sBiasFirst[i] = volumeInterpolationTensorcoresParameters.cBiasFirst[i];
					}
				}
#endif
#endif
			}

			//latent grid -> separate first hidden layer
#if HAS_LATENT_GRID==1
			if (warpID==0)
			{
				static constexpr int coutDiv32 = CUMAT_DIV_UP(HIDDEN_CHANNELS, 32);
#pragma unroll
				for (int cout = 0; cout < coutDiv32; ++cout) {
					const int i = lineID + 32 * cout;
					if (i < HIDDEN_CHANNELS) {
						for (int cin=0; cin<LATENT_GRID_LAYER_TOTAL_INPUTS; ++cin)
						{
							sWeightsLatentGrid[cin + LATENT_GRID_LAYER_TOTAL_INPUTS * i] =
								volumeInterpolationTensorcoresParameters.cWeightsLatentGrid[cin + LATENT_GRID_LAYER_TOTAL_INPUTS * i];
						}
						sBiasLatentGrid[i] = volumeInterpolationTensorcoresParameters.cBiasLatentGrid[ i];
					}
				}
			}
#endif

			//hidden
			for (int layer = warpID; layer < NUM_HIDDEN_LAYERS; layer += numWarps)
			{
				static constexpr int coutDiv32 = CUMAT_DIV_UP(HIDDEN_CHANNELS, 32);
#pragma unroll
				for (int c = 0; c < coutDiv32; ++c) {
					const int i = lineID + 32 * c;
					if (i < HIDDEN_CHANNELS) {
#pragma unroll
						for (int c2 = 0; c2 < HIDDEN_CHANNELS; ++c2)
							sWeightsHidden[layer * HIDDEN_CHANNELS * HIDDEN_CHANNELS + c2 * HIDDEN_CHANNELS + i] =
							volumeInterpolationTensorcoresParameters.cWeightsHidden[layer * HIDDEN_CHANNELS * HIDDEN_CHANNELS + c2 * HIDDEN_CHANNELS + i];
						sBiasHidden[layer * HIDDEN_CHANNELS + i] = volumeInterpolationTensorcoresParameters.cBiasHidden[layer * HIDDEN_CHANNELS + i];
					}
				}
			}

			//last layer
			if (warpID == 0)
			{
#if FIRST_AND_LAST_IN_SHARED_MEMORY==1
				static constexpr int coutDiv32 = CUMAT_DIV_UP(HIDDEN_CHANNELS, 32);
#pragma unroll
				for (int c = 0; c < coutDiv32; ++c) {
					const int i = lineID + 32 * c;
					if (i < HIDDEN_CHANNELS) {
#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_CUBIC)
						sWeightsLast[i + HIDDEN_CHANNELS * 0] = volumeInterpolationTensorcoresParameters.cWeightsLast[i + HIDDEN_CHANNELS * 0];
						sWeightsLast[i + HIDDEN_CHANNELS * 1] = volumeInterpolationTensorcoresParameters.cWeightsLast[i + HIDDEN_CHANNELS * 1];
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT)
						sWeightsLast[i + HIDDEN_CHANNELS * 0] = volumeInterpolationTensorcoresParameters.cWeightsLast[i + HIDDEN_CHANNELS * 0];
						sWeightsLast[i + HIDDEN_CHANNELS * 1] = volumeInterpolationTensorcoresParameters.cWeightsLast[i + HIDDEN_CHANNELS * 1];
						sWeightsLast[i + HIDDEN_CHANNELS * 2] = volumeInterpolationTensorcoresParameters.cWeightsLast[i + HIDDEN_CHANNELS * 2];
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_DIRECT)
						sWeightsLast[i] = volumeInterpolationTensorcoresParameters.cWeightsLast[i];
#else
#error "Unknown output mode, forgot to handle the new case here?"
#endif
					}
				}
				if (lineID == 0) {
#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_CUBIC)
					sBiasLast[0] = volumeInterpolationTensorcoresParameters.cBiasLast[0];
					sBiasLast[1] = volumeInterpolationTensorcoresParameters.cBiasLast[1];
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT)
					sBiasLast[0] = volumeInterpolationTensorcoresParameters.cBiasLast[0];
					sBiasLast[1] = volumeInterpolationTensorcoresParameters.cBiasLast[1];
					sBiasLast[2] = volumeInterpolationTensorcoresParameters.cBiasLast[2];
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_DIRECT)
					sBiasLast = volumeInterpolationTensorcoresParameters.cBiasLast;
#else
#error "Unknown output mode, forgot to handle the new case here?"
#endif
				}
#endif
			}

//			//DEBUG: inspect constants
//			if (warpID==0 && lineID==0)
//			{
//				printf("DUMP CONSTANT MEMORY\n");
//#define DUMP_ARRAY_HALF(ax, count, nl)	\
//	do {printf(#ax ":"); for (int ii=0; ii<(count); ++ii) {if (nl>0 && ii%nl==0)printf("\n"); printf(" %.2f", __half2float(volumeInterpolationTensorcoresParameters.ax[ii]));} printf("\n"); } while(0)
//#define DUMP_ARRAY_INT(ax, count)	\
//	do {printf(#ax ":"); for (int ii=0; ii<(count); ++ii) {printf(" %d", int(volumeInterpolationTensorcoresParameters.ax[ii]));} printf("\n"); } while(0)
//#define DUMP_ARRAY_FLOAT4(ax, count)	\
//	do {printf(#ax ":"); for (int ii=0; ii<(count); ++ii) {	\
//	    printf(" %.2f", volumeInterpolationTensorcoresParameters.ax[ii].x);	\
//		printf(" %.2f", volumeInterpolationTensorcoresParameters.ax[ii].y);	\
//		printf(" %.2f", volumeInterpolationTensorcoresParameters.ax[ii].z);	\
//		printf(" %.2f", volumeInterpolationTensorcoresParameters.ax[ii].w);	\
//	} printf("\n"); } while(0)
//
//#if HAS_FOURIER_FEATURES==1
//				DUMP_ARRAY_HALF(cWeightsFourier, FOURIER_CHANNELS* NUM_FOURIER_FEATURES, NUM_FOURIER_FEATURES);
//#else
//				DUMP_ARRAY_HALF(cWeightsFirst, FIRST_CHANNELS* HIDDEN_CHANNELS);
//				DUMP_ARRAY_HALF(cBiasFirst, HIDDEN_CHANNELS);
//#endif
//#if HAS_LATENT_GRID==1
//				DUMP_ARRAY_INT(cLatentGridA, LATENT_GRID_CHANNELS_DIV4);
//				DUMP_ARRAY_INT(cLatentGridB, LATENT_GRID_CHANNELS_DIV4);
//#if HAS_LATENT_GRID_OFFSET_SCALE==1
//				DUMP_ARRAY_FLOAT4(cLatentGridOffsetA, LATENT_GRID_CHANNELS_DIV4);
//				DUMP_ARRAY_FLOAT4(cLatentGridOffsetB, LATENT_GRID_CHANNELS_DIV4);
//				DUMP_ARRAY_FLOAT4(cLatentGridScaleA, LATENT_GRID_CHANNELS_DIV4);
//				DUMP_ARRAY_FLOAT4(cLatentGridScaleB, LATENT_GRID_CHANNELS_DIV4);
//#endif
//				DUMP_ARRAY_FLOAT4(cLatentGridInterpolation, LATENT_GRID_CHANNELS_DIV4);
//				DUMP_ARRAY_HALF(cWeightsLatentGrid, LATENT_GRID_LAYER_TOTAL_INPUTS* HIDDEN_CHANNELS, LATENT_GRID_LAYER_TOTAL_INPUTS);
//				DUMP_ARRAY_HALF(cBiasLatentGrid, HIDDEN_CHANNELS, 0);
//#endif
//				DUMP_ARRAY_HALF(cWeightsHidden, MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS * HIDDEN_CHANNELS), HIDDEN_CHANNELS);
//				DUMP_ARRAY_HALF(cBiasHidden, MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS), NUM_HIDDEN_LAYERS);
//			}

			__syncthreads();
		}

		/**
		 * Loads 16 channels of the volumetric features at position \c position
		 * starting with channel <code>16*startChannelDiv16</code>
		 * into the output \c target with stride between consecutive channels given by \c targetStride.
		 * For time interpolation, \c time in [0,1] is given.
		 */
		template<int GridEncoding>
		static __device__ __inline__ void LoadVolumetricFeatures(
			float3 position, int startChannelDiv16, half* target, int targetStride)
		{
			//textures store groups of 4 channels -> four textures to fill the 16 channels
			for (int i=0; i<4; ++i)
			{
				const int texIndex = 4 * startChannelDiv16 + i;
				//fetch texture value
				float4 vA = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridA[texIndex],
					position.x, position.y, position.z);
				float4 vB = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridB[texIndex],
					position.x, position.y, position.z);
				//grid encoding
				vA = encoding::EncodeGridValue<GridEncoding, false>(vA, texIndex);
				vB = encoding::EncodeGridValue<GridEncoding, false>(vB, texIndex);
				//time interpolation
				float4 time = volumeInterpolationTensorcoresParameters.cLatentGridInterpolation[texIndex];
				time = fracf(time);
				float4 v = lerp(vA, vB, time);
				//convert to half and write to output
				target[targetStride * (4 * i + 0)] = __float2half(v.x);
				target[targetStride * (4 * i + 1)] = __float2half(v.y);
				target[targetStride * (4 * i + 2)] = __float2half(v.z);
				target[targetStride * (4 * i + 3)] = __float2half(v.w);

				////DEBUG
				//if (threadIdx.x<4)
				//{
				//	printf("[%d] Grid Sampling %d: %.3f, %.3f, %.3f, %.3f\n",
				//		threadIdx.x, i, v.x, v.y, v.z, v.w);
				//}
			}
		}
		//adjoint-code of the one above
		//returns the adjoint of the position
		template<int GridEncoding>
		static __device__ __inline__ real3 LoadVolumetricFeaturesAdjoint(
			float3 position, int startChannelDiv16, const half* adjTarget, int targetStride)
		{
			//textures store groups of 4 channels -> four textures to fill the 16 channels

			//I would need to perform manual tri-linear interpolation here
			// to compute the "perfect" derivatives of tri-linear interpolation.
			//However, this is kind of awkward, since I don't have the volume
			// saved as 3D tensor where I can do perfect integer-voxel access.
			// I would need to simulate it using tex3D and take care of
			// computing the correct voxel position in normalized coordinates.
			//Instead: approximate gradients via central differences.
			// For an infinitesimal small step size and perfect precision,
			// this is exactly the same.
			//It seems using 1/4 of the grid size is the best choice.

			float h = volumeInterpolationTensorcoresParameters.latentGridDifferencesStepSize;
			float s = 1 / (2 * h);
			real3 adjPosition = make_real3(0, 0, 0);

			////DEBUG
			//if (threadIdx.x == 0) {
			//	printf("latentGridDifferencesStepSize=%.5f -> s=%.5f\n", h, s);
			//}
			//if (threadIdx.x < 4)
			//{
			//	printf("[%d] Grid Position: %.5f, %.5f, %.5f\n",
			//		threadIdx.x, position.x, position.y, position.z);
			//}

			for (int i = 0; i < 4; ++i)
			{
				const int texIndex = 4 * startChannelDiv16 + i;
				float4 time = volumeInterpolationTensorcoresParameters.cLatentGridInterpolation[texIndex];
				time = fracf(time);

				//fetch adjoint target
				float4 adjC = make_float4(
					__half2float(adjTarget[targetStride * (4 * i + 0)]),
					__half2float(adjTarget[targetStride * (4 * i + 1)]),
					__half2float(adjTarget[targetStride * (4 * i + 2)]),
					__half2float(adjTarget[targetStride * (4 * i + 3)])
				);

				////DEBUG
				//if (threadIdx.x < 4)
				//{
				//	printf("[%d] Grid Sampling, adjC %d: %.5f, %.5f, %.5f, %.5f\n",
				//		threadIdx.x, i, adjC.x, adjC.y, adjC.z, adjC.w);
				//}

				//dX
				{
					float4 vA_high = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridA[texIndex],
						position.x + h, position.y, position.z);
					float4 vA_low = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridA[texIndex],
						position.x - h, position.y, position.z);
					float4 vB_high = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridB[texIndex],
						position.x + h, position.y, position.z);
					float4 vB_low = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridB[texIndex],
						position.x - h, position.y, position.z);
					vA_high = encoding::EncodeGridValue<GridEncoding, false>(vA_high, texIndex);
					vA_low = encoding::EncodeGridValue<GridEncoding, false>(vA_low, texIndex);
					vB_high = encoding::EncodeGridValue<GridEncoding, false>(vB_high, texIndex);
					vB_low = encoding::EncodeGridValue<GridEncoding, false>(vB_low, texIndex);
					float4 v_high = lerp(vA_high, vB_high, time);
					float4 v_low = lerp(vA_low, vB_low, time);
					float4 gradX = s * (v_high - v_low);
					adjPosition.x += dot(adjC, gradX);
				}

				//dY
				{
					float4 vA_high = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridA[texIndex],
						position.x, position.y + h, position.z);
					float4 vA_low = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridA[texIndex],
						position.x, position.y - h, position.z);
					float4 vB_high = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridB[texIndex],
						position.x, position.y + h, position.z);
					float4 vB_low = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridB[texIndex],
						position.x, position.y - h, position.z);
					vA_high = encoding::EncodeGridValue<GridEncoding, false>(vA_high, texIndex);
					vA_low = encoding::EncodeGridValue<GridEncoding, false>(vA_low, texIndex);
					vB_high = encoding::EncodeGridValue<GridEncoding, false>(vB_high, texIndex);
					vB_low = encoding::EncodeGridValue<GridEncoding, false>(vB_low, texIndex);
					float4 v_high = lerp(vA_high, vB_high, time);
					float4 v_low = lerp(vA_low, vB_low, time);
					float4 gradY = s * (v_high - v_low);
					adjPosition.y += dot(adjC, gradY);
				}

				//dZ
				{
					float4 vA_high = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridA[texIndex],
						position.x, position.y, position.z + h);
					float4 vA_low = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridA[texIndex],
						position.x, position.y, position.z - h);
					float4 vB_high = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridB[texIndex],
						position.x, position.y, position.z + h);
					float4 vB_low = tex3D<float4>(volumeInterpolationTensorcoresParameters.cLatentGridB[texIndex],
						position.x, position.y, position.z - h);
					vA_high = encoding::EncodeGridValue<GridEncoding, false>(vA_high, texIndex);
					vA_low = encoding::EncodeGridValue<GridEncoding, false>(vA_low, texIndex);
					vB_high = encoding::EncodeGridValue<GridEncoding, false>(vB_high, texIndex);
					vB_low = encoding::EncodeGridValue<GridEncoding, false>(vB_low, texIndex);
					float4 v_high = lerp(vA_high, vB_high, time);
					float4 v_low = lerp(vA_low, vB_low, time);
					float4 gradZ = s * (v_high - v_low);
					adjPosition.z += dot(adjC, gradZ);
				}
			}

			////DEBUG
			//if (threadIdx.x < 4)
			//{
			//	printf("[%d] Grid Sampling, adjPosition: %.3f, %.3f, %.3f\n",
			//		threadIdx.x, adjPosition.x, adjPosition.y, adjPosition.z);
			//}

			return adjPosition;
		}

		/**
		 * Evaluates the networks.
		 * Important: The threads per warp must be synchronized!!
		 */
		template<typename Value_t>
		__device__ VolumeNetworkOutput<Value_t> eval(real3 position, real3 direction, int batch)
		{
			assert(__activemask() == 0xffffffff);
			//debug
			//const bool isDebug = blockIdx.x == 0 && threadIdx.x == 0;
			constexpr const bool isDebug = false;

			//Transform from [boxMin, boxMax] to [0,1]
			const real3 boxMax = getBoxMin() + getBoxSize();
			bool inside = all(position >= getBoxMin()) && all(position <= boxMax);
			position = (position - getBoxMin()) / getBoxSize();

			//warp and line ID
			assert(blockDim.x == BLOCK_SIZE);
			const int warpID = threadIdx.x / 32;
			const int lineID = threadIdx.x % 32;
			const float activParam = volumeInterpolationTensorcoresParameters.activationParam;

			//storage for the intermediate results
			half* intermediateResults = sIntermediateResults + 32 * HIDDEN_CHANNELS * warpID;
			
			/////////////////////////////////////
			/// NETWORK - FIRST LAYER
			/////////////////////////////////////

			//DEBUG
			if (isDebug)
			{
				printf("[%d] position: %.3f, %.3f, %.3f\n", threadIdx.x, position.x, position.y, position.z);
			}

			// first layer (col-major to avoid bank conflicts)
#if HAS_FOURIER_FEATURES==1
			{
				const half vx = __float2half(position.x);
				const half vy = __float2half(position.y);
				const half vz = __float2half(position.z);
				//forward position
				intermediateResults[HIDDEN_CHANNELS * lineID + 0] = vx;
				intermediateResults[HIDDEN_CHANNELS * lineID + 1] = vy;
				intermediateResults[HIDDEN_CHANNELS * lineID + 2] = vz;
#if PASS_TIME_TO_NETWORK==1
				const half time = __float2half(volumeInterpolationTensorcoresParameters.cLatentGridInterpolation[0].x);
				intermediateResults[HIDDEN_CHANNELS * lineID + 3] = time;
#else
				intermediateResults[HIDDEN_CHANNELS * lineID + 3] = 0;
#endif
				//forward direction
#if USE_DIRECTION>=1
				const half dx = __float2half(direction.x);
				const half dy = __float2half(direction.y);
				const half dz = __float2half(direction.z);
				intermediateResults[HIDDEN_CHANNELS * lineID + 4] = dx;
				intermediateResults[HIDDEN_CHANNELS * lineID + 5] = dy;
				intermediateResults[HIDDEN_CHANNELS * lineID + 6] = dz;
				intermediateResults[HIDDEN_CHANNELS * lineID + 7] = 0;
#endif
				//fourier features
				constexpr int fourierOffset = (USE_DIRECTION >= 1) ? 8 : 4;
				for (int i = 0; i < NUM_FOURIER_FEATURES; ++i)
				{
					half c = __hmul(vx, sWeightsFourier[i + NUM_FOURIER_FEATURES * 0]);
					c = __hfma(vy, sWeightsFourier[i + NUM_FOURIER_FEATURES * 1], c);
					c = __hfma(vz, sWeightsFourier[i + NUM_FOURIER_FEATURES * 2], c);
#if USE_DIRECTION==2
					c = __hfma(dx, sWeightsFourier[i + NUM_FOURIER_FEATURES * 3], c);
					c = __hfma(dy, sWeightsFourier[i + NUM_FOURIER_FEATURES * 4], c);
					c = __hfma(dz, sWeightsFourier[i + NUM_FOURIER_FEATURES * 5], c);
#endif
					intermediateResults[HIDDEN_CHANNELS * lineID + fourierOffset + i] = hcos(c);
					intermediateResults[HIDDEN_CHANNELS * lineID + fourierOffset + NUM_FOURIER_FEATURES + i] = hsin(c);
				}
			}
#else
			for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
			{
				half c = sBiasFirst[cout];
				c = __hfma(__float2half(position.x), sWeightsFirst[cout + HIDDEN_CHANNELS * 0], c);
				c = __hfma(__float2half(position.y), sWeightsFirst[cout + HIDDEN_CHANNELS * 1], c);
				c = __hfma(__float2half(position.z), sWeightsFirst[cout + HIDDEN_CHANNELS * 2], c);
#if USE_DIRECTION>=1
				c = __hfma(__float2half(direction.x), sWeightsFirst[cout + HIDDEN_CHANNELS * 3], c);
				c = __hfma(__float2half(direction.y), sWeightsFirst[cout + HIDDEN_CHANNELS * 4], c);
				c = __hfma(__float2half(direction.z), sWeightsFirst[cout + HIDDEN_CHANNELS * 5], c);
#endif
				c = ::kernel::activations::ACTIVATION::forward(c, activParam);
				intermediateResults[HIDDEN_CHANNELS * lineID + cout] = c;
			}
#endif

			////DEBUG
			//if (blockIdx.x==0 && threadIdx.x==0)
			//{
			//	printf("[0] First layer:");
			//	for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
			//		printf(" %.3f", __half2float(intermediateResults[HIDDEN_CHANNELS * lineID + cout]));
			//	printf("\n");
			//}

			/////////////////////////////////////
			/// NETWORK - LATENT GRID
			/////////////////////////////////////

#if HAS_LATENT_GRID==1
			//Special first layer that combines fourier features and the latent vector from the grid.
			//This is done to keep the memory for the intermediate results low.
			{
				static constexpr int Cout16 = HIDDEN_CHANNELS_DIV16;
				static constexpr int CinFourier16 = HIDDEN_CHANNELS_DIV16;
				static constexpr int CinGrid16 = LATENT_GRID_CHANNELS_DIV16;
				using namespace nvcuda::wmma;
				//accumulator kept in registers, later written to the shared memory,
				//because at the moment, the shared memory is needed for the latent vectors + fourier features
				fragment<accumulator, 16, 16, 16, half> c_frag[HIDDEN_CHANNELS_DIV16][2];
				//weights
				fragment<matrix_a, 16, 16, 16, half, row_major> a_frag[HIDDEN_CHANNELS_DIV16][HIDDEN_CHANNELS_DIV16]; //row,col
				//inputs
				fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[HIDDEN_CHANNELS_DIV16][2];

				//load C (bias)
				for (int cout = 0; cout < Cout16; ++cout)
				{
					load_matrix_sync(c_frag[cout][0], sBiasLatentGrid + 16 * cout, 0, mem_col_major);
					load_matrix_sync(c_frag[cout][1], sBiasLatentGrid + 16 * cout, 0, mem_col_major);
				}

				//PROCESS FOURIER FEATURES
				//load A (weights)
				for (int cout = 0; cout < Cout16; ++cout)
					for (int cin = 0; cin < CinFourier16; ++cin)
						load_matrix_sync(a_frag[cout][cin],
							sWeightsLatentGrid + 16 * cin + LATENT_GRID_LAYER_TOTAL_INPUTS * 16 * cout,
							LATENT_GRID_LAYER_TOTAL_INPUTS);

				//load B (input)
				for (int cin = 0; cin < CinFourier16; ++cin)
				{
					load_matrix_sync(b_frag[cin][0], intermediateResults + 16 * cin, HIDDEN_CHANNELS);
					load_matrix_sync(b_frag[cin][1], intermediateResults + 16 * cin + 16 * HIDDEN_CHANNELS, HIDDEN_CHANNELS);
				}

				//matmul
				for (int i = 0; i < Cout16; ++i) {
					for (int j = 0; j < 2; ++j) { //batch
						for (int k = 0; k < CinFourier16; ++k) {
							mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
						}
					}
				}

				//PROCESS VOLUMETRIC FEATURES
				//processed in chunks of 16 input channels
				for (int cin = 0; cin < CinGrid16; ++cin)
				{
					//fetch and interpolate the volumetric features (column major)
					LoadVolumetricFeatures<LATENT_GRID_ENCODING>(make_float3(position),
						cin, intermediateResults+lineID*16, 1);

					//and load into B
					load_matrix_sync(b_frag[0][0], intermediateResults, 16);
					load_matrix_sync(b_frag[0][1], intermediateResults + 16 * 16, 16);

					//load weights
					for (int cout = 0; cout < Cout16; ++cout)
						load_matrix_sync(a_frag[cout][0],
							sWeightsLatentGrid + 16 * (cin + HIDDEN_CHANNELS_DIV16) + LATENT_GRID_LAYER_TOTAL_INPUTS * 16 * cout,
							LATENT_GRID_LAYER_TOTAL_INPUTS);

					//matmul
					for (int i = 0; i < Cout16; ++i) {
						for (int j = 0; j < 2; ++j) { //batch
							mma_sync(c_frag[i][j], a_frag[i][0], b_frag[0][j], c_frag[i][j]);
						}
					}
				}

#if GRADIENT_MODE == GRADIENT_MODE_ADJOINT_METHOD
				//copy to shared for the adjoint pass
				half* adjointStorage = sStorageForAdjointHidden + 32 * HIDDEN_CHANNELS * warpID;
				for (int cout = 0; cout < Cout16; ++cout)
				{
					store_matrix_sync(adjointStorage + 16 * cout, c_frag[cout][0], HIDDEN_CHANNELS, mem_col_major);
					store_matrix_sync(adjointStorage + 16 * cout + 16 * HIDDEN_CHANNELS, c_frag[cout][1], HIDDEN_CHANNELS, mem_col_major);
				}
#endif

				//ACTIVATION FUNCTION
				for (int i = 0; i < Cout16; ++i) {
					for (int j = 0; j < 2; ++j) { //batch
						for (int t = 0; t < c_frag[0][0].num_elements; t++) {
							c_frag[i][j].x[t] = ::kernel::activations::ACTIVATION::forward(c_frag[i][j].x[t], activParam);
						}
					}
				}

				//COPY TO SHARED
				for (int cout = 0; cout < Cout16; ++cout)
				{
					store_matrix_sync(intermediateResults + 16 * cout, c_frag[cout][0], HIDDEN_CHANNELS, mem_col_major);
					store_matrix_sync(intermediateResults + 16 * cout + 16 * HIDDEN_CHANNELS, c_frag[cout][1], HIDDEN_CHANNELS, mem_col_major);
				}

				////DEBUG
				//if (blockIdx.x == 0 && threadIdx.x == 0)
				//{
				//	for (int test = 0; test < 4; ++test) {
				//		printf("[%d] Grid layer:", test);
				//		for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
				//			printf(" %.3f", __half2float(intermediateResults[HIDDEN_CHANNELS * test/*lineID*/ + cout]));
				//		printf("\n");
				//	}
				//}
			}
#endif

			/////////////////////////////////////
			/// NETWORK - HIDDEN LAYERS
			/////////////////////////////////////

			// hidden layers
			for (int hidden = 0; hidden < NUM_HIDDEN_LAYERS; ++hidden)
			{
				static constexpr int Cin16 = HIDDEN_CHANNELS_DIV16;
				static constexpr int Cout16 = HIDDEN_CHANNELS_DIV16;

				//Cout = Cin = Batch=32
				using namespace nvcuda::wmma;
				fragment<matrix_a, 16, 16, 16, half, row_major> a_frag[HIDDEN_CHANNELS_DIV16][HIDDEN_CHANNELS_DIV16]; //row,col
				fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[HIDDEN_CHANNELS_DIV16][2];
				fragment<accumulator, 16, 16, 16, half> c_frag[HIDDEN_CHANNELS_DIV16][2];

				//load C (bias)
				for (int cout = 0; cout < Cout16; ++cout)
				{
					load_matrix_sync(c_frag[cout][0], sBiasHidden + HIDDEN_CHANNELS * hidden + 16 * cout, 0, mem_col_major);
					load_matrix_sync(c_frag[cout][1], sBiasHidden + HIDDEN_CHANNELS * hidden + 16 * cout, 0, mem_col_major);
				}

				//load A (weights)
				for (int cout = 0; cout < Cout16; ++cout)
					for (int cin = 0; cin < Cin16; ++cin)
						load_matrix_sync(a_frag[cout][cin],
							sWeightsHidden + HIDDEN_CHANNELS * HIDDEN_CHANNELS * hidden + 16 * cin + HIDDEN_CHANNELS * 16 * cout,
							HIDDEN_CHANNELS);

				//load B (input)
				for (int cin = 0; cin < Cin16; ++cin)
				{
					load_matrix_sync(b_frag[cin][0], intermediateResults + 16 * cin, HIDDEN_CHANNELS);
					load_matrix_sync(b_frag[cin][1], intermediateResults + 16 * cin + 16 * HIDDEN_CHANNELS, HIDDEN_CHANNELS);
				}

				//matmul
				for (int i = 0; i < Cout16; ++i) {
					for (int j = 0; j < 2; ++j) {
						for (int k = 0; k < Cin16; ++k) {
							mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
						}
					}
				}

#if GRADIENT_MODE == GRADIENT_MODE_ADJOINT_METHOD
				//copy to shared for adjoint pass
				half* adjointStorage = sStorageForAdjointHidden + (
					HIDDEN_CHANNELS * (32 * warpID + (hidden+HAS_LATENT_GRID) * BLOCK_SIZE));
				for (int cout = 0; cout < Cout16; ++cout)
				{
					store_matrix_sync(adjointStorage + 16 * cout, c_frag[cout][0], HIDDEN_CHANNELS, mem_col_major);
					store_matrix_sync(adjointStorage + 16 * cout + 16 * HIDDEN_CHANNELS, c_frag[cout][1], HIDDEN_CHANNELS, mem_col_major);
				}
#endif

				//activations
				for (int i = 0; i < Cout16; ++i) {
					for (int j = 0; j < 2; ++j) {
						for (int t = 0; t < c_frag[0][0].num_elements; t++)
						{
							c_frag[i][j].x[t] = ::kernel::activations::ACTIVATION::forward(c_frag[i][j].x[t], activParam);
						}
					}
				}

				//copy to shared
				for (int cout = 0; cout < Cout16; ++cout)
				{
					store_matrix_sync(intermediateResults + 16 * cout, c_frag[cout][0], HIDDEN_CHANNELS, mem_col_major);
					store_matrix_sync(intermediateResults + 16 * cout + 16 * HIDDEN_CHANNELS, c_frag[cout][1], HIDDEN_CHANNELS, mem_col_major);
				}

				////DEBUG
				//if (blockIdx.x == 0 && threadIdx.x == 0)
				//{
				//	printf("[%d] Hidden layer:", hidden+1);
				//	for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
				//		printf(" %.3f", __half2float(intermediateResults[HIDDEN_CHANNELS * lineID + cout]));
				//	printf("\n");
				//}
			}

			/////////////////////////////////////
			/// NETWORK - OUTPUT PARAMETRIZATION
			/////////////////////////////////////
			
			__syncwarp();
			//----------------------------------
#if (OUTPUT_MODE==OUTPUT_MODE_RGBO) || (OUTPUT_MODE==OUTPUT_MODE_RGBO_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_CUBIC)
			//rgbo [direct] and density-gradient [direct]
			half2 out1 = sBiasLast[0];
			half2 out2 = sBiasLast[1];
			for (int cin = 0; cin < HIDDEN_CHANNELS; ++cin)
			{
				half v = intermediateResults[HIDDEN_CHANNELS * lineID + cin];
				half2 v2(v, v);
				out1 = __hfma2(v2, sWeightsLast[2 * cin], out1);
				out2 = __hfma2(v2, sWeightsLast[2 * cin + 1], out2);
			}
#if (OUTPUT_MODE == OUTPUT_MODE_RGBO)
			const real4 value = make_real4(
				::kernel::activations::Sigmoid::forward(__half2float(out1.x), activParam),   //red
				::kernel::activations::Sigmoid::forward(__half2float(out1.y), activParam),   //green
				::kernel::activations::Sigmoid::forward(__half2float(out2.x), activParam),   //blue
				::kernel::activations::Softplus::forward(__half2float(out2.y), activParam)); //absorption
			return VolumeNetworkOutput<real4>{value, value, inside};
#elif (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT)
			//const real4 value = make_real4(__half2float(out1.x), __half2float(out1.y), __half2float(out2.x), __half2float(out2.y));
			const real4 valueNoClamp = make_real4(
				__half2float(out1.x), __half2float(out1.y),
				__half2float(out2.x), __half2float(out2.y)
			);
			const real4 value = make_real4(
				clamp(valueNoClamp.x, 0.f, 1.f), //red
				clamp(valueNoClamp.y, 0.f, 1.f), //green
				clamp(valueNoClamp.z, 0.f, 1.f), //blue
				rmax(valueNoClamp.w, 0.f)); //absorption
			return VolumeNetworkOutput<real4>{value, valueNoClamp, inside};
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT)
#if GRADIENT_MODE == GRADIENT_MODE_ADJOINT_METHOD
			sStorageForAdjointLast[threadIdx.x] = out1.x;
#endif
			const real_t value = ::kernel::activations::Sigmoid::forward(__half2float(out1.x), activParam);
			const real3 normal = make_real3(
				__half2float(out1.y), __half2float(out2.x), __half2float(out2.y));
			return VolumeNetworkOutput<real_t>{value, value, inside, normal};
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_DIRECT)
#if GRADIENT_MODE == GRADIENT_MODE_ADJOINT_METHOD
			sStorageForAdjointLast[threadIdx.x] = out1.x;
#endif
			const real_t valueNoClamp = __half2float(out1.x);
			        //No more clamping, done in the TF
			const real_t value = valueNoClamp; // ::kernel::activations::Clamp::forward(valueNoClamp, activParam);
			const real3 normal = make_real3(
				__half2float(out1.y), __half2float(out2.x), __half2float(out2.y));
			return VolumeNetworkOutput<real_t>{value, valueNoClamp, inside, normal};
#else // (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_CUBIC)
#if GRADIENT_MODE == GRADIENT_MODE_ADJOINT_METHOD
			sStorageForAdjointLast[threadIdx.x] = out1.x;
#endif
			const real_t valueNoClamp = __half2float(out1.x);
			        //No more clamping, done in the TF
			const real_t value = valueNoClamp; // ::kernel::activations::Clamp::forward(valueNoClamp, activParam);
			const real3 normal = make_real3(
				cube(__half2float(out1.y)), cube(__half2float(out2.x)), cube(__half2float(out2.y)));
			return VolumeNetworkOutput<real_t>{value, valueNoClamp, inside, normal};
#endif

			//----------------------------------
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT)
			//rgbo [direct] and density-gradient [direct]
			half2 out1 = sBiasLast[0];
			half2 out2 = sBiasLast[1];
			half2 out3 = sBiasLast[2];
			for (int cin = 0; cin < HIDDEN_CHANNELS; ++cin)
			{
				half v = intermediateResults[HIDDEN_CHANNELS * lineID + cin];
				half2 v2(v, v);
				out1 = __hfma2(v2, sWeightsLast[3 * cin], out1);
				out2 = __hfma2(v2, sWeightsLast[3 * cin + 1], out2);
				out3 = __hfma2(v2, sWeightsLast[3 * cin + 2], out3);
			}
#if (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE)
			const real_t value = ::kernel::activations::Sigmoid::forward(__half2float(out1.x), activParam);
			const real3 normal = make_real3(
				__half2float(out1.y), __half2float(out2.x), __half2float(out2.y));
			const real2 curvature = make_real2(
				__half2float(out3.x), __half2float(out3.y));
			return VolumeNetworkOutput<real_t>{value, value, inside, normal, curvature};
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT)
			const real_t valueNoClamp = __half2float(out1.x);
			        //No more clamping, done in the TF
			const real_t value = valueNoClamp; // ::kernel::activations::Clamp::forward(valueNoClamp, activParam);
			const real3 normal = make_real3(
				__half2float(out1.y), __half2float(out2.x), __half2float(out2.y));
			const real2 curvature = make_real2(
				__half2float(out3.x), __half2float(out3.y));
			return VolumeNetworkOutput<real_t>{value, valueNoClamp, inside, normal, curvature};
#endif

			//----------------------------------
#elif (OUTPUT_MODE == OUTPUT_MODE_DENSITY) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_DIRECT)
			//density [direct]
			half out = sBiasLast;
			for (int cin = 0; cin < HIDDEN_CHANNELS; ++cin)
			{
				half v = intermediateResults[HIDDEN_CHANNELS * lineID + cin];
				out = __hfma(v, sWeightsLast[cin], out);
			}
#if GRADIENT_MODE == GRADIENT_MODE_ADJOINT_METHOD
			sStorageForAdjointLast[threadIdx.x] = out;
			if (isDebug) printf("  Store last: %.4f\n", __half2float(out));
#endif

#if OUTPUT_MODE == OUTPUT_MODE_DENSITY
			const real_t value = static_cast<real_t>(
				::kernel::activations::Sigmoid::forward(__half2float(out), activParam));
			const real_t valueNoClamp = value;
#else //OUTPUT_MODE_DENSITY_DIRECT
			const real_t valueNoClamp = __half2float(out);
			        //No more clamping, done in the TF
			const real_t value = valueNoClamp; // ::kernel::activations::Clamp::forward(valueNoClamp, activParam);
#endif
			return VolumeNetworkOutput<real_t>{value, valueNoClamp, inside};
#else
#error "Unknown output mode, forgot to handle the new case here?"
#endif

			// THE END
		}

		template<typename Value_t>
		__device__ __inline__ real3 evalNormal(real3 position, real3 direction,
			const VolumeNetworkOutput<Value_t>& resultFromEval, int batch)
		{
#if GRADIENT_MODE == GRADIENT_MODE_OFF_OR_DIRECT

#if (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT) || \
    (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_CUBIC) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE) || \
    (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT)
			//network predicts gradients as well as the density
			return resultFromEval.normal;
#else
			return make_real3(0); //not supported

#endif

#elif GRADIENT_MODE == GRADIENT_MODE_FINITE_DIFFERENCES
			static_assert(kernel::is_same_v<Value_t, real_t>, "output must be real_t");
			float h = volumeInterpolationTensorcoresParameters.finiteDifferencesStepsize;
			return make_real3(
				eval<Value_t>(position + make_real3(h, 0, 0), direction, batch).valueNoClamping
				- eval<Value_t>(position - make_real3(h, 0, 0), direction, batch).valueNoClamping,
				eval<Value_t>(position + make_real3(0, h, 0), direction, batch).valueNoClamping
				- eval<Value_t>(position - make_real3(0, h, 0), direction, batch).valueNoClamping,
				eval<Value_t>(position + make_real3(0, 0, h), direction, batch).valueNoClamping
				- eval<Value_t>(position - make_real3(0, 0, h), direction, batch).valueNoClamping
			) / (2 * h);

#elif GRADIENT_MODE == GRADIENT_MODE_ADJOINT_METHOD
			static_assert(kernel::is_same_v<Value_t, real_t>, "output must be real_t");
			return _evalNormalAdjoint(position, direction, resultFromEval, batch);
#endif
		}

		__device__ __inline__ real3 _evalNormalAdjoint(real3 position, real3 direction,
			const VolumeNetworkOutput<real_t>& resultFromEval, int batch)
		{
#if GRADIENT_MODE == GRADIENT_MODE_ADJOINT_METHOD
			//adjoint version of .eval()
			assert(__activemask() == 0xffffffff);
			//debug
			//const bool isDebug = blockIdx.x == 0 && threadIdx.x == 0;
			constexpr const bool isDebug = false;

			//Transform from [boxMin, boxMax] to [0,1]
			position = (position - getBoxMin()) / getBoxSize();

			//warp and line ID
			assert(blockDim.x == BLOCK_SIZE);
			const int warpID = threadIdx.x / 32;
			const int lineID = threadIdx.x % 32;
			const float activParam = volumeInterpolationTensorcoresParameters.activationParam;

			real3 outputAdjointPosition = make_real3(0);

			//storage for the intermediate results
			//these actually store the adjoint values
			half* adjIntermediateResults = sIntermediateResults + 32 * HIDDEN_CHANNELS * warpID;

			/////////////////////////////////////
			/// ADJOINT: NETWORK - OUTPUT PARAMETRIZATION
			/////////////////////////////////////

#if (OUTPUT_MODE == OUTPUT_MODE_DENSITY) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_DIRECT) || \
	(OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT_CUBIC)

			real_t adjValue = 1; //input adjoint of the final output value
			real_t out = __half2float(sStorageForAdjointLast[threadIdx.x]);
			if (isDebug) printf("  Load last: %.4f\n", out);
#if (OUTPUT_MODE == OUTPUT_MODE_DENSITY) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_GRADIENT)
			half adjOut = __float2half(::kernel::activations::Sigmoid::adjoint(
				out, adjValue, activParam));
#else //DIRECT or CUBIC alternatives
			//use unclamped versions
			half adjOut = __float2half(adjValue);
			//half adjOut = __float2half(::kernel::activations::Clamp::adjoint(
			//	out, adjValue, activParam));
#endif

#if (OUTPUT_MODE == OUTPUT_MODE_DENSITY) || (OUTPUT_MODE == OUTPUT_MODE_DENSITY_DIRECT)
			for (int cin = 0; cin < HIDDEN_CHANNELS; ++cin)
			{
				adjIntermediateResults[HIDDEN_CHANNELS * lineID + cin] =
					__hmul(sWeightsLast[cin], adjOut);
			}
#else
			//density gradient, only use density channel
			for (int cin = 0; cin < HIDDEN_CHANNELS; ++cin)
			{
				adjIntermediateResults[HIDDEN_CHANNELS * lineID + cin] =
					__hmul(sWeightsLast[2*cin].x, adjOut);
			}
#endif

#else
			assert(false, "Adjoint-code for RGBO not supported");
#endif

			/////////////////////////////////////
			/// ADJOINT: NETWORK - HIDDEN LAYERS
			/////////////////////////////////////

			const half ZERO = __float2half(0.f);
			for (int hidden = NUM_HIDDEN_LAYERS - 1; hidden >= 0; --hidden)
			{
				//adjoint of the activation function.
				//Has to be performed in shared memory because we can't make asumptions
				//about the memory layout / indexing of the fragments
				const half* adjointStorage = sStorageForAdjointHidden + (
					HIDDEN_CHANNELS * (32 * warpID + (hidden + HAS_LATENT_GRID) * BLOCK_SIZE));
				for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
				{
					const half zAdj = adjIntermediateResults[HIDDEN_CHANNELS * lineID + cout];
					const half x = adjointStorage[HIDDEN_CHANNELS * lineID + cout];
					const half xAdj = ::kernel::activations::ACTIVATION::adjoint(
						x, zAdj, activParam);
					adjIntermediateResults[HIDDEN_CHANNELS * lineID + cout] = xAdj;
				}

				//adjoint of the matrix multiplication (Weight*Values)

				static constexpr int Cin16 = HIDDEN_CHANNELS_DIV16;
				static constexpr int Cout16 = HIDDEN_CHANNELS_DIV16;

				//Cout = Cin = Batch=32
				using namespace nvcuda::wmma;
				fragment<matrix_a, 16, 16, 16, half, col_major> a_frag[HIDDEN_CHANNELS_DIV16][HIDDEN_CHANNELS_DIV16]; //row,col
				fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[HIDDEN_CHANNELS_DIV16][2];
				fragment<accumulator, 16, 16, 16, half> c_frag[HIDDEN_CHANNELS_DIV16][2];

				//fill output with zeros
				for (int cout = 0; cout < Cout16; ++cout)
				{
					fill_fragment(c_frag[cout][0], ZERO);
					fill_fragment(c_frag[cout][1], ZERO);
				}

				//load A (weights) transposed!
				for (int cout = 0; cout < Cout16; ++cout)
					for (int cin = 0; cin < Cin16; ++cin)
						load_matrix_sync(a_frag[cin][cout],
							sWeightsHidden + HIDDEN_CHANNELS * HIDDEN_CHANNELS * hidden + 16 * cin + HIDDEN_CHANNELS * 16 * cout,
							HIDDEN_CHANNELS);

				//load B (input = adjOutput)
				for (int cin = 0; cin < Cin16; ++cin)
				{
					load_matrix_sync(b_frag[cin][0], adjIntermediateResults + 16 * cin, HIDDEN_CHANNELS);
					load_matrix_sync(b_frag[cin][1], adjIntermediateResults + 16 * cin + 16 * HIDDEN_CHANNELS, HIDDEN_CHANNELS);
				}

				//matmul
				for (int i = 0; i < Cout16; ++i) {
					for (int j = 0; j < 2; ++j) {
						for (int k = 0; k < Cin16; ++k) {
							mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
						}
					}
				}

				//copy to shared
				for (int cout = 0; cout < Cout16; ++cout)
				{
					store_matrix_sync(adjIntermediateResults + 16 * cout, c_frag[cout][0], HIDDEN_CHANNELS, mem_col_major);
					store_matrix_sync(adjIntermediateResults + 16 * cout + 16 * HIDDEN_CHANNELS, c_frag[cout][1], HIDDEN_CHANNELS, mem_col_major);
				}
			}

			/////////////////////////////////////
			/// ADJOINT: NETWORK - LATENT GRID
			/////////////////////////////////////

#if HAS_LATENT_GRID==1
//Special first layer that combines fourier features and the latent vector from the grid.
//This is done to keep the memory for the intermediate results low.
			{
				static constexpr int Cout16 = HIDDEN_CHANNELS_DIV16;
				static constexpr int CinFourier16 = HIDDEN_CHANNELS_DIV16;
				static constexpr int CinGrid16 = LATENT_GRID_CHANNELS_DIV16;

				//adjoint of the activation function.
				half* adjointStorage = sStorageForAdjointHidden + HIDDEN_CHANNELS * 32 * warpID;
				for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
				{
					const half zAdj = adjIntermediateResults[HIDDEN_CHANNELS * lineID + cout];
					const half x = adjointStorage[HIDDEN_CHANNELS * lineID + cout];
					const half xAdj = ::kernel::activations::ACTIVATION::adjoint(
						x, zAdj, activParam);
					adjIntermediateResults[HIDDEN_CHANNELS * lineID + cout] = xAdj;
				}

				//DEBUG
                if (isDebug)
                {
	                for (int test = 0; test < 4; ++test) {
		                printf("[%d] adj latent grid, adj-output (*100):", test);
		                for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
			                printf(" %.3f", __half2float(adjIntermediateResults[HIDDEN_CHANNELS * test/*lineID*/ + cout]) * 100.f);
		                printf("\n");
	                }
                }

				using namespace nvcuda::wmma;
				//accumulator kept in registers, later written to the shared memory,
				//because at the moment, the shared memory is needed for the latent vectors + fourier features
				fragment<accumulator, 16, 16, 16, half> c_frag[HIDDEN_CHANNELS_DIV16][2];
				//weights
				fragment<matrix_a, 16, 16, 16, half, col_major> a_frag[HIDDEN_CHANNELS_DIV16][HIDDEN_CHANNELS_DIV16]; //row,col
				//inputs
				fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[HIDDEN_CHANNELS_DIV16][2];

				//load B (input = adjOutput)
				for (int cout = 0; cout < Cout16; ++cout)
				{
					load_matrix_sync(b_frag[cout][0], adjIntermediateResults + 16 * cout, HIDDEN_CHANNELS);
					load_matrix_sync(b_frag[cout][1], adjIntermediateResults + 16 * cout + 16 * HIDDEN_CHANNELS, HIDDEN_CHANNELS);
				}

				//PROCESS VOLUMETRIC FEATURES
				//processed in chunks of 16 input channels
				for (int cin = 0; cin < CinGrid16; ++cin)
				{
					//reset C (adj of input)
					fill_fragment(c_frag[0][0], ZERO);
					fill_fragment(c_frag[0][1], ZERO);

					//load weights (transposed)
					for (int cout = 0; cout < Cout16; ++cout)
						load_matrix_sync(a_frag[0][cout],
							sWeightsLatentGrid + 16 * (cin + HIDDEN_CHANNELS_DIV16) + LATENT_GRID_LAYER_TOTAL_INPUTS * 16 * cout,
							LATENT_GRID_LAYER_TOTAL_INPUTS);

					//matmul
					for (int cout = 0; cout < Cout16; ++cout) {
						for (int j = 0; j < 2; ++j) { //batch
							mma_sync(c_frag[0][j], a_frag[0][cout], b_frag[cout][j], c_frag[0][j]);
						}
					}

					//save c_frag to shared (adjoint of input)
					//we can use adjointStorage since we don't need it anymore
					store_matrix_sync(adjointStorage, c_frag[0][0], 16, mem_col_major);
					store_matrix_sync(adjointStorage + 16 * 16, c_frag[0][1], 16, mem_col_major);

					//adjoint of volumetric features
					outputAdjointPosition += LoadVolumetricFeaturesAdjoint<LATENT_GRID_ENCODING>(make_float3(position),
						cin, adjointStorage + lineID * 16, 1);
				}

				//PROCESS FOURIER FEATURES
				//fill output with zeros
				for (int cout = 0; cout < Cout16; ++cout)
				{
					fill_fragment(c_frag[cout][0], ZERO);
					fill_fragment(c_frag[cout][1], ZERO);
				}

				//load A (weights) transposed!
				for (int cout = 0; cout < Cout16; ++cout)
					for (int cin = 0; cin < CinFourier16; ++cin)
						load_matrix_sync(a_frag[cin][cout],
							sWeightsLatentGrid + 16 * cin + LATENT_GRID_LAYER_TOTAL_INPUTS * 16 * cout,
							LATENT_GRID_LAYER_TOTAL_INPUTS);

				//matmul
				for (int i = 0; i < Cout16; ++i) {
					for (int j = 0; j < 2; ++j) { //batch
						for (int k = 0; k < CinFourier16; ++k) {
							mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
						}
					}
				}

				//copy to shared
				for (int cout = 0; cout < Cout16; ++cout)
				{
					store_matrix_sync(adjIntermediateResults + 16 * cout, c_frag[cout][0], HIDDEN_CHANNELS, mem_col_major);
					store_matrix_sync(adjIntermediateResults + 16 * cout + 16 * HIDDEN_CHANNELS, c_frag[cout][1], HIDDEN_CHANNELS, mem_col_major);
				}

				//DEBUG
				if (isDebug)
				{
					for (int test = 0; test < 4; ++test) {
						printf("[%d] adj latent grid, adj-input (*100):", test);
						for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
							printf(" %.3f", __half2float(adjIntermediateResults[HIDDEN_CHANNELS * test/*lineID*/ + cout]) * 100.f);
						printf("\n");
					}
				}
			}
#endif

			/////////////////////////////////////
			/// ADJOINT: NETWORK - FIRST LAYER
			/////////////////////////////////////

#if HAS_FOURIER_FEATURES==1
			{
				const half vx = __float2half(position.x);
				const half vy = __float2half(position.y);
				const half vz = __float2half(position.z);

				//adjoint: forward position
				outputAdjointPosition.x += __half2float(adjIntermediateResults[HIDDEN_CHANNELS * lineID + 0]);
				outputAdjointPosition.y += __half2float(adjIntermediateResults[HIDDEN_CHANNELS * lineID + 1]);
				outputAdjointPosition.z += __half2float(adjIntermediateResults[HIDDEN_CHANNELS * lineID + 2]);

				//adjoint: fourier features
				constexpr int fourierOffset = (USE_DIRECTION >= 1) ? 8 : 4;
				for (int i = 0; i < NUM_FOURIER_FEATURES; ++i)
				{
					//forward code to recompute c
					half c = __hmul(vx, sWeightsFourier[i + NUM_FOURIER_FEATURES * 0]);
					c = __hfma(vy, sWeightsFourier[i + NUM_FOURIER_FEATURES * 1], c);
					c = __hfma(vz, sWeightsFourier[i + NUM_FOURIER_FEATURES * 2], c);
					//intermediateResults[HIDDEN_CHANNELS * lineID + fourierOffset + i] = hcos(c);
					//intermediateResults[HIDDEN_CHANNELS * lineID + fourierOffset + NUM_FOURIER_FEATURES + i] = hsin(c);

					//adjoint code:
					half adjC = __hsub(
						__hmul( //sinus
							adjIntermediateResults[HIDDEN_CHANNELS * lineID + fourierOffset + NUM_FOURIER_FEATURES + i],
							hcos(c)
						),
						__hmul( //cosine
							adjIntermediateResults[HIDDEN_CHANNELS * lineID + fourierOffset + i],
							hsin(c)
						)
					);
					outputAdjointPosition.x += __half2float(
						__hmul(sWeightsFourier[i + NUM_FOURIER_FEATURES * 0], adjC));
					outputAdjointPosition.y += __half2float(
						__hmul(sWeightsFourier[i + NUM_FOURIER_FEATURES * 1], adjC));
					outputAdjointPosition.z += __half2float(
						__hmul(sWeightsFourier[i + NUM_FOURIER_FEATURES * 2], adjC));
				}
			}
#else
			for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
			{
				//recompute to get the input for the activation function
				half c = sBiasFirst[cout];
				c = __hfma(__float2half(position.x), sWeightsFirst[cout + HIDDEN_CHANNELS * 0], c);
				c = __hfma(__float2half(position.y), sWeightsFirst[cout + HIDDEN_CHANNELS * 1], c);
				c = __hfma(__float2half(position.z), sWeightsFirst[cout + HIDDEN_CHANNELS * 2], c);
#if USE_DIRECTION>=1
				c = __hfma(__float2half(direction.x), sWeightsFirst[cout + HIDDEN_CHANNELS * 3], c);
				c = __hfma(__float2half(direction.y), sWeightsFirst[cout + HIDDEN_CHANNELS * 4], c);
				c = __hfma(__float2half(direction.z), sWeightsFirst[cout + HIDDEN_CHANNELS * 5], c);
#endif
				//c2 = ::kernel::activations::ACTIVATION::forward(c, activParam);

				//adjoint:
				half adjC2 = adjIntermediateResults[HIDDEN_CHANNELS * lineID + cout];
				half adjC1 = ::kernel::activations::ACTIVATION::adjoint(c, adjC2, activParam);

				outputAdjointPosition.x += __half2float(
					__hmul(sWeightsFirst[cout + HIDDEN_CHANNELS * 0], adjC1));
				outputAdjointPosition.y += __half2float(
					__hmul(sWeightsFirst[cout + HIDDEN_CHANNELS * 1], adjC1));
				outputAdjointPosition.z += __half2float(
					__hmul(sWeightsFirst[cout + HIDDEN_CHANNELS * 2], adjC1));
			}
#endif

			return outputAdjointPosition;

#else
			return make_real3(0);
#endif
		}

		template<typename Value_t>
		__device__ __inline__ real2 evalCurvature(real3 position, real3 direction,
			const VolumeNetworkOutput<Value_t>& resultFromEval, int batch)
		{
#if (GRADIENT_MODE == GRADIENT_MODE_OFF_OR_DIRECT) && \
	((OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE) || \
    (OUTPUT_MODE == OUTPUT_MODE_DENSITY_CURVATURE_DIRECT))
			return resultFromEval.curvature;
#else
			//Curvature is only supported if the network directly estimates it
			__trap();
			return make_real2(0); //never called
#endif
		}
	};

}
