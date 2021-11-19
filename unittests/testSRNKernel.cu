#include <catch.hpp>

#include <string>
#include <vector>
#include <random>
#include <cuda_fp16.h>
#include <mma.h>
#include <device_launch_parameters.h>
#include <tinyformat.h>
#include <vector_types.h>
#include <cuMat/src/Errors.h>
#include <third-party/Eigen/Core> // in cuMat
#include "helper_math.cuh"

#define real4 float4
#define real_t float
#define make_real4 make_float4

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#pragma message "Current CUDA architecture: " CUMAT_STR(__CUDA_ARCH__)
#error "TensorCores require a CUDA architecture >= 700"
#endif

//config
#define BLOCK_SIZE 512
#define NUM_HIDDEN_LAYERS 0
#define HIDDEN_CHANNELS_DIV16 2
#define HAS_FOURIER_FEATURES 0
#define NUM_FOURIER_FEATURES ((HIDDEN_CHANNELS_DIV16*16-4)/2)
#define ACTIVATION ReLU

#define OUTPUT_MODE_DENSITY 0 //real_t output
#define OUTPUT_MODE_DENSITY_DIRECT 1 //real_t output
#define OUTPUT_MODE_RGBO 2   //real4 output
#define OUTPUT_MODE_RGBO_DIRECT 3   //real4 output
#define OUTPUT_MODE OUTPUT_MODE_DENSITY

#define FIRST_AND_LAST_IN_SHARED_MEMORY 0

#define HIDDEN_CHANNELS (16*(HIDDEN_CHANNELS_DIV16))
#if HAS_FOURIER_FEATURES==1
#if (4+2*NUM_FOURIER_FEATURES) != HIDDEN_CHANNELS
#error "(4+2*NUM_FOURIER_FEATURES) == HIDDEN_CHANNELS not satisfied"
#endif
#endif
#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT)
#define OUTPUT_IS_COLOR 1
#else
#define OUTPUT_IS_COLOR 0
#endif

#define MIN_ONE(val) (((val)>0)?(val):(1))

struct alignas(32) VolumeInterpolationTensorcoresParameters
{
#if HAS_FOURIER_FEATURES==1
	alignas(32) half cWeightsFourier[3 * NUM_FOURIER_FEATURES]; //column-major (output channels are fastest)
#else
	alignas(32) half cWeightsFirst[3 * HIDDEN_CHANNELS]; //column-major (output channels are fastest)
	alignas(32) half cBiasFirst[HIDDEN_CHANNELS];
#endif
	alignas(32) half cWeightsHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS * HIDDEN_CHANNELS)]; //row-major (input channels are fastest)
	alignas(32) half cBiasHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS)];
#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT)
	alignas(32) half2 cWeightsLast[HIDDEN_CHANNELS * 2]; //col-major (output channels are fastest)
	alignas(32) half2 cBiasLast[2];
#else
	alignas(32) half cWeightsLast[HIDDEN_CHANNELS];
	alignas(32) half cBiasLast;
#endif
	alignas(32) float3 boxMin;
	alignas(32) float3 boxSize;
};
__constant__ VolumeInterpolationTensorcoresParameters volumeInterpolationTensorcoresParameters;

//WARNING: all memory involved in tensor-core operations (wmma)
// must be aligned to 32 bytes.
__shared__ alignas(32) half sWeightsHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS * HIDDEN_CHANNELS)];
__shared__ alignas(32) half sBiasHidden[MIN_ONE(NUM_HIDDEN_LAYERS * HIDDEN_CHANNELS)];
__shared__ alignas(32) half sIntermediateResults[BLOCK_SIZE * HIDDEN_CHANNELS];

#if FIRST_AND_LAST_IN_SHARED_MEMORY==1

#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT)
__shared__ alignas(32) half2 sWeightsLast[HIDDEN_CHANNELS * 2];
__shared__ half2 sBiasLast[2];
#else
__shared__ alignas(32) half sWeightsLast[HIDDEN_CHANNELS];
__shared__ half sBiasLast;
#endif
#if HAS_FOURIER_FEATURES==1
__shared__ half sWeightsFourier[3 * NUM_FOURIER_FEATURES];
#else
__shared__ half sWeightsFirst[3 * HIDDEN_CHANNELS];
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

template<typename Value_t>
struct VolumeGridOutput
{
	Value_t value;
	bool isInside;
};

namespace kernel {
	namespace activations
	{
		template<typename T>
		__device__ __forceinline__ T None(const T& v) { return v; }

		__device__ __forceinline__ half ReLU(const half& v)
		{
#if __CUDA_ARCH__ >= 800 && defined(__CUDA_ARCH__)
			const half ZERO{ __half_raw{0} };
			return __hmax(v, ZERO);
#else
			return __float2half(fmaxf(0.f, __half2float(v)));
#endif
		}
		__device__ __forceinline__ half2 ReLU(const half2& v)
		{
#if __CUDA_ARCH__ >= 800 && defined(__CUDA_ARCH__)
			const half2 ZERO2{ __half_raw{0}, __half_raw{0} };
			return __h2max(v, ZERO2);
#else
			return half2{
				__float2half(fmaxf(0.f, __half2float(v.x))),
				__float2half(fmaxf(0.f, __half2float(v.y)))
			};
#endif
		}

		__device__ __forceinline__ half Sine(const half& v)
		{
			return hsin(v);
		}
		__device__ __forceinline__ half2 Sine(const half2& v)
		{
			return h2sin(v);
		}

		__device__ __forceinline__ half Sigmoid(const half& v)
		{
			const half ONE = __float2half(1.0f);
			return __hdiv(ONE, __hadd(ONE, hexp(__hneg(v))));
		}
		__device__ __forceinline__ half2 Sigmoid(const half2& v)
		{
#if __CUDA_ARCH__ >= 800 && defined(__CUDA_ARCH__)
			const half ONE = __float2half(1.0f);
			const half2 ONE2{ ONE, ONE };
			return __h2div(ONE2, __h2add(ONE2, h2exp(__h2neg(v))));
#else
			return half2{ Sigmoid(v.x), Sigmoid(v.y) };
#endif
		}
		__device__ __forceinline__ float Sigmoid(const float& v)
		{
			return 1.0f / (1.0f + expf(-v));
		}
		__device__ __forceinline__ double Sigmoid(const double& v)
		{
			return 1.0 / (1.0 + exp(-v));
		}

		__device__ __forceinline__ float Softplus(float x)
		{
			static constexpr float beta = 1.0f;
			static constexpr float inv_beta = 1.0f / beta;
			static constexpr float threshold = 20.0f;
			static constexpr float threshold_times_beta = beta * threshold;

			if (x > threshold_times_beta) return x;
			return inv_beta * logf(1 + expf(beta * x));
		}
		__device__ __forceinline__ double Softplus(double x)
		{
			static constexpr double beta = 1.0f;
			static constexpr double inv_beta = 1.0f / beta;
			static constexpr double threshold = 20.0f;
			static constexpr double threshold_times_beta = beta * threshold;

			if (x > threshold_times_beta) return x;
			return inv_beta * log(1 + exp(beta * x));
		}
	}
}

struct VolumeInterpolationTensorcores
{
	__device__ __inline__ VolumeInterpolationTensorcores()
	{
		assert(blockDim.x == BLOCK_SIZE);
		const int warpID = threadIdx.x / 32;
		const int lineID = threadIdx.x % 32;

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
					sWeightsFourier[i + NUM_FOURIER_FEATURES * 0] = volumeInterpolationTensorcoresParameters.cWeightsFourier[i + NUM_FOURIER_FEATURES * 0];
					sWeightsFourier[i + NUM_FOURIER_FEATURES * 1] = volumeInterpolationTensorcoresParameters.cWeightsFourier[i + NUM_FOURIER_FEATURES * 1];
					sWeightsFourier[i + NUM_FOURIER_FEATURES * 2] = volumeInterpolationTensorcoresParameters.cWeightsFourier[i + NUM_FOURIER_FEATURES * 2];
				}
			}
#else
			static constexpr int cinDiv32 = CUMAT_DIV_UP(HIDDEN_CHANNELS, 32);
#pragma unroll
			for (int c = 0; c < cinDiv32; ++c) {
				const int i = lineID + 32 * c;
				if (i < HIDDEN_CHANNELS) {
					sWeightsFirst[i + HIDDEN_CHANNELS * 0] = volumeInterpolationTensorcoresParameters.cWeightsFirst[i + HIDDEN_CHANNELS * 0];
					sWeightsFirst[i + HIDDEN_CHANNELS * 1] = volumeInterpolationTensorcoresParameters.cWeightsFirst[i + HIDDEN_CHANNELS * 1];
					sWeightsFirst[i + HIDDEN_CHANNELS * 2] = volumeInterpolationTensorcoresParameters.cWeightsFirst[i + HIDDEN_CHANNELS * 2];
					sBiasFirst[i] = volumeInterpolationTensorcoresParameters.cBiasFirst[i];
				}
			}
#endif
#endif
		}

		//hidden
		for (int layer = warpID; layer < NUM_HIDDEN_LAYERS; layer += blockDim.x)
		{
			static constexpr int coutDiv32 = CUMAT_DIV_UP(HIDDEN_CHANNELS, 32);
#pragma unroll
			for (int c = 0; c < coutDiv32; ++c) {
				const int i = lineID + 32 * c;
				if (i < HIDDEN_CHANNELS) {
#pragma unroll
					for (int c = 0; c < HIDDEN_CHANNELS; ++c)
						sWeightsHidden[layer * HIDDEN_CHANNELS * HIDDEN_CHANNELS + c * HIDDEN_CHANNELS + i] =
						volumeInterpolationTensorcoresParameters.cWeightsHidden[layer * HIDDEN_CHANNELS * HIDDEN_CHANNELS + c * HIDDEN_CHANNELS + i];
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
#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT)
					sWeightsLast[i + HIDDEN_CHANNELS * 0] = volumeInterpolationTensorcoresParameters.cWeightsLast[i + HIDDEN_CHANNELS * 0];
					sWeightsLast[i + HIDDEN_CHANNELS * 1] = volumeInterpolationTensorcoresParameters.cWeightsLast[i + HIDDEN_CHANNELS * 1];
#else
					sWeightsLast[i] = volumeInterpolationTensorcoresParameters.cWeightsLast[i];
#endif
				}
			}
			if (lineID == 0) {
#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT)
				sBiasLast[0] = volumeInterpolationTensorcoresParameters.cBiasLast[0];
				sBiasLast[1] = volumeInterpolationTensorcoresParameters.cBiasLast[1];
#else
				sBiasLast = volumeInterpolationTensorcoresParameters.cBiasLast;
#endif
			}
#endif
		}

		__syncthreads();
	}

	template<typename Value_t>
	__device__ __inline__ VolumeGridOutput<Value_t> eval(float3 position, int batch)
	{
		//warp and line ID
		assert(blockDim.x == BLOCK_SIZE);
		const int warpID = threadIdx.x / 32;
		const int lineID = threadIdx.x % 32;

		//storage for the intermediate results
		half* intermediateResults = sIntermediateResults + 32 * HIDDEN_CHANNELS * warpID;

		/////////////////////////////////////
		/// NETWORK
		/////////////////////////////////////

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
			intermediateResults[HIDDEN_CHANNELS * lineID + 3] = 0;
			//fourier features
			for (int i = 0; i < NUM_FOURIER_FEATURES; ++i)
			{
				half c = __hmul(vx, sWeightsFourier[i + NUM_FOURIER_FEATURES * 0]);
				c = __hfma(vy, sWeightsFourier[i + NUM_FOURIER_FEATURES * 1], c);
				c = __hfma(vz, sWeightsFourier[i + NUM_FOURIER_FEATURES * 2], c);
				intermediateResults[HIDDEN_CHANNELS * lineID + 4 + i] = hcos(c);
				intermediateResults[HIDDEN_CHANNELS * lineID + 4 + NUM_FOURIER_FEATURES + i] = hsin(c);
			}
		}
#else
#pragma unroll
		for (int cout = 0; cout < HIDDEN_CHANNELS; ++cout)
		{
			half c = sBiasFirst[cout];
			c = __hfma(__float2half(position.x), sWeightsFirst[cout + HIDDEN_CHANNELS * 0], c);
			c = __hfma(__float2half(position.y), sWeightsFirst[cout + HIDDEN_CHANNELS * 1], c);
			c = __hfma(__float2half(position.z), sWeightsFirst[cout + HIDDEN_CHANNELS * 2], c);
			c = ::kernel::activations::ACTIVATION(c);
			intermediateResults[HIDDEN_CHANNELS * lineID + cout] = c;
		}
#endif

		// hidden layers
		for (int hidden = 0; hidden < NUM_HIDDEN_LAYERS; ++hidden)
		{
			const int Cin16 = HIDDEN_CHANNELS_DIV16;
			static constexpr int Cout16 = HIDDEN_CHANNELS_DIV16;

			//Cout = Cin = Batch=32
			using namespace nvcuda::wmma;
			fragment<matrix_a, 16, 16, 16, half, row_major> a_frag[HIDDEN_CHANNELS_DIV16][HIDDEN_CHANNELS_DIV16]; //row,col
			fragment<matrix_b, 16, 16, 16, half, col_major> b_frag[HIDDEN_CHANNELS_DIV16][2];
			fragment<accumulator, 16, 16, 16, half> c_frag[HIDDEN_CHANNELS_DIV16][2];

			//load C (bias)
#pragma unroll
			for (int cout = 0; cout < Cout16; ++cout)
			{
				load_matrix_sync(c_frag[cout][0], sBiasHidden + HIDDEN_CHANNELS * hidden + 16 * cout, 0, mem_col_major);
				load_matrix_sync(c_frag[cout][1], sBiasHidden + HIDDEN_CHANNELS * hidden + 16 * cout, 0, mem_col_major);
			}

			//load A (weights)
#pragma unroll
			for (int cout = 0; cout < Cout16; ++cout)
#pragma unroll
				for (int cin = 0; cin < Cin16; ++cin)
					load_matrix_sync(a_frag[cout][cin],
						sWeightsHidden + HIDDEN_CHANNELS * HIDDEN_CHANNELS * hidden + 16 * cin + HIDDEN_CHANNELS * 16 * cout,
						HIDDEN_CHANNELS);

			//load B (input)
#pragma unroll
			for (int cin = 0; cin < Cin16; ++cin)
			{
				load_matrix_sync(b_frag[cin][0], intermediateResults + 16 * cin, HIDDEN_CHANNELS);
				load_matrix_sync(b_frag[cin][1], intermediateResults + 16 * cin + 16 * HIDDEN_CHANNELS, HIDDEN_CHANNELS);
			}

			//matmul
#pragma unroll
			for (int i = 0; i < Cout16; ++i) {
#pragma unroll
				for (int j = 0; j < 2; ++j) {
#pragma unroll
					for (int k = 0; k < Cin16; ++k) {
						mma_sync(c_frag[i][j], a_frag[i][k], b_frag[k][j], c_frag[i][j]);
					}
					//activations + bias
					for (int t = 0; t < c_frag[0][0].num_elements; t++)
					{
						c_frag[i][j].x[t] = ::kernel::activations::ACTIVATION(c_frag[i][j].x[t]);
					}
				}
			}

			//copy to shared
#pragma unroll
			for (int cout = 0; cout < Cout16; ++cout)
			{
				store_matrix_sync(intermediateResults + 16 * cout, c_frag[cout][0], HIDDEN_CHANNELS, mem_col_major);
				store_matrix_sync(intermediateResults + 16 * cout + 16 * HIDDEN_CHANNELS, c_frag[cout][1], HIDDEN_CHANNELS, mem_col_major);
			}
		}

		// last layer (output)
		__syncwarp();
#if (OUTPUT_MODE==OUTPUT_MODE_RGBO) || (OUTPUT_MODE==OUTPUT_MODE_RGBO_DIRECT)
	//rgbo [direct]
		half2 out1 = sBiasLast[0];
		half2 out2 = sBiasLast[1];
#pragma unroll
		for (int cin = 0; cin < HIDDEN_CHANNELS; ++cin)
		{
			half v = intermediateResults[HIDDEN_CHANNELS * lineID + cin];
			half2 v2(v, v);
			out1 = __hfma2(v2, sWeightsLast[2 * cin], out1);
			out2 = __hfma2(v2, sWeightsLast[2 * cin + 1], out2);
		}
#if OUTPUT_MODE == OUTPUT_MODE_RGBO
		const real4 value = make_real4(
			::kernel::activations::Sigmoid(__half2float(out1.x)),   //red
			::kernel::activations::Sigmoid(__half2float(out1.y)),   //green
			::kernel::activations::Sigmoid(__half2float(out2.x)),   //blue
			::kernel::activations::Softplus(__half2float(out2.y))); //absorption
#else //OUTPUT_MODE_RGBO_DIRECT
		const real4 value = make_real4(
			clamp(__half2float(out1.x), 0.f, 1.f), //red
			clamp(__half2float(out1.y), 0.f, 1.f), //green
			clamp(__half2float(out2.x), 0.f, 1.f), //blue
			rmax(__half2float(out2.y), 0.f)); //absorption
#endif
#else
		//density [direct]
		half out = sBiasLast;
#pragma unroll
		for (int cin = 0; cin < HIDDEN_CHANNELS; ++cin)
		{
			half v = intermediateResults[HIDDEN_CHANNELS * lineID + cin];
			out = __hfma(v, sWeightsLast[cin], out);
		}
#if OUTPUT_MODE == OUTPUT_MODE_DENSITY
		const real_t value = static_cast<real_t>(
			::kernel::activations::Sigmoid(__half2float(out)));
#else //OUTPUT_MODE_DENSITY_DIRECT
		const real_t value = static_cast<real_t>(
			clamp(__half2float(out), 0.0f, 1.0f));
#endif
#endif

		return VolumeGridOutput<Value_t>{value, true};
	}
};

__global__ void SRNTestKernel()
{
	int idx = threadIdx.x + blockDim.x * blockIdx.x;
	VolumeInterpolationTensorcores srn;
#if OUTPUT_IS_COLOR==1
	auto out = srn.eval<float4>(make_float3(idx, 0, 0), 0);
	printf("[%04d] -> r=%.4f, g=%.4f, b=%.4f, a=%.4f\n",
		idx, out.value.x, out.value.y, out.value.z, out.value.w);
#else
	auto out = srn.eval<float>(make_float3(idx, 0, 0), 0);
	printf("[%04d] -> d=%.4f\n",
		idx, out.value);
#endif
}

std::default_random_engine RND(42);

static void fillRandomHalfMatrix_RowMajor(half* mem, //row-major
	int rows, int cols, bool normalizeRows, bool normalizeCols)
{
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m;
	m.resize(rows, cols);
	std::uniform_real_distribution<float> distr(-1, +1);
	for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
		m(r, c) = distr(RND);
	if (normalizeRows)
		m /= rows;
	if (normalizeCols)
		m /= cols;
	for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
		mem[r + c * rows] = __half2float(m(r, c));
}
static void fillRandomHalfMatrix_ColMajor(half* mem, //row-major
	int rows, int cols, bool normalizeRows, bool normalizeCols)
{
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> m;
	m.resize(rows, cols);
	std::uniform_real_distribution<float> distr(-1, +1);
	for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
		m(r, c) = distr(RND);
	if (normalizeRows)
		m /= rows;
	if (normalizeCols)
		m /= cols;
	for (int r = 0; r < rows; ++r) for (int c = 0; c < cols; ++c)
		mem[c + r * cols] = __half2float(m(r, c));
}

#define MY_INFO(...) INFO(__VA_ARGS__); std::cout << __VA_ARGS__ << std::endl

TEST_CASE("SRN-Kernel", "[modules]")
{
	VolumeInterpolationTensorcoresParameters p;
#if HAS_FOURIER_FEATURES==1
	fillRandomHalfMatrix_ColMajor(p.cWeightsFourier, 3, NUM_FOURIER_FEATURES, false, true);
#else
	fillRandomHalfMatrix_ColMajor(p.cWeightsFirst, 3, HIDDEN_CHANNELS, false, true);
	fillRandomHalfMatrix_ColMajor(p.cBiasFirst, 1, HIDDEN_CHANNELS, false, true);
#endif
	fillRandomHalfMatrix_RowMajor(p.cWeightsHidden, HIDDEN_CHANNELS, HIDDEN_CHANNELS * NUM_HIDDEN_LAYERS, false, true);
	fillRandomHalfMatrix_RowMajor(p.cBiasHidden, HIDDEN_CHANNELS, NUM_HIDDEN_LAYERS, false, false);
	//for (int i = 0; i < HIDDEN_CHANNELS * NUM_HIDDEN_LAYERS; ++i)
	//	p.cBiasHidden[i] = __float2half(i);
#if (OUTPUT_MODE == OUTPUT_MODE_RGBO) || (OUTPUT_MODE == OUTPUT_MODE_RGBO_DIRECT)
	fillRandomHalfMatrix_ColMajor(reinterpret_cast<half*>(p.cWeightsLast), 4, HIDDEN_CHANNELS, false, true);
	fillRandomHalfMatrix_ColMajor(reinterpret_cast<half*>(p.cBiasLast), 4, 1, false, false);
#else
	fillRandomHalfMatrix_ColMajor(p.cWeightsLast, 1, HIDDEN_CHANNELS, false, true);
	fillRandomHalfMatrix_ColMajor(&p.cBiasLast, 1, 1, false, false);
#endif

	CUMAT_SAFE_CALL(cudaMemcpyToSymbol(volumeInterpolationTensorcoresParameters,
		&p, sizeof(VolumeInterpolationTensorcoresParameters)));

	//launch kernel
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	MY_INFO("Launch");

	SRNTestKernel<<<1, BLOCK_SIZE>>>();

	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	MY_INFO("Complete");
}