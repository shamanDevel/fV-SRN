#include <cuda.h>
#include <cuMat/src/Context.h>

#include "helper_math.cuh"

namespace renderer {
	namespace detail {

		__global__ void Compute1DPreintegrationTableKernel(
			cudaTextureObject_t srcTFTexture, cudaSurfaceObject_t table,
			int resolution)
		{
			if (blockIdx.x==0 && threadIdx.x==0)
			{
			    //I'm lazy, no paralleliation
				float4 integral = make_float4(0);
				float lastDensity = 0.0f;
				float4 lastValue = tex1D<float4>(srcTFTexture, lastDensity);
				for (int i = 0; i < resolution; ++i)
				{
					float currentDensity = (float(i) + 0.5f) / float(resolution);
					//rectangle rule
					float4 currentValue = tex1D<float4>(srcTFTexture, currentDensity);
					float3 currentRGB = make_float3(currentValue);
					float3 lastRGB = make_float3(lastValue);
					integral += (currentDensity - lastDensity) * make_float4(
						0.5f * (lastRGB * lastValue.w + currentRGB * currentValue.w),
						0.5f * (lastValue.w + currentValue.w)
					);
					surf1Dwrite(integral, table, (int)sizeof(float4) * i, cudaBoundaryModeTrap);
					lastValue = currentValue;
					lastDensity = currentDensity;
				}
			}
		}

		void Compute1DPreintegrationTable(cudaTextureObject_t srcTFTexture, cudaSurfaceObject_t dstSurface,
			int dstResolution, CUstream stream)
		{
			Compute1DPreintegrationTableKernel
				<<< 1, 32, 0, stream >>>
				(srcTFTexture, dstSurface, dstResolution);
			CUMAT_CHECK_ERROR();
		}

		__global__ void Compute2DPreintegrationTableKernel(
			dim3 virtualSize,
			cudaTextureObject_t srcTFTexture, cudaSurfaceObject_t table,
			float stepsize, int resolution, int N)
		{
			CUMAT_KERNEL_2D_LOOP(istart, iend, virtualSize)
			{
				float dstart = (static_cast<float>(istart) + 0.5f) / static_cast<float>(resolution);
				float dend = (static_cast<float>(iend) + 0.5f) / static_cast<float>(resolution);
				//Riemann-Sum integration
				float3 rgb_sum = make_float3(0, 0, 0);
				float alpha_sum = 0;
				float h = 1.0f / static_cast<float>(N);
				for (int i = 1; i <= N; ++i)
				{
					float omega = i * h;
					float dcurrent = (1 - omega) * dstart + omega * dend;
					float4 value = tex1D<float4>(srcTFTexture, dcurrent);
					float3 rgbCurrent = make_float3(value);
					float alphaCurrent = value.w;

					alpha_sum += alphaCurrent * h * stepsize;
					rgb_sum += h * (rgbCurrent * alphaCurrent * stepsize * expf(-alpha_sum));
				}
				float final_alpha = 1 - expf(-alpha_sum);
				//printf("d=[%.3f, %.3f] -> rgb=(%.3f, %.3f, %.3f), alpha=%.3f\n",
				//	dstart, dend, rgb_sum.x, rgb_sum.y, rgb_sum.z, final_alpha);
				float4 rgba = make_float4(rgb_sum, final_alpha);
				surf2Dwrite(rgba, table, (int)sizeof(float4) * istart, iend, cudaBoundaryModeTrap);
			}
			CUMAT_KERNEL_2D_LOOP_END
		}

		void Compute2DPreintegrationTable(cudaTextureObject_t srcTFTexture, cudaSurfaceObject_t dstSurface,
			int dstResolution, float stepsize, int quadratureSteps, CUstream stream)
		{
			cuMat::Context& ctx = cuMat::Context::current();
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(
				dstResolution, dstResolution, Compute2DPreintegrationTableKernel);
			Compute2DPreintegrationTableKernel
				<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
				(cfg.virtual_size, srcTFTexture, dstSurface, stepsize, dstResolution, quadratureSteps);
			CUMAT_CHECK_ERROR();
		}
	}
}