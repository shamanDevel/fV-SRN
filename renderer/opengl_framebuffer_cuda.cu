#if RENDERER_OPENGL_SUPPORT==1

#include <cuda.h>
#include <cuMat/src/Context.h>

#include "renderer_tensor.cuh"
#include "helper_math.cuh"
#include "renderer_utils.cuh"

texture<float4, cudaTextureType2D, cudaReadModeElementType> framebufferTexRef0;
texture<float4, cudaTextureType2D, cudaReadModeElementType> framebufferTexRef1;

namespace kernel
{
	template<typename T>
	__global__ void FramebufferCopyToCuda(dim3 virtual_size,
		kernel::Tensor4RW<T> output)
	{
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size)
		{
			float4 rgba = tex2D(framebufferTexRef0, i, j);
			float depth = tex2D(framebufferTexRef1, i, j).x;
			//if (rgba.w > 0) printf("{%04d, %04d} d=%f\n", int(i), int(j), depth);
			output[0][0][j][i] = rgba.x;
			output[0][1][j][i] = rgba.y;
			output[0][2][j][i] = rgba.z;
			output[0][3][j][i] = rgba.w;
			output[0][4][j][i] = depth;
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	template<typename T>
	void CopyFramebufferToCudaImpl(
		cudaGraphicsResource_t colorTexture,
		cudaGraphicsResource_t depthTexture,
		kernel::Tensor4RW<T> output,
		cudaStream_t stream)
	{
		CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &colorTexture, stream));
		cudaArray_t array0;
		CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array0, colorTexture, 0, 0));
		CUMAT_SAFE_CALL(cudaBindTextureToArray(framebufferTexRef0, array0));

		CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &depthTexture, stream));
		cudaArray_t array1;
		CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array1, depthTexture, 0, 0));
		CUMAT_SAFE_CALL(cudaBindTextureToArray(framebufferTexRef1, array1));

		int width = output.size(3);
		int height = output.size(2);
		cuMat::Context& ctx = cuMat::Context::current();
		const auto cfg = ctx.createLaunchConfig2D(width, height, FramebufferCopyToCuda<T>);
		FramebufferCopyToCuda<T>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size, output);
		CUMAT_CHECK_ERROR();

		CUMAT_SAFE_CALL(cudaUnbindTexture(framebufferTexRef1));
		CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &depthTexture, stream));

		CUMAT_SAFE_CALL(cudaUnbindTexture(framebufferTexRef0));
		CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &colorTexture, stream));
	}

	void CopyFramebufferToCuda(
		cudaGraphicsResource_t colorTexture,
		cudaGraphicsResource_t depthTexture,
		kernel::Tensor4RW<float> output,
		cudaStream_t stream)
	{
		CopyFramebufferToCudaImpl<float>(colorTexture, depthTexture, output, stream);
	}

	void CopyFramebufferToCuda(
		cudaGraphicsResource_t colorTexture,
		cudaGraphicsResource_t depthTexture,
		kernel::Tensor4RW<double> output,
		cudaStream_t stream)
	{
		CopyFramebufferToCudaImpl<double>(colorTexture, depthTexture, output, stream);
	}
}

#endif
