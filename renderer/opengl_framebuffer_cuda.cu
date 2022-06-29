#if RENDERER_OPENGL_SUPPORT==1

#include <cuda.h>
#include <cuMat/src/Context.h>

#include "renderer_tensor.cuh"
#include "helper_math.cuh"
#include "renderer_utils.cuh"

namespace
{
    class OpenGL2TexRef
    {
		cudaGraphicsResource_t tex_;
		cudaStream_t stream_;
		cudaArray_t array_;
		cudaTextureObject_t ref_;

    public:
		OpenGL2TexRef(cudaGraphicsResource_t tex, cudaStream_t stream)
		    : tex_(tex), stream_(stream), array_(nullptr), ref_(0)
		{
			CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &tex, stream));
			CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array_, tex, 0, 0));
			
			cudaResourceDesc            texRes;
			memset(&texRes, 0, sizeof(cudaResourceDesc));

			texRes.resType = cudaResourceTypeArray;
			texRes.res.array.array = array_;

			cudaTextureDesc             texDescr;
			memset(&texDescr, 0, sizeof(cudaTextureDesc));

			texDescr.normalizedCoords = false;
			texDescr.filterMode = cudaFilterModePoint;
			texDescr.addressMode[0] = cudaAddressModeClamp;
			texDescr.addressMode[1] = cudaAddressModeClamp;
			texDescr.readMode = cudaReadModeElementType;

			CUMAT_SAFE_CALL(cudaCreateTextureObject(&ref_, &texRes, &texDescr, nullptr));
		}

		~OpenGL2TexRef()
		{
			cudaDestroyTextureObject(ref_);
			CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &tex_, stream_));
		}

		cudaTextureObject_t get() const
		{
			return ref_;
		}
    };
}

namespace kernel
{
	template<typename T>
	__global__ void FramebufferCopyToCuda(dim3 virtual_size,
		kernel::Tensor4RW<T> output,
		cudaTextureObject_t framebufferTexRef0, cudaTextureObject_t framebufferTexRef1)
	{
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size)
		{
			float4 rgba = tex2D<float4>(framebufferTexRef0, i, j);
			float depth = tex2D<float4>(framebufferTexRef1, i, j).x;
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
		OpenGL2TexRef colorRef(colorTexture, stream);
		OpenGL2TexRef depthRef(depthTexture, stream);

		int width = output.size(3);
		int height = output.size(2);
		cuMat::Context& ctx = cuMat::Context::current();
		const auto cfg = ctx.createLaunchConfig2D(width, height, FramebufferCopyToCuda<T>);
		FramebufferCopyToCuda<T>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size, output, colorRef.get(), depthRef.get());
		CUMAT_CHECK_ERROR();
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
