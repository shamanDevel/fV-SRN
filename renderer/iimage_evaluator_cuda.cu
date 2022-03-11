#include <cuda.h>
#include <GL/glew.h>
#include <cuMat/src/Context.h>

#include "renderer_tensor.cuh"
#include "helper_math.cuh"
#include "renderer_utils.cuh"

namespace kernel
{
	template<typename T>
	__device__ inline float fetchChannel(
		const Tensor4Read<T>& input,
		int channel, int x, int y)
	{
		if (channel == -1) return 0;
		if (channel == -2) return 1;
		return input[0][channel][y][x];
	}
	template<typename T>
	__global__ void SelectOutputChannelKernel(
		dim3 virtual_size,
		Tensor4Read<T> input,
		unsigned int* output,
		int rId, int gId, int bId, int aId,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA)
	{
		CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)
		{
			float r = fetchChannel(input, rId, x, y) * scaleRGB + offsetRGB;
			float g = fetchChannel(input, gId, x, y) * scaleRGB + offsetRGB;
			float b = fetchChannel(input, bId, x, y) * scaleRGB + offsetRGB;
			float a = fetchChannel(input, aId, x, y) * scaleA + offsetA;
			//printf("%d, %d, %d\n", int(y), input.size(3), int(x));
			output[y * input.size(3) + x] = rgbaToInt(r, g, b, a);
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	template<typename T>
	void CopyOutputToTextureImpl(
		int width, int height,
		const Tensor4Read<T>& output,
		GLubyte* texture,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, SelectOutputChannelKernel<T>);
		SelectOutputChannelKernel<T>
			<<<cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size, output, reinterpret_cast<unsigned*>(texture),
				r, g, b, a, scaleRGB, offsetRGB, scaleA, offsetA);
		CUMAT_CHECK_ERROR();
	}

	void CopyOutputToTexture(
		int width, int height,
		const Tensor4Read<float>& input,
		GLubyte* output,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream)
	{
		CopyOutputToTextureImpl<float>(width, height, input, output,
			r, g, b, a, scaleRGB, offsetRGB, scaleA, offsetA, stream);
	}
	void CopyOutputToTexture(
		int width, int height,
		const Tensor4Read<double>& input,
		GLubyte* output,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream)
	{
		CopyOutputToTextureImpl<double>(width, height, input, output,
			r, g, b, a, scaleRGB, offsetRGB, scaleA, offsetA, stream);
	}

	template<typename T>
	__global__ void SelectOutputChannelKernel2(
		dim3 virtual_size,
		Tensor4Read<T> input,
		Tensor4RW<T> output,
		int rId, int gId, int bId, int aId,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA)
	{
		CUMAT_KERNEL_3D_LOOP(x, y, batch, virtual_size)
		{
			float r = fetchChannel(input, rId, x, y) * scaleRGB + offsetRGB;
			float g = fetchChannel(input, gId, x, y) * scaleRGB + offsetRGB;
			float b = fetchChannel(input, bId, x, y) * scaleRGB + offsetRGB;
			float a = fetchChannel(input, aId, x, y) * scaleA + offsetA;
			//printf("%d, %d, %d\n", int(y), input.size(3), int(x));
			output[batch][0][y][x] = r;
			output[batch][1][y][x] = g;
			output[batch][2][y][x] = b;
			output[batch][3][y][x] = a;
		}
		CUMAT_KERNEL_2D_LOOP_END
	}
	template<typename T>
	void CopyOutputToTextureImpl2(
		int width, int height, int batches,
		const Tensor4Read<T>& input,
		Tensor4RW<T>& output,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(width, height, batches, SelectOutputChannelKernel<T>);
		SelectOutputChannelKernel2<T>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size, input, output,
				r, g, b, a, scaleRGB, offsetRGB, scaleA, offsetA);
		CUMAT_CHECK_ERROR();
	}

	void CopyOutputToTexture(
		int width, int height, int batches,
		const Tensor4Read<float>& input,
		Tensor4RW<float>& output,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream)
	{
		CopyOutputToTextureImpl2<float>(width, height, batches, input, output,
			r, g, b, a, scaleRGB, offsetRGB, scaleA, offsetA, stream);
	}
	void CopyOutputToTexture(
		int width, int height, int batches,
		const Tensor4Read<double>& input,
		Tensor4RW<double>& output,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream)
	{
		CopyOutputToTextureImpl2<double>(width, height, batches, input, output,
			r, g, b, a, scaleRGB, offsetRGB, scaleA, offsetA, stream);
	}

	template<typename T, typename V3 = typename scalar_traits<T>::real3>
	__device__ V3 tonemappingFunction(V3 rgb, float maxExposure)
	{
		//constants
		const T A = 2.51f;
		const T B = 0.03f;
		const T C = 2.43f;
		const T D = 0.59f;
		const T E = 0.14f;
		const T div_gamma = 1.0f / 2.4f;

		//divide by exposure
		rgb /= static_cast<T>(maxExposure);

		//ACES filmic curve
		rgb = (rgb * (A * rgb + B)) / (rgb * (C * rgb + D) + E);
		rgb = clamp(rgb, T(0), T(1));

		//gamma
		rgb = rpow(rgb, V3{ div_gamma, div_gamma, div_gamma });
		return rgb;
	}
	
	template<typename T>
	__global__ void TonemappingKernel(
		dim3 virtual_size,
		Tensor4Read<T> input,
		unsigned int* output,
		float maxExposure)
	{
		using V3 = typename scalar_traits<T>::real3;
		CUMAT_KERNEL_2D_LOOP(x, y, virtual_size)
		{
			T r = input[0][0][y][x];
			T g = input[0][1][y][x];
			T b = input[0][2][y][x];
			T a = input[0][3][y][x];
			V3 rgb{ r,g,b };

			rgb = tonemappingFunction<T>(rgb, maxExposure);
			
			output[y * input.size(3) + x] = rgbaToInt(rgb.x, rgb.y, rgb.z, a);
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	template<typename T>
	void TonemappingImpl(
		int width, int height,
		const Tensor4Read<T>& output,
		GLubyte* texture,
		float maxExposure,
		CUstream stream)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(width, height, TonemappingKernel<T>);
		TonemappingKernel<T>
			<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
			(cfg.virtual_size, output, reinterpret_cast<unsigned*>(texture),
				maxExposure);
		CUMAT_CHECK_ERROR();
	}

	void Tonemapping(
		int width, int height,
		const Tensor4Read<float>& output,
		GLubyte* texture,
		float maxExposure,
		CUstream stream)
	{
		TonemappingImpl<float>(width, height, output, texture,
			maxExposure, stream);
	}
	void Tonemapping(
		int width, int height,
		const Tensor4Read<double>& output,
		GLubyte* texture,
		float maxExposure,
		CUstream stream)
	{
		TonemappingImpl<double>(width, height, output, texture,
			maxExposure, stream);
	}


	template<typename T>
	__global__ void TonemappingKernel2(
		dim3 virtual_size,
		Tensor4Read<T> input,
		Tensor4RW<T> output,
		float maxExposure)
	{
		using V3 = typename scalar_traits<T>::real3;
		CUMAT_KERNEL_3D_LOOP(x, y, batch, virtual_size)
		{
			T r = input[batch][0][y][x];
			T g = input[batch][1][y][x];
			T b = input[batch][2][y][x];
			T a = input[batch][3][y][x];
			V3 rgb{ r,g,b };

			rgb = tonemappingFunction<T>(rgb, maxExposure);

			output[batch][0][y][x] = rgb.x;
			output[batch][1][y][x] = rgb.y;
			output[batch][2][y][x] = rgb.z;
			output[batch][3][y][x] = a;
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	template<typename T>
	void TonemappingImpl2(
		int width, int height, int batches,
		const Tensor4Read<T>& input,
		Tensor4RW<T>& output,
		float maxExposure,
		CUstream stream)
	{
		cuMat::Context& ctx = cuMat::Context::current();
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig3D(width, height, batches, TonemappingKernel2<T>);
		TonemappingKernel2<T>
			<< <cfg.block_count, cfg.thread_per_block, 0, stream >> >
			(cfg.virtual_size, input, output, maxExposure);
		CUMAT_CHECK_ERROR();
	}

	void Tonemapping(
		int width, int height, int batches,
		const Tensor4Read<float>& input,
		Tensor4RW<float>& output,
		float maxExposure,
		CUstream stream)
	{
		TonemappingImpl2<float>(width, height, batches, input, output, maxExposure, stream);
	}
	void Tonemapping(
		int width, int height, int batches,
		const Tensor4Read<double>& input,
		Tensor4RW<double>& output,
		float maxExposure,
		CUstream stream)
	{
		TonemappingImpl2<double>(width, height, batches, input, output, maxExposure, stream);
	}
}
