#include "warping.h"
#include "helper_math.cuh"

namespace
{
	__device__ float access(
		const renderer::Warping::DataTensor& m, int x, int y, int c)
	{
		if (x < 0 || y < 0 || x >= m.cols() || y >= m.rows())
			return 0.0f;
		else
			return m.coeff(y, x, c, -1);
	}

	__device__ float interpolate(
		const renderer::Warping::DataTensor& m, float x, float y, int c)
	{
		int ix = int(x);
		int iy = int(y);
		float fx = x - ix;
		float fy = y - iy;
		return lerp(
			lerp(access(m, ix, iy, c), access(m, ix + 1, iy, c), fx),
			lerp(access(m, ix, iy + 1, c), access(m, ix + 1, iy + 1, c), fx),
			fy);
	}

	__global__ void WarpKernel(dim3 virtual_size,
		const renderer::Warping::DataTensor input,
		const renderer::Warping::FlowTensor flow,
		renderer::Warping::DataTensor output)
	{
		const int H = input.rows();
		const int W = input.cols();
		const int C = input.batches();
		CUMAT_KERNEL_2D_LOOP(y, x, virtual_size)
		{
			float flowX = flow.coeff(y, x, 0, -1);
			float flowY = flow.coeff(y, x, 1, -1);
			float sourceX = x - W * flowX;
			float sourceY = y - H * flowY;
			for (int c = 0; c < C; ++c)
				output.coeff(y, x, c, -1) = interpolate(input, sourceX, sourceY, c);
		}
		CUMAT_KERNEL_2D_LOOP_END
	}
}

renderer::Warping::DataTensor renderer::Warping::warp(const DataTensor& data, const FlowTensor& flow)
{
	int64_t H = data.rows();
	int64_t W = data.cols();
	int64_t C = data.batches();
	DataTensor output(H, W, C);

	cuMat::Context& ctx = cuMat::Context::current();
	cudaStream_t stream = ctx.stream();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(
		H, W, WarpKernel);
	WarpKernel <<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, data, flow, output);
	CUMAT_CHECK_ERROR();

	return output;
}
