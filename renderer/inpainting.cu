#include "inpainting.h"

#include <cuMat/src/Context.h>
#include <stack>

#include "errors.h"

#define MAX_CHANNELS 16

///////////////////////////////////////////////////////////////////////
// fast inpainting - discrete version
///////////////////////////////////////////////////////////////////////

namespace
{
	__device__ inline int start_index(int a, int b, int c) {
		return (int)floor((float)(a * c) / b);
	}

	__device__ inline int end_index(int a, int b, int c) {
		return (int)ceil((float)((a + 1) * c) / b);
	}

	template<typename scalar_t>
	__global__ void FastInpaintingKernel_Down(dim3 virtual_size,
		const renderer::Inpainting::MaskTensor mask,
		const renderer::Inpainting::DataTensor data,
		renderer::Inpainting::MaskTensor maskLow,
		renderer::Inpainting::DataTensor  dataLow)
	{
		const int H = mask.rows();
		const int W = mask.cols();
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = data.batches();
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size) //virtual_size: size of low resolution
		{
			int N = 0;
			scalar_t d[MAX_CHANNELS] = { 0 };
			for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
				for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
				{
					if (mask.coeff(ii, jj, 0, -1) >= 0.5)
					{
						N++;
						for (int c = 0; c < C; ++c)
							d[c] += data.coeff(ii, jj, c, -1);
					}
				}
			maskLow.coeff(i, j, 0, -1) = N > 0 ? 1 : 0;
			for (int c = 0; c < C; ++c)
				dataLow.coeff(i, j, c, -1) = N > 0 ? d[c] / N : 0;
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	template<typename scalar_t>
	__global__ void FastInpaintingKernel_Up(dim3 virtual_size,
		const renderer::Inpainting::MaskTensor mask,
		const renderer::Inpainting::DataTensor data,
		const renderer::Inpainting::MaskTensor maskLow,
		const renderer::Inpainting::DataTensor dataLow,
		renderer::Inpainting::MaskTensor maskHigh,
		renderer::Inpainting::DataTensor dataHigh)
	{
		const int H = mask.rows();
		const int W = mask.cols();
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = data.batches();
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size) //virtual_size: size of low resolution
		{
			if (mask.coeff(i, j, 0, -1) >= 0.5)
			{
				//copy unchanged
				maskHigh.coeff(i, j, 0, -1) = 1;
				for (int c = 0; c < C; ++c)
					dataHigh.coeff(i, j, c, -1) = data.coeff(i, j, c, -1);
			}
			else
			{
				//interpolate from low resolution (bilinear)
				//get neighbor offsets
				int io = i % 2 == 0 ? -1 : +1;
				int jo = j % 2 == 0 ? -1 : +1;
				//accumulate
				scalar_t N = 0;
				scalar_t d[MAX_CHANNELS] = { 0 };
#define ITEM(ii,jj,w)													\
	if ((ii)>=0 && (jj)>=0 && (ii)<oH && (jj)<oW && maskLow.coeff((ii), (jj), 0, -1)>=0.5) {	\
		N += w;															\
		for (int c = 0; c < C; ++c) d[c] += w * dataLow.coeff((ii), (jj), c, -1);		\
	}
				ITEM(i / 2, j / 2, 0.75f*0.75f);
				ITEM(i / 2 + io, j / 2, 0.25f*0.75f);
				ITEM(i / 2, j / 2 + jo, 0.25f*0.75f);
				ITEM(i / 2 + io, j / 2 + jo, 0.25f*0.25f);
#undef ITEM
				//write output
				maskHigh.coeff(i, j, 0, -1) = N > 0 ? 1 : 0;
				for (int c = 0; c < C; ++c)
					dataHigh.coeff(i, j, c, -1) = N > 0 ? d[c] / N : 0;
			}
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	std::tuple<renderer::Inpainting::MaskTensor, renderer::Inpainting::DataTensor>
	fastInpaint_recursion(
		const renderer::Inpainting::MaskTensor& mask,
		const renderer::Inpainting::DataTensor& data)
	{
		int64_t C = data.batches();
		int64_t H = data.rows();
		int64_t W = data.cols();

		if (H <= 1 && W <= 1)
			return std::make_tuple(mask, data); //end of recursion

		int64_t oH = H / 2;
		int64_t oW = W / 2;

		//prepare launching
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = ctx.stream();

		//downsample
		renderer::Inpainting::MaskTensor maskLow(oH, oW);
		renderer::Inpainting::DataTensor dataLow(oH, oW, C);
		cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(
			oH, oW, FastInpaintingKernel_Down<float>);
		FastInpaintingKernel_Down<float>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size,
				mask, data, maskLow, dataLow);
		CUMAT_CHECK_ERROR();

		//recursion
		const auto tuple = fastInpaint_recursion(maskLow, dataLow);
		const auto& maskLow2 = std::get<0>(tuple);
		const auto& dataLow2 = std::get<1>(tuple);

		//upsample
		renderer::Inpainting::MaskTensor maskHigh(H, W);
		renderer::Inpainting::DataTensor dataHigh(H, W, C);
		cfg = ctx.createLaunchConfig2D(
			H, W, FastInpaintingKernel_Up<float>);
		FastInpaintingKernel_Up<float>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size,
				mask, data,
				maskLow2, dataLow2,
				maskHigh, dataHigh);
		CUMAT_CHECK_ERROR();

		//done
		return std::make_tuple(maskHigh, dataHigh);
	}

}

renderer::Inpainting::DataTensor renderer::Inpainting::fastInpaint(
	const renderer::Inpainting::MaskTensor& mask,
	const renderer::Inpainting::DataTensor& data)
{
	//check input
	int64_t H = mask.rows();
	int64_t W = mask.cols();
	int64_t C = data.batches();
	CHECK_ERROR(data.rows() == H, "expected the data to have the same number of rows as the mask. Expected ", mask.rows(), ", but got ", data.rows());
	CHECK_ERROR(data.cols() == W, "expected the data to have the same number of columns as the mask. Expected ", mask.cols(), ", but got ", data.cols());
	CHECK_ERROR(C < 16, "Inpainting::fastInpaint only supports up to 16 channels, but got ", C);

	//inpaint recursivly
	return std::get<1>(fastInpaint_recursion(mask, data));
}


///////////////////////////////////////////////////////////////////////
// fast inpainting - fractional version
///////////////////////////////////////////////////////////////////////

namespace
{

	template<typename scalar_t>
	__global__ void FastInpaintingFractionalKernel_Down(dim3 virtual_size,
		const renderer::Inpainting::MaskTensor mask,
		const renderer::Inpainting::DataTensor data,
		renderer::Inpainting::MaskTensor maskLow,
		renderer::Inpainting::DataTensor dataLow)
	{
		const int H = mask.rows();
		const int W = mask.cols();
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = data.batches();
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size) //virtual_size: size of low resolution
		{
			int Count = 0;
			scalar_t N1 = 0;
			scalar_t N2 = 0;
			scalar_t d[MAX_CHANNELS] = { 0 };
			for (int ii = start_index(i, oH, H); ii < end_index(i, oH, H); ++ii)
				for (int jj = start_index(j, oW, W); jj < end_index(j, oW, W); ++jj)
				{
					Count++;
					float m = mask.coeff(ii, jj, 0, -1);
					N1 += m;
					N2 = max(N2, m);
					for (int c = 0; c < C; ++c)
						d[c] += m * data.coeff(ii, jj, c, -1);
				}
			//maskLow[b][i][j] = N1 / Count;
			maskLow.coeff(i, j, 0, -1) = N2;
			for (int c = 0; c < C; ++c)
				dataLow.coeff(i, j, c, -1) = N1 > 0 ? d[c] / N1 : 0;
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	template<typename scalar_t>
	__global__ void FastInpaintingFractionalKernel_Up(dim3 virtual_size,
		const renderer::Inpainting::MaskTensor mask,
		const renderer::Inpainting::DataTensor data,
		const renderer::Inpainting::MaskTensor maskLow,
		const renderer::Inpainting::DataTensor dataLow,
		renderer::Inpainting::MaskTensor maskHigh,
		renderer::Inpainting::DataTensor dataHigh)
	{
		const int H = mask.rows();
		const int W = mask.cols();
		const int oH = H / 2;
		const int oW = W / 2;
		const int C = data.batches();
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size) //virtual_size: size of high resolution
		{
			//interpolate from low resolution (bilinear)
			//get neighbor offsets
			int io = i % 2 == 0 ? -1 : +1;
			int jo = j % 2 == 0 ? -1 : +1;
			//accumulates
			scalar_t Weight = 0;
			scalar_t N = 0;
			scalar_t d[MAX_CHANNELS] = { 0 };
#define ITEM(ii,jj,w)														\
	if ((ii)>=0 && (jj)>=0 && (ii)<oH && (jj)<oW) {								\
		Weight += w;															\
		N += w * maskLow.coeff((ii), (jj), 0, -1);								\
		for (int c = 0; c < C; ++c)												\
			d[c] += w * maskLow.coeff((ii), (jj), 0, -1) * dataLow.coeff((ii), (jj), c, -1);	\
	}
			ITEM(i / 2, j / 2, 0.75f*0.75f);
			ITEM(i / 2 + io, j / 2, 0.25f*0.75f);
			ITEM(i / 2, j / 2 + jo, 0.25f*0.75f);
			ITEM(i / 2 + io, j / 2 + jo, 0.25f*0.25f);
#undef ITEM
			//write output
			scalar_t m = mask.coeff(i, j,  0, -1);
			maskHigh.coeff(i, j, 0, -1) = m + (N > 0 ? (1 - m) * (N / Weight) : 0);
			for (int c = 0; c < C; ++c)
			{
				dataHigh.coeff(i, j, c, -1) =
					m * data.coeff(i, j, c, -1) +
					(1 - m) * (N > 0 ? d[c] / N : 0);
			}
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	std::tuple<renderer::Inpainting::MaskTensor, renderer::Inpainting::DataTensor>
		fastInpaintFractional_recursion(
			const renderer::Inpainting::MaskTensor& mask,
			const renderer::Inpainting::DataTensor& data)
	{
		int64_t C = data.batches();
		int64_t H = data.rows();
		int64_t W = data.cols();

		//std::cout << "fastInpaintFractional_recursion - Pre:"
		//	<< " shape=(" << H << ", " << W << ")"
		//	<< ", data min=" << torch::min(data).item().toFloat()
		//	<< ", max=" << torch::max(data).item().toFloat()
		//	<< ", avg=" << torch::mean(data).item().toFloat()
		//	<< std::endl;

		if (H <= 1 || W <= 1)
			return std::make_tuple(mask, data); //end of recursion

		int64_t oH = H / 2;
		int64_t oW = W / 2;

		//prepare launching
		cuMat::Context& ctx = cuMat::Context::current();
		cudaStream_t stream = ctx.stream();

		//downsample
		renderer::Inpainting::MaskTensor maskLowPre(oH, oW);
		renderer::Inpainting::DataTensor dataLowPre(oH, oW, C);
			cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(
			oH, oW, FastInpaintingFractionalKernel_Down<float>);
		FastInpaintingFractionalKernel_Down<float>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size,
				mask, data, maskLowPre, dataLowPre);
		CUMAT_CHECK_ERROR();

		//recursion
		const auto tuple = fastInpaintFractional_recursion(maskLowPre, dataLowPre);
		const auto& maskLowPost = std::get<0>(tuple);
		const auto& dataLowPost = std::get<1>(tuple);

		//upsample
		renderer::Inpainting::MaskTensor maskHigh(H, W);
		renderer::Inpainting::DataTensor dataHigh(H, W, C);
		cfg = ctx.createLaunchConfig2D(
			H, W, FastInpaintingFractionalKernel_Up<float>);
		FastInpaintingFractionalKernel_Up<float>
			<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
			(cfg.virtual_size,
				mask, data,
				maskLowPost, dataLowPost,
				maskHigh, dataHigh);
		CUMAT_CHECK_ERROR();

		//done
		return std::make_tuple(maskHigh, dataHigh);
	}

}

renderer::Inpainting::DataTensor renderer::Inpainting::fastInpaintFractional(
	const renderer::Inpainting::MaskTensor& mask,
	const renderer::Inpainting::DataTensor& data)
{
	//check input
	int64_t H = mask.rows();
	int64_t W = mask.cols();
	int64_t C = data.batches();
	CHECK_ERROR(data.rows() == H, "expected the data to have the same number of rows as the mask. Expected ", mask.rows(), ", but got ", data.rows());
	CHECK_ERROR(data.cols() == W, "expected the data to have the same number of columns as the mask. Expected ", mask.cols(), ", but got ", data.cols());
	CHECK_ERROR(C < 16, "Inpainting::fastInpaint only supports up to 16 channels, but got ", C);

	//inpaint recursivly
	return std::get<1>(fastInpaintFractional_recursion(mask, data));
}
