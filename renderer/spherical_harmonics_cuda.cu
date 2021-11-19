//#include "spherical_harmonics.h" //no include

#include <cuda.h>
#include <stdexcept>
#include <cuMat/src/Context.h>

#include "renderer_tensor.cuh"
#include "renderer_spherical_harmonics.cuh"
#include "renderer_utils.cuh"
#include "helper_math.cuh"

namespace {
	template<typename T, int degree, bool normalize>
	__global__ void SphericalHarmonicsKernel(
		dim3 virtual_size, kernel::Tensor3Read<T> in, kernel::Tensor3RW<T> out)
	{
		CUMAT_KERNEL_2D_LOOP(a, b, virtual_size)
		{
			T x = in[a][0][b];
			T y = in[a][1][b];
			T z = in[a][2][b];
			if constexpr (normalize)
			{
				T invLen = rsqrtf(x * x + y * y + z * z);
				x *= invLen;
				y *= invLen;
				z *= invLen;
			}

			out[a][0][b] = kernel::SphericalHarmonicsCoeff<T, 0, 0>::eval(x, y, z);
			if constexpr (degree >= 1)
			{
				out[a][1][b] = kernel::SphericalHarmonicsCoeff<T, 1, -1>::eval(x, y, z);
				out[a][2][b] = kernel::SphericalHarmonicsCoeff<T, 1, 0 >::eval(x, y, z);
				out[a][3][b] = kernel::SphericalHarmonicsCoeff<T, 1, +1>::eval(x, y, z);
			}
			if constexpr (degree >= 2)
			{
				out[a][4][b] = kernel::SphericalHarmonicsCoeff<T, 2, -2>::eval(x, y, z);
				out[a][5][b] = kernel::SphericalHarmonicsCoeff<T, 2, -1>::eval(x, y, z);
				out[a][6][b] = kernel::SphericalHarmonicsCoeff<T, 2, 0 >::eval(x, y, z);
				out[a][7][b] = kernel::SphericalHarmonicsCoeff<T, 2, +1>::eval(x, y, z);
				out[a][8][b] = kernel::SphericalHarmonicsCoeff<T, 2, +2>::eval(x, y, z);
			}
			if constexpr (degree >= 3)
			{
				out[a][ 9][b] = kernel::SphericalHarmonicsCoeff<T, 3, -3>::eval(x, y, z);
				out[a][10][b] = kernel::SphericalHarmonicsCoeff<T, 3, -2>::eval(x, y, z);
				out[a][11][b] = kernel::SphericalHarmonicsCoeff<T, 3, -1>::eval(x, y, z);
				out[a][12][b] = kernel::SphericalHarmonicsCoeff<T, 3, 0 >::eval(x, y, z);
				out[a][13][b] = kernel::SphericalHarmonicsCoeff<T, 3, +1>::eval(x, y, z);
				out[a][14][b] = kernel::SphericalHarmonicsCoeff<T, 3, +2>::eval(x, y, z);
				out[a][15][b] = kernel::SphericalHarmonicsCoeff<T, 3, +3>::eval(x, y, z);
			}
			if constexpr (degree >= 4)
			{
				out[a][16][b] = kernel::SphericalHarmonicsCoeff<T, 4, -4>::eval(x, y, z);
				out[a][17][b] = kernel::SphericalHarmonicsCoeff<T, 4, -3>::eval(x, y, z);
				out[a][18][b] = kernel::SphericalHarmonicsCoeff<T, 4, -2>::eval(x, y, z);
				out[a][19][b] = kernel::SphericalHarmonicsCoeff<T, 4, -1>::eval(x, y, z);
				out[a][20][b] = kernel::SphericalHarmonicsCoeff<T, 4, 0 >::eval(x, y, z);
				out[a][21][b] = kernel::SphericalHarmonicsCoeff<T, 4, +1>::eval(x, y, z);
				out[a][22][b] = kernel::SphericalHarmonicsCoeff<T, 4, +2>::eval(x, y, z);
				out[a][23][b] = kernel::SphericalHarmonicsCoeff<T, 4, +3>::eval(x, y, z);
				out[a][24][b] = kernel::SphericalHarmonicsCoeff<T, 4, +4>::eval(x, y, z);
			}
		}
		CUMAT_KERNEL_2D_LOOP_END
	}
}

template<typename T, int degree, bool normalize>
static void SphericalHarmonicsImpl3(
	const kernel::Tensor3Read<T>& in, kernel::Tensor3RW<T>& out, CUstream stream)
{
	cuMat::Context& ctx = cuMat::Context::current();
	cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(
		in.size(0), in.size(2), SphericalHarmonicsKernel<T, degree, normalize>);
	SphericalHarmonicsKernel<T, degree, normalize>
		<<<cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, in, out);
	CUMAT_CHECK_ERROR();
}

template<typename T>
static void SphericalHarmonicsImpl2(
	const kernel::Tensor3Read<T>& in, kernel::Tensor3RW<T>& out, 
	int degree, bool normalize, CUstream stream)
{
#define CASE(d)															\
	case d:																\
		{																\
		if (normalize)													\
			SphericalHarmonicsImpl3<T, d, true>(in, out, stream);		\
		else															\
			SphericalHarmonicsImpl3<T, d, false>(in, out, stream);		\
		} break
	
	switch (degree)
	{
		CASE(0);
		CASE(1);
		CASE(2);
		CASE(3);
		CASE(4);
	default:
		throw std::runtime_error("Only degree 0 to 4 supported");
	}
}

namespace kernel {

	void SphericalHarmonicsImpl(const Tensor3Read<float>& in, Tensor3RW<float>& out, int degree, bool normalize, CUstream stream)
	{
		SphericalHarmonicsImpl2(in, out, degree, normalize, stream);
	}

	void SphericalHarmonicsImpl(const Tensor3Read<double>& in, Tensor3RW<double>& out, int degree, bool normalize, CUstream stream)
	{
		SphericalHarmonicsImpl2(in, out, degree, normalize, stream);
	}

}
