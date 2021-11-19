#include "spherical_harmonics.h"

#include <magic_enum.hpp>
#include <torch/csrc/cuda/Stream.h>
#include <cuMat/src/Context.h>

#include "pytorch_utils.h"
#include "module_registry.h"

#include "renderer_spherical_harmonics.cuh"

void renderer::SphericalHarmonics::registerPybindModules(pybind11::module& m)
{
	namespace py = pybind11;
	py::class_<SphericalHarmonics>(m, "SphericalHarmonics")
		.def_static("max_degree", &SphericalHarmonics::MaxDegree,
			py::doc("Returns the maximal degree that is supported."))
		.def_static("get_coefficient_count", &SphericalHarmonics::GetCoefficientCount,
			py::doc(R"(
	Get the total number of coefficients for a function represented by
    all spherical harmonic basis of degree <= \c degree.
    Note: 0 <= degree <= MaxDegree()
		)"), py::arg("degree"))
		.def_static("get_index", &SphericalHarmonics::GetIndex,
			py::doc(R"(
	Get the one-dimensional index associated with a particular degree \c l
	and order \c m. This is the index that can be used to access
	the coefficients returned by \ref evaluate.
		)"), py::arg("l"), py::arg("m"))
		.def_static("evaluate", &SphericalHarmonics::Evaluate,
			py::doc(R"(
	Evaluates the spherical harmonics up to degree <= \c degree.
	The given tensor must have a size of 3 at the given dimension \c dim,
	all other dimensions are treated as batch dimension.
	
	See \ref get_index for the mapping from degree and order
	to output index
	
	:param: t the tensor of size '3' in dimension 'dim'
	:param: dim the dimension containing the coordinates
	:param: degree evaluates the spherical harmonics up to degree <= \c degree
	:param: normalize false=treat the input vectors as unit, true=normalize them to unit vectors
	:return: a tensor of shape [...,N,...] with \c N=GetCoefficientCount(degree) entries in dimension \c dim
			)"), py::arg("t"), py::arg("dim"), py::arg("degree"),
			py::arg("normalize") = false);
}

int renderer::SphericalHarmonics::MaxDegree()
{
	return kernel::SHCoeffDegreeLimit;
}

int renderer::SphericalHarmonics::GetCoefficientCount(int degree)
{
	TORCH_CHECK(degree >= 0, "SH degree must be non-negative");
	TORCH_CHECK(degree <= MaxDegree(), "only ", MaxDegree(), " degrees supported");
	return kernel::SHGetCoefficientCount(degree);
}

int renderer::SphericalHarmonics::GetIndex(int l, int m)
{
	TORCH_CHECK(l >= 0, "SH degree must be non-negative");
	TORCH_CHECK(l <= MaxDegree(), "only ", MaxDegree(), " degrees supported");
	TORCH_CHECK((-l <= m) && (m <= l), "-l <= m <= +l violated");
	return kernel::SHGetIndex(l, m);
}

torch::Tensor renderer::SphericalHarmonics::Evaluate(const torch::Tensor& t, int dim, int degree, bool normalize)
{
	TORCH_CHECK(degree >= 0, "SH degree must be non-negative");
	TORCH_CHECK(degree <= MaxDegree(), "only ", MaxDegree(), " degrees supported");
	TORCH_CHECK((-t.dim() <= dim) && (dim < t.dim()), "dimension must be in -t.dim() <= dim < t.dim()");
	if (dim < 0) dim = t.dim() + dim;

	CHECK_CUDA(t, true);
	CHECK_SIZE(t, dim, 3);
	CHECK_CONTIGUOUS(t); //for simplicity

	int N = GetCoefficientCount(degree);

	const auto& oldSize = t.sizes();
	torch::SmallVector<int64_t, 5> newSize(oldSize);
	newSize[dim] = N;
	torch::Tensor output = torch::empty(newSize, t.options());

	int64_t A = 1, B = 1;
	for (int i = 0; i < dim; ++i) A *= t.size(i);
	for (int i = dim + 1; i < t.dim(); ++i) B *= t.size(i);
	//t can be written as of shape [A,3,B]
	//output of shape [A,N,B]

	CUstream stream = c10::cuda::getCurrentCUDAStream();
	RENDERER_DISPATCH_FLOATING_TYPES(t.scalar_type(), "SphericalHarmonics::Evaluate", [&]()
		{
			int64_t inSizes[] = { A, 3, B };
			int64_t inStrides[] = { 3 * B, B, 1 };
			::kernel::Tensor3Read<scalar_t> in(t.data_ptr<scalar_t>(), inSizes, inStrides);

			int64_t outSizes[] = { A, N, B };
			int64_t outStrides[] = { N * B, B, 1 };
			::kernel::Tensor3RW<scalar_t> out(output.data_ptr<scalar_t>(), outSizes, outStrides);

			::kernel::SphericalHarmonicsImpl(in, out, degree, normalize, stream);
		});
	
	return output;
}


