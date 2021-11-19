#pragma once

#include <torch/types.h>
#include <vector_types.h>
#include <cuda.h>

#include "imodule.h"
#include "kernel_loader.h"
#include "renderer_tensor.cuh"

BEGIN_RENDERER_NAMESPACE

class ModuleRegistry;

/**
 * Utility function to map unit directions in cartesian coordinates
 * to spherical harmonics basis functions.
 */
class SphericalHarmonics
{
	friend class ModuleRegistry;
	static void registerPybindModules(pybind11::module& m);
	
public:
	/**
	 * Returns the maximal degree that is supported.
	 */
	static int MaxDegree();

	/**
	 * Get the total number of coefficients for a function represented by
     * all spherical harmonic basis of degree <= \c degree.
     * Note: 0 <= degree <= MaxDegree()
     */
	static int GetCoefficientCount(int degree);

	/**
	 * Get the one-dimensional index associated with a particular degree \c l
	 * and order \c m. This is the index that can be used to access
	 * the coefficients returned by \ref Evaluate.
	 */
	static int GetIndex(int l, int m);

	/**
	 * Evaluates the spherical harmonics up to degree <= \c degree.
	 * The given tensor must have a size of 3 at the given dimension \c dim,
	 * all other dimensions are treated as batch dimension.
	 *
	 * See \ref GetIndex for the mapping from degree and order
	 * to output index
	 * 
	 * \param t the tensor of size '3' in dimension 'dim'
	 * \param dim the dimension containing the coordinates
	 * \param degree evaluates the spherical harmonics up to degree <= \c degree
	 * \param normalize false=treat the input vectors as unit, true=normalize them to unit vectors
	 * \return a tensor of shape [...,N,...] with \c N=GetCoefficientCount(degree) entries in dimension \c dim
	 */
	static torch::Tensor Evaluate(const torch::Tensor& t, int dim, int degree, bool normalize=false);
};

END_RENDERER_NAMESPACE

namespace kernel
{
	void SphericalHarmonicsImpl(
		const Tensor3Read<float>& in, Tensor3RW<float>& out,
		int degree, bool normalize, CUstream stream);
	void SphericalHarmonicsImpl(
		const Tensor3Read<double>& in, Tensor3RW<double>& out,
		int degree, bool normalize, CUstream stream);
}
