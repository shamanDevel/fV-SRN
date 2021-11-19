#pragma once

#include <cuda.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "commons.h"
#include "renderer_tensor.cuh"

BEGIN_RENDERER_NAMESPACE

/**
 * Collection of (differentiable) functions for PyTorch
 * that are not available natively.
 */
class PytorchFunctions
{
public:
    static void registerPybindModule(pybind11::module& m);

    /**
     * Given a piecewise function defined by \c fp (see below), interpolate that function
     * at the positions given by \c x.
     *
     * The tensor \c fp has shape (B,C,N) with B being the batch dimension, C the channels
     * and N the number of control points.
     * Then the function is defined using piecewise linear interpolation with the control
     * points at x-positions 0,1,2,...,N-1 given by fp[b,:,x].
     *
     * This function is evaluated at the positions given by tensor \c x of shape
     * (B,M) where B is the batch dimension and M is the number of sample points.
     * Values lower than zero or larger than N-1 are clamped.
     *
     * The output is a tensor of shape (B,C,M) where entry \c out[b,:,m] is the linear
     * interpolation of the values of \c fp at the position \c x[b,m].
     *
     * The batch dimensions of \c fp and \c x must match or be equal to 1 for broadcasting.
     * Supported types: CUDA tensors of dtype \c float and \c double.
     *
     * This function is differentiable.
     *
     * \param fp values at the control points of shape (B,C,N)
     * \param x positions to interpolate of shape (B,M)
     * \return interpolated values of shape (B,C,M)
     */
    static torch::Tensor interp1D(const torch::Tensor& fp, const torch::Tensor& x);
};

END_RENDERER_NAMESPACE

namespace kernel
{
    void PytorchFunctions_inter1D_forward(
        const Tensor3Read<float>& fp, const Tensor2Read<float>& x,
        const Tensor3RW<float>& out, CUstream stream);
    void PytorchFunctions_inter1D_forward(
        const Tensor3Read<double>& fp, const Tensor2Read<double>& x,
        const Tensor3RW<double>& out, CUstream stream);
    void PytorchFunctions_inter1D_backward(
        const Tensor3Read<float>& fp, const Tensor2Read<float>& x,
        const Tensor3Read<float>& grad_out, 
        const BTensor3RW<float>& grad_fp, const BTensor2RW<float>& grad_x,
        CUstream stream);
    void PytorchFunctions_inter1D_backward(
        const Tensor3Read<double>& fp, const Tensor2Read<double>& x,
        const Tensor3Read<double>& grad_out,
        const BTensor3RW<double>& grad_fp, const BTensor2RW<double>& grad_x,
        CUstream stream);
}
