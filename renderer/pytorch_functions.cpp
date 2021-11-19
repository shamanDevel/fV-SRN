#include "pytorch_functions.h"

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>
#include "pytorch_utils.h"

void renderer::PytorchFunctions::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;

	namespace py = pybind11;

    m.def("interp1D", &PytorchFunctions::interp1D,
        py::doc(R"(
     Given a piecewise function defined by \c fp (see below), interpolate that function
     at the positions given by \c x.
     
     The tensor \c fp has shape (B,C,N) with B being the batch dimension, C the channels
     and N the number of control points.
     Then the function is defined using piecewise linear interpolation with the control
     points at x-positions 0,1,2,...,N-1 given by fp[b,:,x].
     
     This function is evaluated at the positions given by tensor \c x of shape
     (B,M) where B is the batch dimension and M is the number of sample points.
     Values lower than zero or larger than N-1 are clamped.
     
     The output is a tensor of shape (B,C,M) where entry \c out[b,:,m] is the linear
     interpolation of the values of \c fp at the position \c x[b,m].
     
     The batch dimensions of \c fp and \c x must match or be equal to 1 for broadcasting.
     Supported types: CUDA tensors of dtype \c float and \c double.
     
     This function is differentiable.
     
     :param fp: values at the control points of shape (B,C,N)
     :param x: positions to interpolate of shape (B,M)
     :return: interpolated values of shape (B,C,M)
        )"),
        py::arg("fp"), py::arg("x"))
        ;
}

namespace
{
    class Interp1DFunction : public torch::autograd::Function<Interp1DFunction>
    {
    public:
        static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx, torch::Tensor fp, torch::Tensor x)
        {
            CHECK_CUDA(fp, true);
            CHECK_DIM(fp, 3);
            int b = CHECK_BATCH(fp, 1);
            int c = fp.size(1);

            CHECK_CUDA(x, true);
            CHECK_DIM(x, 2);
            b = CHECK_BATCH(x, b);
            int m = x.size(1);
            TORCH_CHECK(fp.dtype() == x.dtype(), "fp and x must have the same dtype");

            ctx->save_for_backward({ fp, x });
            torch::Tensor out = torch::empty({ b, c, m }, fp.options());

            CUstream stream = c10::cuda::getCurrentCUDAStream();
            RENDERER_DISPATCH_FLOATING_TYPES(fp.scalar_type(), "interp1D-forward", [&]()
                {
                    auto fp_acc = accessor<kernel::Tensor3Read<scalar_t>>(fp);
                    auto x_acc = accessor<kernel::Tensor2Read<scalar_t>>(x);
                    auto out_acc = accessor<kernel::Tensor3RW<scalar_t>>(out);
                    ::kernel::PytorchFunctions_inter1D_forward(fp_acc, x_acc, out_acc, stream);
                });
            return out;
        }

        static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx, torch::autograd::tensor_list grad_outputs)
        {
            auto saved = ctx->get_saved_variables();
            auto fp = saved[0];
            auto x = saved[1];
            auto grad_output = grad_outputs[0];

            int b = std::max(fp.size(0), x.size(0));
            int c = fp.size(1);
            int m = x.size(1);

            auto grad_fp = torch::zeros_like(fp);
            auto grad_x = torch::zeros_like(x);

            CUstream stream = c10::cuda::getCurrentCUDAStream();
            RENDERER_DISPATCH_FLOATING_TYPES(fp.scalar_type(), "interp1D-backward", [&]()
                {
                    auto fp_acc = accessor<kernel::Tensor3Read<scalar_t>>(fp);
                    auto x_acc = accessor<kernel::Tensor2Read<scalar_t>>(x);
                    auto grad_out_acc = accessor<kernel::Tensor3Read<scalar_t>>(grad_output);
                    auto grad_fp_acc = accessor<kernel::BTensor3RW<scalar_t>>(grad_fp);
                    auto grad_x_acc = accessor<kernel::BTensor2RW<scalar_t>>(grad_x);
                    ::kernel::PytorchFunctions_inter1D_backward(
                        fp_acc, x_acc, grad_out_acc, grad_fp_acc, grad_x_acc, stream);
                });

            return { grad_fp, grad_x };
        }
    };
}

torch::Tensor renderer::PytorchFunctions::interp1D(const torch::Tensor& fp, const torch::Tensor& x)
{
    return Interp1DFunction::apply(fp, x);
}
