//#include "pytorch_functions.h" //no include

#include <cuda.h>
#include <stdexcept>
#include <cuMat/src/Context.h>

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
#include "helper_math.cuh"

namespace
{
#if __CUDA_ARCH__ < 600
    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                   __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }
    __device__ float atomicAdd(float* address, float val)
    {
        return ::atomicAdd(address, val);
    }
#endif

    template<typename T>
    __global__ void Interp1DForwardKernel(dim3 virtual_size,
        const kernel::Tensor3Read<T> fp, const kernel::Tensor2Read<T> x,
        kernel::Tensor3RW<T> out)
    {
        const int C = fp.size(1);
        const int N = fp.size(2);
        CUMAT_KERNEL_2D_LOOP(b,m, virtual_size)
        {
            T xv = x[b][m];
            int ilow = static_cast<int>(xv);
            T fx = xv - ilow;
            //clamp
            ilow = clamp(ilow, 0, N - 1);
            int ihigh = min(ilow + 1, N - 1);
            //interpolate
            for (int c=0; c<C; ++c)
            {
                out[b][c][m] = (1 - fx) * fp[b][c][ilow] + fx * fp[b][c][ihigh];
            }
        }
        CUMAT_KERNEL_2D_LOOP_END
    }

    template<typename T>
    void Interp1DForwardImpl(const kernel::Tensor3Read<T>& fp, const kernel::Tensor2Read<T>& x,
        const kernel::Tensor3RW<T>& out, CUstream stream)
    {
        int B = out.size(0);
        int M = out.size(2);
        cuMat::Context& ctx = cuMat::Context::current();
        cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(
            B, M, Interp1DForwardKernel<T>);
        Interp1DForwardKernel<T>
            <<<cfg.block_count, cfg.thread_per_block, 0, stream >>>
            (cfg.virtual_size, fp, x, out);
        CUMAT_CHECK_ERROR();
        CUMAT_SAFE_CALL(cudaDeviceSynchronize());
    }

    template<typename T>
    __global__ void Interp1DBackwardKernel(dim3 virtual_size,
        const kernel::Tensor3Read<T> fp, const kernel::Tensor2Read<T> x,
        const kernel::Tensor3Read<T> grad_out, kernel::BTensor3RW<T> grad_fp,
        kernel::BTensor2RW<T> grad_x)
    {
        const int C = fp.size(1);
        const int N = fp.size(2);
        CUMAT_KERNEL_2D_LOOP(b, m, virtual_size)
        {
            T xv = x[b][m];
            int ilow = static_cast<int>(xv);
            T fx = xv - ilow;
            //clamp
            ilow = clamp(ilow, 0, N - 1);
            int ihigh = min(ilow + 1, N - 1);
            //interpolate
            T gradFx = 0;
            for (int c = 0; c < C; ++c)
            {
                //out[b][c][m] = (1 - fx) * fp[b][c][ilow] + fx * fp[b][c][ihigh];
                T gradOut = grad_out[b][c][m];
                atomicAdd(&grad_fp[b][c][ilow], (1 - fx) * gradOut);
                atomicAdd(&grad_fp[b][c][ihigh], fx * gradOut);
                gradFx += gradOut * (fp[b][c][ihigh] - fp[b][c][ilow]);
            }
            grad_x[b][m] = gradFx;
        }
        CUMAT_KERNEL_2D_LOOP_END
    }

    template<typename T>
    void Interp1DBackwardImpl(const kernel::Tensor3Read<T>& fp, const kernel::Tensor2Read<T>& x,
        const kernel::Tensor3Read<T>& grad_out, const kernel::BTensor3RW<T>& grad_fp, 
        const kernel::BTensor2RW<T>& grad_x, CUstream stream)
    {
        int B = grad_out.size(0);
        int M = grad_out.size(2);
        cuMat::Context& ctx = cuMat::Context::current();
        cuMat::KernelLaunchConfig cfg = ctx.createLaunchConfig2D(
            B, M, Interp1DBackwardKernel<T>);
        Interp1DBackwardKernel<T>
            << <cfg.block_count, cfg.thread_per_block, 0, stream >> >
            (cfg.virtual_size, fp, x, grad_out, grad_fp, grad_x);
        CUMAT_CHECK_ERROR();
        CUMAT_SAFE_CALL(cudaDeviceSynchronize());
    }
}

namespace kernel
{

    void PytorchFunctions_inter1D_forward(const Tensor3Read<float>& fp, const Tensor2Read<float>& x,
        const Tensor3RW<float>& out, CUstream stream)
    {
        Interp1DForwardImpl(fp, x, out, stream);
    }

    void PytorchFunctions_inter1D_forward(const Tensor3Read<double>& fp, const Tensor2Read<double>& x,
        const Tensor3RW<double>& out, CUstream stream)
    {
        Interp1DForwardImpl(fp, x, out, stream);
    }

    void PytorchFunctions_inter1D_backward(const Tensor3Read<float>& fp, const Tensor2Read<float>& x,
        const Tensor3Read<float>& grad_out, const BTensor3RW<float>& grad_fp, const BTensor2RW<float>& grad_x,
        CUstream stream)
    {
        Interp1DBackwardImpl(fp, x, grad_out, grad_fp, grad_x, stream);
    }

    void PytorchFunctions_inter1D_backward(const Tensor3Read<double>& fp, const Tensor2Read<double>& x,
        const Tensor3Read<double>& grad_out, const BTensor3RW<double>& grad_fp, const BTensor2RW<double>& grad_x,
        CUstream stream)
    {
        Interp1DBackwardImpl(fp, x, grad_out, grad_fp, grad_x, stream);
    }

}
