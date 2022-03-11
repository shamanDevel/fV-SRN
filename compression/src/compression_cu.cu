#include <cuMat/src/Context.h>

#include <cuda_runtime.h>

namespace {
    __global__ void FillChunkKernelFloat(
        dim3 virtual_size,
        cudaSurfaceObject_t dst,
        const float* src,
        int strideX, int strideY, int strideZ)
    {
        CUMAT_KERNEL_3D_LOOP(x, y, z, virtual_size)
        {
            float v = src[x * strideX + y * strideY + z * strideZ];
            surf3Dwrite(v, dst, x*sizeof(float), y, z);
        }
        CUMAT_KERNEL_3D_LOOP_END
    }

    __global__ void FillChunkKernelUChar(
        dim3 virtual_size,
        cudaSurfaceObject_t dst,
        const float* src,
        int strideX, int strideY, int strideZ)
    {
        CUMAT_KERNEL_3D_LOOP(x, y, z, virtual_size)
        {
            float v = src[x * strideX + y * strideY + z * strideZ];
            v = fmaxf(0.f, fminf(1.f, v));
            unsigned char vi = static_cast<unsigned char>(v * 0xff);
            surf3Dwrite(vi, dst, x, y, z);
        }
        CUMAT_KERNEL_3D_LOOP_END
    }

    __global__ void FillChunkKernelUShort(
        dim3 virtual_size,
        cudaSurfaceObject_t dst,
        const float* src,
        int strideX, int strideY, int strideZ)
    {
        CUMAT_KERNEL_3D_LOOP(x, y, z, virtual_size)
        {
            float v = src[x * strideX + y * strideY + z * strideZ];
            v = fmaxf(0.f, fminf(1.f, v));
            unsigned short vi = static_cast<unsigned short>(v * 0xffff);
            surf3Dwrite(vi, dst, x*sizeof(short), y, z);
        }
        CUMAT_KERNEL_3D_LOOP_END
    }
}

namespace compression
{
    void fillChunkFloat(
        cudaSurfaceObject_t dst,
        const float* src,
        int sizeX, int sizeY, int sizeZ,
        int strideX, int strideY, int strideZ,
        cudaStream_t stream)
    {
        auto& ctx = cuMat::Context::current();
        auto cfg = ctx.createLaunchConfig3D(sizeX, sizeY, sizeZ, FillChunkKernelFloat);
        FillChunkKernelFloat <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (
            cfg.virtual_size, dst, src, strideX, strideY, strideZ);
        CUMAT_CHECK_ERROR();
    }

    void fillChunkUChar(
        cudaSurfaceObject_t dst,
        const float* src,
        int sizeX, int sizeY, int sizeZ,
        int strideX, int strideY, int strideZ,
        cudaStream_t stream)
    {
        auto& ctx = cuMat::Context::current();
        auto cfg = ctx.createLaunchConfig3D(sizeX, sizeY, sizeZ, FillChunkKernelUChar);
        FillChunkKernelUChar <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (
            cfg.virtual_size, dst, src, strideX, strideY, strideZ);
        CUMAT_CHECK_ERROR();
    }

    void fillChunkUShort(
        cudaSurfaceObject_t dst,
        const float* src,
        int sizeX, int sizeY, int sizeZ,
        int strideX, int strideY, int strideZ,
        cudaStream_t stream)
    {
        auto& ctx = cuMat::Context::current();
        auto cfg = ctx.createLaunchConfig3D(sizeX, sizeY, sizeZ, FillChunkKernelUShort);
        FillChunkKernelUShort <<< cfg.block_count, cfg.thread_per_block, 0, stream >>> (
            cfg.virtual_size, dst, src, strideX, strideY, strideZ);
        CUMAT_CHECK_ERROR();
    }
}