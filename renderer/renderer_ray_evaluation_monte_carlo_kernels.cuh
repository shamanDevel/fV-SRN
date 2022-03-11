#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

#include "renderer_sampler_curand.cuh"
#include "renderer_ray_evaluation_monte_carlo.cuh"

#ifndef FLT_MAX
#define FLT_MAX          3.402823466e+38F
#endif

__global__ void RayEvaluationMCSampleLight(
    dim3 virtual_size, kernel::Tensor2RW<real_t> output, unsigned int time)
{
    ::kernel::RayEvaluationMonteCarlo rayEvaluator;
    ::kernel::Sampler sampler(42, time);

    KERNEL_1D_LOOP(i, virtual_size)
    {
        real3 p = rayEvaluator.sampleLightPosition(sampler);
        output[i][0] = p.x;
        output[i][1] = p.y;
        output[i][2] = p.z;
    }
    KERNEL_1D_LOOP_END
}

__global__ void RayEvaluationMCEvalBackground(
    dim3 virtual_size,
    kernel::Tensor2Read<real_t> rayStart,
    kernel::Tensor2Read<real_t> rayDir,
    kernel::Tensor2RW<real_t> output, unsigned int time)
{
    ::kernel::RayEvaluationMonteCarlo rayEvaluator;
    ::kernel::Sampler sampler(42, time);

    KERNEL_1D_LOOP(i, virtual_size)
    {
        real3 start = make_real3(rayStart[i][0], rayStart[i][1], rayStart[i][2]);
        real3 dir = make_real3(rayDir[i][0], rayDir[i][1], rayDir[i][2]);
        auto o = rayEvaluator.evalBackground(start, dir, 0, sampler);
        output[i][0] = o.color.x;
        output[i][1] = o.color.y;
        output[i][2] = o.color.z;
        output[i][3] = o.color.w;
    }
    KERNEL_1D_LOOP_END
}

__global__ void RayEvaluationMCNextDir(
    dim3 virtual_size,
    kernel::Tensor2Read<real_t> rayStart,
    kernel::Tensor2Read<real_t> rayDir,
    kernel::Tensor2RW<real_t> outDir,
    kernel::Tensor2RW<real_t> outBetaScale,
    unsigned int time)
{
    ::kernel::RayEvaluationMonteCarlo rayEvaluator;
    ::kernel::Sampler sampler(42, time);

    KERNEL_1D_LOOP(i, virtual_size)
    {
        real3 start = make_real3(rayStart[i][0], rayStart[i][1], rayStart[i][2]);
        real3 dir = make_real3(rayDir[i][0], rayDir[i][1], rayDir[i][2]);

        real3 nextDir = rayEvaluator.phase.sample(dir, start, sampler, 0);
        real_t betaScale = rayEvaluator.phase.prob(dir, nextDir, start, 0);

        outDir[i][0] = nextDir.x;
        outDir[i][1] = nextDir.y;
        outDir[i][2] = nextDir.z;
        outBetaScale[i][0] = betaScale;
    }
    KERNEL_1D_LOOP_END
}

__global__ void RayEvaluationMCPhaseFunctionProbability(
    dim3 virtual_size,
    kernel::Tensor2Read<real_t> dirIn,
    kernel::Tensor2Read<real_t> dirOut,
    kernel::Tensor2Read<real_t> pos,
    kernel::Tensor2RW<real_t> output)
{
    ::kernel::RayEvaluationMonteCarlo rayEvaluator;

    KERNEL_1D_LOOP(i, virtual_size)
    {
        real3 dIn = make_real3(dirIn[i][0], dirIn[i][1], dirIn[i][2]);
        real3 dOut = make_real3(dirOut[i][0], dirOut[i][1], dirOut[i][2]);
        real3 p = make_real3(pos[i][0], pos[i][1], pos[i][2]);
        auto o = rayEvaluator.phase.prob(dIn, dOut, p, 0);
        output[i][0] = o;
    }
    KERNEL_1D_LOOP_END
}
