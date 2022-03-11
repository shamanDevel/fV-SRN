#pragma once

#include "renderer_commons.cuh"
#include "helper_matrixmath.cuh"
#include <forward.h>

/**
 * Before this include file, the generated code with the parameters and evaluation
 * function is included.
 * This generated code defines VOLUME_INTERPOLATION_IMPLICIT__CODE_GENERATION.
 */

#ifndef VOLUME_INTERPOLATION_IMPLICIT__CODE_GENERATION
//for intelli-sense
namespace kernel
{
    struct VolumeInterpolationImplicitParameters
    {
        float4 sourceBoxMin;
        float4 sourceBoxSize;
        float targetBoxMin;
        float targetBoxSize;
        float hessianStepsize;
        //the code generation might define more parameters
    };

    __host__ __device__ __forceinline__
    float VolumeInterpolationImplicit_forward(float x, float y, float z, float t)
    {
        static_assert(false, "implemented in code generation");
    }

    __host__ __device__ __forceinline__
    float3 VolumeInterpolationImplicit_gradient(float x, float y, float z, float t)
    {
        static_assert(false, "implemented in code generation");
    }
}

__constant__ kernel::VolumeInterpolationImplicitParameters volumeInterpolationImplicitParameters;

#endif

namespace kernel
{
    template<typename Value_t = real_t>
    struct VolumeImplicitOutput
    {
        Value_t value;
        bool isInside;
    };

    struct VolumeInterpolationImplicit
    {
        __host__ __device__ __forceinline__
            real3 getBoxMin() const
        {
            return make_real3(
                volumeInterpolationImplicitParameters.sourceBoxMin.x,
                volumeInterpolationImplicitParameters.sourceBoxMin.y,
                volumeInterpolationImplicitParameters.sourceBoxMin.z);
        }
        __host__ __device__ __forceinline__
            real3 getBoxSize() const
        {
            return make_real3(
                volumeInterpolationImplicitParameters.sourceBoxSize.x,
                volumeInterpolationImplicitParameters.sourceBoxSize.y,
                volumeInterpolationImplicitParameters.sourceBoxSize.z);
        }
        using density_t = real_t;

        //transform from source box to target box
        __device__ __inline__ real3 transformPosition(real3 position)
        {
            position = (position - getBoxMin()) / getBoxSize();
            return (position * volumeInterpolationImplicitParameters.targetBoxSize)
                + volumeInterpolationImplicitParameters.targetBoxMin;
        }
        __device__ __inline__ real3 normalScale()
        {
            return volumeInterpolationImplicitParameters.targetBoxSize / getBoxSize();
        }

        template<typename Value_t>
        __device__ __inline__ VolumeImplicitOutput<Value_t> eval(real3 position, real3 direction, int batch)
        {
            bool isInside = all(position >= getBoxMin()) &&
                all(position <= getBoxMin()+getBoxSize());

            //transport position from sourceBox to targetBox
            position = transformPosition(position);

            //TODO: time
            float time = 0;
            auto value = VolumeInterpolationImplicit_forward(
                position.x, position.y, position.z, time);

            return VolumeImplicitOutput<Value_t>{ value, isInside };
        }

        __device__ __inline__ real3 evalNormalImpl(real3 position)
        {
            //transform position from sourceBox to targetBox
            position = transformPosition(position);

            //TODO: time
            float time = 0;
            real3 normal = make_real3(VolumeInterpolationImplicit_gradient(position.x, position.y, position.z, time));

            return normal * normalScale();
        }

        template<typename Value_t>
        __device__ __inline__ real3 evalNormal(real3 position, real3 direction,
            const VolumeImplicitOutput<Value_t>& resultFromEval, int batch)
        {
            return evalNormalImpl(position);
        }

        template<typename Value_t>
        __device__ __inline__ real2 evalCurvature(real3 position, real3 direction,
            const VolumeImplicitOutput<Value_t>& resultFromEval, int batch)
        {
            // 1. measure gradient
            real3 g = evalNormalImpl(position);
            real_t gNorm = rmax(length(g), real_t(1e-7));
            real3 n = -g / gNorm;
            real3x3 P = real3x3::Identity() - real3x3::OuterProduct(n, n);

            // 2. measure Hessian matrix
            real_t h = volumeInterpolationImplicitParameters.hessianStepsize;
            real_t denom = 1 / (2 * h);
            real3x3 Hprime = real3x3::FromColumns(
                denom * (evalNormalImpl(position + make_real3(h, 0, 0)) - evalNormalImpl(position - make_real3(h, 0, 0))),
                denom * (evalNormalImpl(position + make_real3(0, h, 0)) - evalNormalImpl(position - make_real3(0, h, 0))),
                denom * (evalNormalImpl(position + make_real3(0, 0, h)) - evalNormalImpl(position - make_real3(0, 0, h)))
            );
            real3x3 H = 0.5 * (Hprime + Hprime.transpose()); //make symmetric
            real3x3 G = (-1 / gNorm) * P.matmul(H.matmul(P));

            // 3. extract curvature 
            real_t T = G.trace();
            real_t F = G.frobenius();
            real_t discr = sqrtr(2 * F * F - T * T);
            real_t k1 = 0.5 * (T + discr);
            real_t k2 = 0.5 * (T - discr);
            if (isnan(k1) || isnan(k2))
                k1 = k2 = real_t(0); //degenerated point
            return make_real2(k1, k2);
        }
    };
}
