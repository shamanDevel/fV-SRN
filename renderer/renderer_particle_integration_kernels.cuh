#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"
#include "renderer_sampler_curand.cuh"

/**
 * Defines: PARTICLE_INTEGRATION__VOLUME_INTERPOLATION_T
 */

namespace kernel
{
    struct Vertex
    {
        float4 position; //x,y,z, batch
        float4 normals;  //vx,vy,vz, time
    };

}

__global__ void ParticleIntegrateSeed(
    dim3 virtual_size,
    kernel::Vertex* particles, int indexOffset, int indexLength,
    float3 seedMin, float3 seedSize, unsigned int time)
{
    ::kernel::Sampler sampler(42, time);
    KERNEL_1D_LOOP(b, virtual_size)
    {
        //sample position
        float3 pos = seedMin + seedSize * make_float3(sampler.sampleUniform(), sampler.sampleUniform(), sampler.sampleUniform());
        //create vertex instance
        int index = (b + indexOffset) % indexLength;
        particles[index].position = make_float4(pos, 0);
        particles[index].normals = make_float4(0);
    }
    KERNEL_1D_LOOP_END
}

__global__ void ParticleIntegrationAdvect(
    dim3 virtual_size, kernel::Vertex* particles, float speed)
{
    using VolumeInterpolation_t = PARTICLE_INTEGRATION__VOLUME_INTERPOLATION_T;
    VolumeInterpolation_t volume;

    KERNEL_1D_LOOP(b, virtual_size)
    {
        kernel::Vertex v = particles[b];
        if (v.normals.w>=0)
        {
            //evaluate the volume
            int batch = static_cast<int>(v.position.w);
            real3 position = make_real3(v.position.x, v.position.y, v.position.z);
            real3 direction = make_real3(v.normals.x, v.normals.y, v.normals.z);
            const auto [velocity, isInside] = volume.eval<real3>(position, direction, batch);

            //check if we left the volume
            if (!isInside)
            {
                v.normals.w = -1; //disable the particle
            } else
            {
                //forward euler advection
                v.position.x += velocity.x * speed;
                v.position.y += velocity.y * speed;
                v.position.z += velocity.z * speed;
                v.normals.x = velocity.x;
                v.normals.y = velocity.y;
                v.normals.z = velocity.z;
                v.normals.w += speed; //time
            }
            particles[b] = v;
        }
    }
    KERNEL_1D_LOOP_END
}