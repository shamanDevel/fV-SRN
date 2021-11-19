#pragma once

/*
 * A few helpers to bridge cudAD and the renderer
 */

#include "forward.h"
#include "forward_vector.h"
#include "renderer_commons.cuh"

namespace cudAD
{

    template<int N>
    CUDAD_CALL_NOCONSTEXPR fvar<float2, N> make_real2in(
        const float2& xy, const int2& indices)
    {
        return make_float2in<N>(xy, indices);
    }
    template<int N>
    CUDAD_CALL_NOCONSTEXPR fvar<float3, N> make_real3in(
        const float3& xy, const int3& indices)
    {
        return make_float3in<N>(xy, indices);
    }
    template<int N>
    CUDAD_CALL_NOCONSTEXPR fvar<float4, N> make_real4in(
        const float4& xy, const int4& indices)
    {
        return make_float4in<N>(xy, indices);
    }
    template<int N>
    CUDAD_CALL_NOCONSTEXPR fvar<double2, N> make_real2in(
        const double2& xy, const int2& indices)
    {
        return make_double2in<N>(xy, indices);
    }
    template<int N>
    CUDAD_CALL_NOCONSTEXPR fvar<double3, N> make_real3in(
        const double3& xy, const int3& indices)
    {
        return make_double3in<N>(xy, indices);
    }
    template<int N>
    CUDAD_CALL_NOCONSTEXPR fvar<double4, N> make_real4in(
        const double4& xy, const int4& indices)
    {
        return make_double4in<N>(xy, indices);
    }

}
	