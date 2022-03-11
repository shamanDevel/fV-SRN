#pragma once
/*
 * Activation functions for renderer_volume_tensorcores.cuh.
 *
 * Each activation function provides two functions:
 *     z = forward(v, parameter)
 *     vAdj = adjoint(v, zAdj, parameter)
 * Each function is defined for at least double-precision floats (for unit tests).
 * Activation functions used in the last layer are also defined for single-precision floats
 * and activations for hidden layers are defined for half and half2.
 */

#include "helper_math.cuh"

#ifndef __CUDACC__
#include <exception>
#endif
#include <cuda_fp16.h>

namespace kernel
{

	namespace activations
	{
		struct None {
			template<typename T>
			static __host__ __device__ __forceinline__ T forward(const T& v, float /*param*/)
			{
			    return v;
			}

			template<typename T>
			static __host__ __device__ __forceinline__ T adjoint(const T& v, const T& zAdj, float /*param*/)
			{
			    return zAdj;
			}
		};

		struct ReLU {
			static __host__ __device__ __forceinline__ half forward(const half& v, float /*param*/)
			{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
				const half ZERO{ __half_raw{0} };
				return __hmax(v, ZERO);
#else
				return __float2half(fmaxf(0.f, __half2float(v)));
#endif
			}
			static __host__ __device__ __forceinline__ half2 forward(const half2& v, float /*param*/)
			{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
				const half2 ZERO2{ __half_raw{0}, __half_raw{0} };
				return __hmax2(v, ZERO2);
#else
				return half2{
					__float2half(fmaxf(0.f, __half2float(v.x))),
					__float2half(fmaxf(0.f, __half2float(v.y)))
				};
#endif
			}
			static __host__ __device__ __forceinline__ double forward(const double& v, float /*param*/)
			{
				return fmax(v, 0.0);
			}

			static __host__ __device__ __forceinline__ half adjoint(const half& v, const half& zAdj, float /*param*/)
			{
				const half ZERO{ __half_raw{0} };
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
				return __hge(v, ZERO) ? v : ZERO;
#else
				return __half2float(v)>0 ? zAdj : ZERO;
#endif
			}
			static __host__ __device__ __forceinline__ half2 adjoint(const half2& v, const half2& zAdj, float /*param*/)
			{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
				const half2 ZERO2{ __half_raw{0}, __half_raw{0} };
				return __hmul2(zAdj, __hget2(v, ZERO2));
#else
				const half ZERO{ __half_raw{0} };
				return half2{
					__half2float(v.x) > 0 ? zAdj.x : ZERO,
					__half2float(v.y) > 0 ? zAdj.y : ZERO
				};
#endif
			}
			static __host__ __device__ __forceinline__ float adjoint(const float& v, const float& zAdj, float /*param*/)
			{
				return v > 0 ? zAdj : 0.;
			}
			static __host__ __device__ __forceinline__ double adjoint(const double& v, const double& zAdj, float /*param*/)
			{
				return v > 0 ? zAdj : 0.;
			}
		};

		struct Sine {
			static __device__ __forceinline__ half forward(const half& v, float param)
			{
#ifdef __CUDA_ARCH__
				const half paramh = __float2half(param);
				return hsin(__hmul(v, paramh));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __device__ __forceinline__ half2 forward(const half2& v, float param)
			{
#ifdef __CUDA_ARCH__
				const half paramh = __float2half(param);
				const half2 param2{ paramh, paramh };
				return h2sin(__hmul2(v, param2));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __host__ __device__ __forceinline__ double forward(const double& v, float param)
			{
				return sin(v*param);
			}

			static __device__ __forceinline__ half adjoint(const half& v, const half& zAdj, float param)
			{
#ifdef __CUDA_ARCH__
				const half paramh = __float2half(param);
				return __hmul(zAdj, __hmul(paramh, hcos(__hmul(v, paramh))));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __device__ __forceinline__ half2 adjoint(const half2& v, const half2& zAdj, float param)
			{
#ifdef __CUDA_ARCH__
				const half paramh = __float2half(param);
				const half2 param2{ paramh, paramh };
				return __hmul2(zAdj, __hmul2(param2, h2cos(__hmul2(v, param2))));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __host__ __device__ __forceinline__ float adjoint(const float& v, const float& zAdj, float param)
			{
				return zAdj * param * cosf(v * param);
			}
			static __host__ __device__ __forceinline__ double adjoint(const double& v, const double& zAdj, float param)
			{
				return zAdj * param * cos(v * param);
			}
		};

		struct Sigmoid {
			static __device__ __forceinline__ half forward(const half& v, float /*param*/)
			{
#ifdef __CUDA_ARCH__
				const half ONE = __float2half(1.0f);
				return __hdiv(ONE, __hadd(ONE, hexp(__hneg(v))));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __device__ __forceinline__ half2 forward(const half2& v, float /*param*/)
			{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
				const half ONE = __float2half(1.0f);
				const half2 ONE2{ ONE, ONE };
				return __h2div(ONE2, __hadd2(ONE2, h2exp(__hneg2(v))));
#else
				return half2{ Sigmoid::forward(v.x, 0.f), Sigmoid::forward(v.y, 0.f) };
#endif
			}
			static __host__ __device__ __forceinline__ float forward(const float& v, float /*param*/)
			{
				return 1.0f / (1.0f + expf(-v));
			}
			static __host__ __device__ __forceinline__ double forward(const double& v, float /*param*/)
			{
				return 1.0 / (1.0 + exp(-v));
			}

			static __device__ __forceinline__ half adjoint(const half& v, const half& zAdj, float /*param*/)
			{
#ifdef __CUDA_ARCH__
				const half ONE = __float2half(1.0f);
				const half ev = hexp(v);
				const half ev1 = __hadd(ONE, ev);
				return __hmul(zAdj, __hdiv(ev, __hmul(ev1, ev1)));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __device__ __forceinline__ half2 adjoint(const half2& v, const half2& zAdj, float /*param*/)
			{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
				const half ONE = __float2half(1.0f);
				const half2 ONE2{ ONE, ONE };
				const half2 ev = h2exp(v);
				const half2 ev1 = __hadd2(ONE, ev);
				return __h2mul(zAdj, __h2div(ev, __h2mul(ev1, ev1)));
#else
				return half2{ Sigmoid::adjoint(v.x, zAdj.x, 0.f), Sigmoid::adjoint(v.y, zAdj.y, 0.f) };
#endif
			}
			static __host__ __device__ __forceinline__ float adjoint(const float& v, const float& zAdj, float /*param*/)
			{
				const double ev = expf(v);
				const double ev1 = ev + 1;
				return zAdj * ev / (ev1 * ev1);
			}
			static __host__ __device__ __forceinline__ double adjoint(const double& v, const double& zAdj, float /*param*/)
			{
				const double ev = exp(v);
				const double ev1 = ev + 1;
				return zAdj * ev / (ev1 * ev1);
			}
		};

		struct Softplus {
			static __host__ __device__ __forceinline__ float forward(float x, float /*param*/)
			{
				static constexpr float beta = 1.0f;
				static constexpr float inv_beta = 1.0f / beta;
				static constexpr float threshold = 20.0f;
				static constexpr float threshold_times_beta = beta * threshold;

				if (x > threshold_times_beta) return x;
				return inv_beta * logf(1 + expf(beta * x));
			}
			static __host__ __device__ __forceinline__ double forward(double x, float /*param*/)
			{
				static constexpr double beta = 1.0;
				static constexpr double inv_beta = 1.0 / beta;
				static constexpr double threshold = 20.0;
				static constexpr double threshold_times_beta = beta * threshold;

				if (x > threshold_times_beta) return x;
				return inv_beta * log(1 + exp(beta * x));
			}

			static __host__ __device__ __forceinline__ float adjoint(const float& x, const float& zAdj, float /*param*/)
			{
				static constexpr float beta = 1.0f;
				static constexpr float threshold = 20.0f;
				static constexpr float threshold_times_beta = beta * threshold;

				if (x > threshold_times_beta) return zAdj;
				const float ebx = expf(beta * x);
				return zAdj * ebx / (ebx + 1);
			}
			static __host__ __device__ __forceinline__ double adjoint(const double& x, const double& zAdj, float /*param*/)
			{
				static constexpr double beta = 1.0;
				static constexpr double threshold = 20.0;
				static constexpr double threshold_times_beta = beta * threshold;

				if (x > threshold_times_beta) return zAdj;
				const double ebx = exp(beta * x);
				return zAdj * ebx / (ebx + 1);
			}
		};

		struct Snake {
			static __device__ __forceinline__ half forward(const half& v, float param)
			{
#ifdef __CUDA_ARCH__
				const half f = __float2half(param);
				const half divf = __float2half(1.0f / param);
				const half v2 = hsin(__hmul(f, v));
				return __hadd(v, __hmul(divf, __hmul(v2, v2)));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __device__ __forceinline__ half2 forward(const half2& v, float param)
			{
#ifdef __CUDA_ARCH__
				const half f = __float2half(param);
				const half divf = __float2half(1.0f / param);
				const half2 f2{ f, f };
				const half2 divf2{ divf, divf };
				const half2 v2 = h2sin(__hmul2(f2, v));
				return __hadd2(v, __hmul2(divf2, __hmul2(v2, v2)));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __host__ __device__ __forceinline__ double forward(const double& v, float param)
			{
				const double f = param;
				const double divf = 1.0 / param;
				const double v2 = sin(f * v);
				return v + divf * v2 * v2;
			}

			static __device__ __forceinline__ half adjoint(const half& v, const half& zAdj, float param)
			{
#ifdef __CUDA_ARCH__
				const half ONE = __float2half(1.0f);
				const half paramh = __float2half(2*param);
				return __hmul(zAdj, __hadd(ONE, hsin(__hmul(paramh, v))));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __device__ __forceinline__ half2 adjoint(const half2& v, const half2& zAdj, float param)
			{
#ifdef __CUDA_ARCH__
				const half ONE = __float2half(1.0f);
				const half2 ONE2{ ONE, ONE };

				const half paramh = __float2half(2*param);
				const half2 param2{ paramh, paramh };
				const half param2h = __float2half(2 * param);
				return __hmul2(zAdj, __hadd2(ONE2, h2sin(__hmul2(param2, v))));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __host__ __device__ __forceinline__ float adjoint(const float& v, const float& zAdj, float param)
			{
				return zAdj * (1 + sinf(2 * param * v));
			}
			static __host__ __device__ __forceinline__ double adjoint(const double& v, const double& zAdj, float param)
			{
				return zAdj * (1 + sin(2 * param * v));
			}
		};

		struct SnakeAlt {
			static __device__ __forceinline__ half forward(const half& v, float param)
			{
#ifdef __CUDA_ARCH__
				const half ONE = __float2half(1.0f);
				const half f2 = __float2half(2 * param);
				const auto x0 = hcos(__hmul(f2, v));
				const auto x1 = __hsub(__hadd(v, ONE), x0);
				return __hdiv(x1, f2);
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __device__ __forceinline__ half2 forward(const half2& v, float param)
			{
#ifdef __CUDA_ARCH__
				const half ONE = __float2half(1.0f);
				const half f2 = __float2half(2 * param);

				const half2 ONE2{ ONE, ONE };
				const half2 F2{ f2, f2 };

				const auto x0 = h2cos(__hmul2(F2, v));
				const auto x1 = __hsub2(__hadd2(v, ONE2), x0);
				return __h2div(x1, F2);
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __host__ __device__ __forceinline__ double forward(const double& v, float param)
			{
				const double f2 = 2 * param;
				const double x0 = cos(f2 * v);
				const double x1 = v + 1 - x0;
				return x1 / f2;
			}

			static __device__ __forceinline__ half adjoint(const half& v, const half& zAdj, float param)
			{
#ifdef __CUDA_ARCH__
				const half f2 = __float2half(2 * param);
				const half divf2 = __float2half(1 / (2 * param));
				return __hmul(zAdj, __hadd(hsin(__hmul(f2, v)), divf2));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __device__ __forceinline__ half2 adjoint(const half2& v, const half2& zAdj, float param)
			{
#ifdef __CUDA_ARCH__
				const half f2 = __float2half(2 * param);
				const half divf2 = __float2half(1 / (2 * param));
				const half2 f2_2{ f2, f2 };
				const half2 divf2_2{ divf2, divf2 };
				return __hmul2(zAdj, __hadd2(h2sin(__hmul2(f2_2, v)), divf2_2));
#else
				throw std::exception("Attempt to invoke device-only function in host-mode");
#endif
			}
			static __host__ __device__ __forceinline__ float adjoint(const float& v, const float& zAdj, float param)
			{
				const double f2 = 2 * param;
				const double divf2 = 1 / f2;
				return zAdj * (sinf(f2 * v) + divf2);
			}
			static __host__ __device__ __forceinline__ double adjoint(const double& v, const double& zAdj, float param)
			{
				const double f2 = 2 * param;
				const double divf2 = 1 / f2;
				return zAdj * (sin(f2*v)+divf2);
			}
		};

		struct Clamp {
			static __host__ __device__ __forceinline__ float forward(float x, float /*param*/)
			{
				return clamp(x, 0.f, 1.f);
			}
			static __host__ __device__ __forceinline__ double forward(double x, float /*param*/)
			{
				return clamp(x, 0., 1.);
			}

			static __host__ __device__ __forceinline__ float adjoint(const float& v, const float& zAdj, float /*param*/)
			{
				return (0. <= v && v <= 1.) ? zAdj : 0.;
			}
			static __host__ __device__ __forceinline__ double adjoint(const double& v, const double& zAdj, float /*param*/)
			{
				return (0. <= v && v <= 1.) ? zAdj : 0.;
			}
		};
	}

}
