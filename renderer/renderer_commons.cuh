#pragma once

#include "helper_math.cuh"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef KERNEL_DOUBLE_PRECISION
#define KERNEL_DOUBLE_PRECISION 0
#pragma warning "KERNEL_DOUBLE_PRECISION not specified, fallback to floats"
#endif

#ifndef KERNEL_SYNCHRONIZED_TRACING
#define KERNEL_SYNCHRONIZED_TRACING 0
#endif

#if KERNEL_DOUBLE_PRECISION==0
typedef float real_t;
typedef float2 real2;
typedef float3 real3;
typedef float4 real4;
#define make_real2 make_float2
#define make_real3 make_float3
#define make_real4 make_float4
#else
typedef double real_t;
typedef double2 real2;
typedef double3 real3;
typedef double4 real4;
#define make_real2 make_double2
#define make_real3 make_double3
#define make_real4 make_double4
#endif

namespace kernel {

	template<typename T>
	struct scalar_traits;
	template<typename T>
	struct vector_traits;

	template<>
	struct scalar_traits<float>
	{
		using real_t = float;
		using real2 = float2;
		using real3 = float3;
		using real4 = float4;
	};
	template<>
	struct vector_traits<float2>
	{
		using scalar_t = float;
	};
	template<>
	struct vector_traits<float3>
	{
		using scalar_t = float;
	};
	template<>
	struct vector_traits<float4>
	{
		using scalar_t = float;
	};

	template<>
	struct scalar_traits<double>
	{
		using real_t = double;
		using real2 = double2;
		using real3 = double3;
		using real4 = double4;
	};
	template<>
	struct vector_traits<double2>
	{
		using scalar_t = double;
	};
	template<>
	struct vector_traits<double3>
	{
		using scalar_t = double;
	};
	template<>
	struct vector_traits<double4>
	{
		using scalar_t = double;
	};

	template<typename T>
	__host__ __device__ __forceinline__
		typename scalar_traits<T>::real2 cast2(const float2& v)
	{
		return typename scalar_traits<T>::real2{
			static_cast<T>(v.x),
			static_cast<T>(v.y)
		};
	}
	template<typename T>
	__host__ __device__ __forceinline__
		typename scalar_traits<T>::real3 cast3(const float3& v)
	{
		return typename scalar_traits<T>::real3{
			static_cast<T>(v.x),
			static_cast<T>(v.y),
			static_cast<T>(v.z)
		};
	}
	template<typename T>
	__host__ __device__ __forceinline__
		typename scalar_traits<T>::real4 cast4(const float4& v)
	{
		return typename scalar_traits<T>::real4{
			static_cast<T>(v.x),
			static_cast<T>(v.y),
			static_cast<T>(v.z),
			static_cast<T>(v.w)
		};
	}

	template<typename T>
	__host__ __device__ __forceinline__
	typename scalar_traits<T>::real2 cast2(const double2& v)
	{
		return typename scalar_traits<T>::real2{
			static_cast<T>(v.x),
			static_cast<T>(v.y)
		};
	}
	template<typename T>
	__host__ __device__ __forceinline__
	typename scalar_traits<T>::real3 cast3(const double3& v)
	{
		return typename scalar_traits<T>::real3{
			static_cast<T>(v.x),
			static_cast<T>(v.y),
			static_cast<T>(v.z)
		};
	}
	template<typename T>
	__host__ __device__ __forceinline__
	typename scalar_traits<T>::real4 cast4(const double4& v)
	{
		return typename scalar_traits<T>::real4{
			static_cast<T>(v.x),
			static_cast<T>(v.y),
			static_cast<T>(v.z),
			static_cast<T>(v.w)
		};
	}

	// COMMON DATASTRUCTURES
	struct RayEvaluationOutput
	{
		real4 color;
		real3 normal;
		real_t depth;
	};

	// STRUCT TEMPLATE integral_constant
	template<class _Ty, _Ty _Val>
	struct integral_constant
	{	// convenient template for integral constant types
		enum
		{
			value = _Val
		};
	};
	using true_type = ::kernel::integral_constant<bool, true>;
	using false_type = ::kernel::integral_constant<bool, false>;
	// STRUCT TEMPLATE conditional
	template <bool _Test, class _Ty1, class _Ty2>
	struct conditional { // Choose _Ty1 if _Test is true, and _Ty2 otherwise
		using type = _Ty1;
	};
	template <class _Ty1, class _Ty2>
	struct conditional<false, _Ty1, _Ty2> {
		using type = _Ty2;
	};

	// Choose _Ty1 if _Test is true, and _Ty2 otherwise
	template <bool _Test, class _Ty1, class _Ty2>
	using conditional_t = typename conditional<_Test, _Ty1, _Ty2>::type;

	template<class T, class U>
	struct is_same : ::kernel::false_type {};

	template<class T>
	struct is_same<T, T> : ::kernel::true_type {};

	template< class T, class U >
	inline constexpr bool is_same_v = is_same<T, U>::value;

	template<typename T1, typename T2>
	struct pair
	{
		T1 first;
		T2 second;
	};
	
}

#ifdef __VECTOR_FUNCTIONS_DECL__NEWLY_DEFINED__
#undef __VECTOR_FUNCTIONS_DECL__
#undef __VECTOR_FUNCTIONS_DECL__NEWLY_DEFINED__
#endif

// ----------------------
// STREAMING
// ----------------------
#ifndef CUDA_NO_HOST
#include <ostream>
namespace std {
	inline std::ostream& operator<< (std::ostream& os, const float2& v)
	{
		os << "(" << v.x << ", " << v.y << ")";
		return os;
	}
	inline std::ostream& operator<< (std::ostream& os, const float3& v)
	{
		os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
		return os;
	}
	inline std::ostream& operator<< (std::ostream& os, const float4& v)
	{
		os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
		return os;
	}
	inline std::ostream& operator<< (std::ostream& os, const double2& v)
	{
		os << "(" << v.x << ", " << v.y << ")";
		return os;
	}
	inline std::ostream& operator<< (std::ostream& os, const double3& v)
	{
		os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
		return os;
	}
	inline std::ostream& operator<< (std::ostream& os, const double4& v)
	{
		os << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
		return os;
	}
}
#endif
