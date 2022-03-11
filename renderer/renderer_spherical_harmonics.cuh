#pragma once

#include <host_defines.h>

namespace kernel
{
constexpr int SHCoeffDegreeLimit = 4;

/**
 * Evaluates the spherical harmonics of degree l and order m,
 * where -l <= m <= l.
 * The maximal degree (inclusive) is defined by SHCoeffDegreeLimit.
 *
 * \tparam T the scalar type
 * \tparam l the degree of the SH (row, 0 <= l <= SHCoeffDegreeLimit=4)
 * \tparam m the order of the SH (col, -l <= m <= +l)
 */
template<typename T, int l, int m>	
struct SphericalHarmonicsCoeff
{
	static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z);
};

// Get the total number of coefficients for a function represented by
// all spherical harmonic basis of degree <= @degree.
constexpr __host__ __device__ __forceinline__ int SHGetCoefficientCount(int order) {
    return (order + 1) * (order + 1);
}

// Get the one dimensional index associated with a particular degree @l
// and order @m. This is the index that can be used to access the Coeffs
// returned by the spherical harmonics mapper.
constexpr __host__ __device__ __forceinline__ int SHGetIndex(int l, int m) {
    return l * (l + 1) + m;
}

//implementation

template<typename T>
struct SphericalHarmonicsCoeff<T, 0, 0>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 0.5 * sqrt(1/pi)
        return T(0.282095);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 1, -1>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -sqrt(3/(4pi)) * y
        return T(-0.488603) * y;
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 1, 0>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // sqrt(3/(4pi)) * z
        return T(0.488603) * z;
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 1, 1>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -sqrt(3/(4pi)) * x
        return T(-0.488603) * x;
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 2, -2>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 0.5 * sqrt(15/pi) * x * y
        return T(1.092548) * x * y;
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 2, -1>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.5 * sqrt(15/pi) * y * z
        return T(-1.092548) * y * z;
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 2, 0>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 0.25 * sqrt(5/pi) * (-x^2-y^2+2z^2)
        return T(0.315392) * (-x * x - y * y + T(2.0) * z * z);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 2, 1>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.5 * sqrt(15/pi) * x * z
        return T(-1.092548) * x * z;
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 2, 2>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 0.5 * sqrt(15/pi) * (x^2 - y^2)
        return T(1.092548) * (x * x - y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 3, -3>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.25 * sqrt(35/(2pi)) * y * (3x^2 - y^2)
        return T(-0.590044) * y * (3.0 * x * x - y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 3, -2>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 0.5 * sqrt(105/pi) * x * y * z
        return T(2.890611) * x * y * z;
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 3, -1>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.25 * sqrt(21/(2pi)) * y * (4z^2-x^2-y^2)
        return T(-0.457046) * y * (T(4.0) * z * z - x * x
            - y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 3, 0>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 0.25 * sqrt(7/pi) * z * (2z^2 - 3x^2 - 3y^2)
        return T(0.373176) * z * (T(2.0) * z * z - T(3.0) * x * x
            - T(3.0) * y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 3, 1>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.25 * sqrt(21/(2pi)) * x * (4z^2-x^2-y^2)
        return T(-0.457046) * x * (T(4.0) * z * z - x * x
            - y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 3, 2>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 0.25 * sqrt(105/pi) * z * (x^2 - y^2)
        return T(1.445306) * z * (x * x - y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 3, 3>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.25 * sqrt(35/(2pi)) * x * (x^2-3y^2)
        return T(-0.590044) * x * (x * x - T(3.0) * y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 4, -4>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 0.75 * sqrt(35/pi) * x * y * (x^2-y^2)
        return T(2.503343) * x * y * (x * x - y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 4, -3>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.75 * sqrt(35/(2pi)) * y * z * (3x^2-y^2)
        return T(-1.770131) * y * z * (T(3.0) * x * x - y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 4, -2>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 0.75 * sqrt(5/pi) * x * y * (7z^2-1)
        return T(0.946175) * x * y * (T(7.0) * z * z - T(1.0));
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 4, -1>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.75 * sqrt(5/(2pi)) * y * z * (7z^2-3)
        return T(-0.669047) * y * z * (T(7.0) * z * z - T(3.0));
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 4, 0>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 3/16 * sqrt(1/pi) * (35z^4-30z^2+3)
        double z2 = z * z;
        return T(0.105786) * (T(35.0) * z2 * z2 - T(30.0) * z2 + T(3.0));
    }
};
template<typename T>
struct SphericalHarmonicsCoeff<T, 4, 1>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.75 * sqrt(5/(2pi)) * x * z * (7z^2-3)
        return T(-0.669047) * x * z * (T(7.0) * z * z - T(3.0));
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 4, 2>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 3/8 * sqrt(5/pi) * (x^2 - y^2) * (7z^2 - 1)
        return T(0.473087) * (x * x - y * y)
            * (T(7.0) * z * z - T(1.0));
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 4, 3>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // -0.75 * sqrt(35/(2pi)) * x * z * (x^2 - 3y^2)
        return T(-1.770131) * x * z * (x * x - T(3.0) * y * y);
    }
};

template<typename T>
struct SphericalHarmonicsCoeff<T, 4, 4>
{
    static __host__ __device__ __forceinline__ T eval(const T& x, const T& y, const T& z) {
        // 3/16*sqrt(35/pi) * (x^2 * (x^2 - 3y^2) - y^2 * (3x^2 - y^2))
        double x2 = x * x;
        double y2 = y * y;
        return T(0.625836) * (x2 * (x2 - T(3.0) * y2) - y2 * (T(3.0) * x2 - y2));
    }
};

	
}
