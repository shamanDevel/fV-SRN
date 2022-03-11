#pragma once

#include "helper_math.cuh"
#include "renderer_commons.cuh"


//-------------------------------------
// Small matrix math on build-in types
//-------------------------------------

namespace kernel{
    
/**
 * \brief Row-major 3x3 matrix.
 * The three rows are stored using real3 (=build-in float4).
 */
struct real3x3
{
    real3 r1, r2, r3;

    __host__ __device__ __inline__ real3x3() {}
    explicit __host__ __device__ __inline__ real3x3(real_t v) : r1(make_real3(v,v,v)), r2(make_real3(v,v,v)), r3(make_real3(v,v,v)) {}
    __host__ __device__ __inline__ real3x3(const real3& r1, const real3& r2, const real3& r3) : r1(r1), r2(r2), r3(r3) {}

    static __host__ __device__ __inline__ real3x3 FromRows(const real3& r1, const real3& r2, const real3& r3)
    {
        return real3x3(r1, r2, r3);
    }

    static __host__ __device__ __inline__ real3x3 FromColumns(const real3& c1, const real3& c2, const real3& c3)
    {
        return real3x3(
            make_real3(c1.x, c2.x, c3.x),
            make_real3(c1.y, c2.y, c3.y),
            make_real3(c1.z, c2.z, c3.z));
    }
    
    static __host__ __device__ __inline__ real3x3 Identity()
    {
        return real3x3(make_real3(1, 0, 0), make_real3(0, 1, 0), make_real3(0, 0, 1));
    }

    static __host__ __device__ __inline__ real3x3 SingleEntry(int i, int j)
    {
        real3x3 m;
        m.r1 = i == 0 ? make_real3(j == 0 ? 1 : 0, j == 1 ? 1 : 0, j == 2 ? 1 : 0) : make_real3(0, 0, 0);
        m.r2 = i == 1 ? make_real3(j == 0 ? 1 : 0, j == 1 ? 1 : 0, j == 2 ? 1 : 0) : make_real3(0, 0, 0);
        m.r3 = i == 2 ? make_real3(j == 0 ? 1 : 0, j == 1 ? 1 : 0, j == 2 ? 1 : 0) : make_real3(0, 0, 0);
        return m;
    }

    __host__ __device__ __inline__ real_t& entry(int i, int j)
    {
        if (i==0)
        {
            if (j == 0) return r1.x;
            if (j == 1) return r1.y;
            return r1.z;
        } else if (i==1)
        {
            if (j == 0) return r2.x;
            if (j == 1) return r2.y;
            return r2.z;
        } else
        {
            if (j == 0) return r3.x;
            if (j == 1) return r3.y;
            return r3.z;
        }
    }
    __host__ __device__ __inline__ const real_t& entry(int i, int j) const
    {
        if (i == 0)
        {
            if (j == 0) return r1.x;
            if (j == 1) return r1.y;
            return r1.z;
        }
        else if (i == 1)
        {
            if (j == 0) return r2.x;
            if (j == 1) return r2.y;
            return r2.z;
        }
        else
        {
            if (j == 0) return r3.x;
            if (j == 1) return r3.y;
            return r3.z;
        }
    }

    __host__ __device__ __inline__ real3x3 operator+(const real3x3& other) const
    {
        return real3x3(r1 + other.r1, r2 + other.r2, r3 + other.r3);
    }
    __host__ __device__ __inline__ real3x3 operator-(const real3x3& other) const
    {
        return real3x3(r1 - other.r1, r2 - other.r2, r3 - other.r3);
    }
    __host__ __device__ __inline__ real3x3 operator-() const
    {
        return real3x3(-r1, -r2, -r3);
    }
    __host__ __device__ __inline__ real3x3& operator+=(const real3x3& other)
    {
        r1 += other.r1;
        r2 += other.r2;
        r3 += other.r3;
        return *this;
    }
    __host__ __device__ __inline__ real3x3& operator-=(const real3x3& other)
    {
        r1 -= other.r1;
        r2 -= other.r2;
        r3 -= other.r3;
        return *this;
    }
    __host__ __device__ __inline__ real3x3 operator*(const real3x3& other) const //cwise multiplication
    {
        return real3x3(r1 * other.r1, r2 * other.r2, r3 * other.r3);
    }

    __host__ __device__ __inline__ real_t det() const
    {
        return (r1.x*r2.y*r3.z + r1.y*r2.z*r3.x + r1.z*r2.x*r3.y - r1.z*r2.y*r3.x - r1.x*r2.z*r3.y - r1.y*r2.x*r3.z);
    }

    __host__ __device__ __inline__ real_t trace() const
    {
        return r1.x + r2.y + r3.z;
    }

    //Computes the Frobenius norm
    __host__ __device__ __inline__ real_t frobenius() const
    {
        return sqrtr(
            r1.x * r1.x + r1.y * r1.y + r1.z * r1.z +
            r2.x * r2.x + r2.y * r2.y + r2.z * r2.z +
            r3.x * r3.x + r3.y * r3.y + r3.z * r3.z
        );
    }

    __host__ __device__ __inline__ real3x3 inverse(bool& canInvert, const real_t eps = 1e-5) const
    {
        real_t det = r1.x * r2.y * r3.z + r1.y * r2.z * r3.x + r1.z * r2.x * r3.y - r1.z * r2.y * r3.x - r1.x * r2.z * r3.y - r1.y * r2.x * r3.z;
        if (rabs(det) < eps)
        {
            canInvert = false;
            return {};
        }
        canInvert = true;
        real_t detInv = real_t(1) / det;
        return real3x3(
            detInv * make_real3(r2.y * r3.z - r2.z * r3.y, r1.z * r3.y - r1.y * r3.z, r1.y * r2.z - r1.z * r2.y),
            detInv * make_real3(r2.z * r3.x - r2.x * r3.z, r1.x * r3.z - r1.z * r3.x, r1.z * r2.x - r1.x * r2.z),
            detInv * make_real3(r2.x * r3.y - r2.y * r3.x, r1.y * r3.x - r1.x * r3.y, r1.x * r2.y - r1.y * r2.x));
    }
    
    __host__ __device__ __inline__ real3x3 inverse() const
    {
        real_t detInv = real_t(1) / (r1.x*r2.y*r3.z + r1.y*r2.z*r3.x + r1.z*r2.x*r3.y - r1.z*r2.y*r3.x - r1.x*r2.z*r3.y - r1.y*r2.x*r3.z);
        return real3x3(
            detInv * make_real3(r2.y*r3.z-r2.z*r3.y, r1.z*r3.y-r1.y*r3.z, r1.y*r2.z-r1.z*r2.y),
            detInv * make_real3(r2.z*r3.x-r2.x*r3.z, r1.x*r3.z-r1.z*r3.x, r1.z*r2.x-r1.x*r2.z),
            detInv * make_real3(r2.x*r3.y-r2.y*r3.x, r1.y*r3.x-r1.x*r3.y, r1.x*r2.y-r1.y*r2.x));
    }

    __host__ __device__ __inline__ real3x3 transpose() const
    {
        return real3x3(
            make_real3(r1.x, r2.x, r3.x),
            make_real3(r1.y, r2.y, r3.y),
            make_real3(r1.z, r2.z, r3.z));
    }

    __host__ __device__ __inline__ real3x3 matmul(const real3x3& rhs) const
    {
        return real3x3(
            make_real3(dot(r1, make_real3(rhs.r1.x, rhs.r2.x, rhs.r3.x)), dot(r1, make_real3(rhs.r1.y, rhs.r2.y, rhs.r3.y)), dot(r1, make_real3(rhs.r1.z, rhs.r2.z, rhs.r3.z))),
            make_real3(dot(r2, make_real3(rhs.r1.x, rhs.r2.x, rhs.r3.x)), dot(r2, make_real3(rhs.r1.y, rhs.r2.y, rhs.r3.y)), dot(r2, make_real3(rhs.r1.z, rhs.r2.z, rhs.r3.z))),
            make_real3(dot(r3, make_real3(rhs.r1.x, rhs.r2.x, rhs.r3.x)), dot(r3, make_real3(rhs.r1.y, rhs.r2.y, rhs.r3.y)), dot(r3, make_real3(rhs.r1.z, rhs.r2.z, rhs.r3.z)))
        );
    }

    __host__ __device__ __inline__ real3x3 matmulT(const real3x3& rhsTransposed) const
    {
        return real3x3(
            make_real3(dot(r1, rhsTransposed.r1), dot(r1, rhsTransposed.r2), dot(r1, rhsTransposed.r3)),
            make_real3(dot(r2, rhsTransposed.r1), dot(r2, rhsTransposed.r2), dot(r2, rhsTransposed.r3)),
            make_real3(dot(r3, rhsTransposed.r1), dot(r3, rhsTransposed.r2), dot(r3, rhsTransposed.r3))
        );
    }

    /**
     * \brief Multiplies vector 'right' at the right side of this matrix
     * \param right 
     * \return this * right
     */
    __host__ __device__ __inline__ real3 matmul(const real3& right) const
    {
        return make_real3(
            dot(r1, right),
            dot(r2, right),
            dot(r3, right));
    }

    /**
     * \brief Multiplies vector 'left' at the left side of this matrix
     * \param left
     * \return left^T * this
     */
    __host__ __device__ __inline__ real3 matmulLeft(const real3& left) const
    {
        return make_real3(
            dot(left, make_real3(r1.x, r2.x, r3.x)),
            dot(left, make_real3(r1.y, r2.y, r3.y)),
            dot(left, make_real3(r1.z, r2.z, r3.z))
        );
    }

    /**
     * \brief Multiplies vector 'left' at the left side of this matrix transposed
     * \param left
     * \return left^T * this^T
     */
    __host__ __device__ __inline__ real3 matmulLeftT(const real3& left) const
    {
        return make_real3(
            dot(left, r1),
            dot(left, r2),
            dot(left, r3)
        );
    }

    /**
     * \brief Computes vec(this).dot(vec(right)) = trace(this.transpose() * right)
     * \param right the other matrix
     * \return the vectorized inner product
     */
    __host__ __device__ __inline__ real_t vecProd(const real3x3& right) const
    {
        return dot(r1, right.r1) + dot(r2, right.r2) + dot(r3, right.r3);
    }

    static __host__ __device__ __inline__ real3x3 OuterProduct(const real3& left, const real3& right)
    {
        return real3x3(
            left.x*right,
            left.y*right,
            left.z*right
            );
    }
};

__host__ __device__ __inline__ real3x3 operator*(real_t scalar, const real3x3& mat)
{
    return real3x3(scalar*mat.r1, scalar*mat.r2, scalar*mat.r3);
}
__host__ __device__ __inline__ real3x3 operator*(const real3x3& mat, real_t scalar)
{
    return real3x3(scalar*mat.r1, scalar*mat.r2, scalar*mat.r3);
}

__host__ __device__ __inline__ real3 operator*(real3x3& mat, const real3& v)
{
    return mat.matmul(v);
}

__host__ __device__ __inline__ real3 operator*(const real3& v, real3x3& mat)
{
    return mat.matmulLeft(v);
}

//Returns the skew-symmetric part
__host__ __device__ __inline__ real3x3 skew(const real3x3& A)
{
    return real_t(0.5) * (A - A.transpose());
}

}

// ----------------------
// STREAMING
// ----------------------
#ifndef CUDA_NO_HOST
#include <ostream>
namespace std {
    inline std::ostream& operator << (std::ostream& os, const kernel::real3x3& v)
    {
        os << "[" << v.r1.x << "," << v.r1.y << "," << v.r1.z 
            << ";" << v.r2.x << "," << v.r2.y << "," << v.r2.z
            << ";" << v.r3.x << "," << v.r3.y << "," << v.r3.z << "]";
        return os;
    }
}
#endif
