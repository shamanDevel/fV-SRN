#pragma once

#include <catch.hpp>
#include <Eigen/Core>
#include <iomanip>
#include <numeric>

#include "renderer_commons.cuh"

template<typename real_t> using Vector3r = Eigen::Matrix<real_t, 3, 1, 0>;
template<typename real_t> using Vector4r = Eigen::Matrix<real_t, 4, 1, 0>;
template<typename real_t> using VectorXr = Eigen::Matrix<real_t, Eigen::Dynamic, 1, 0>;
template<typename real_t> using MatrixXr = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic, 0>;

inline Vector3r<float> toEigen(const float3& v)
{
	return Vector3r<float>(v.x, v.y, v.z);
}
inline Vector4r<float> toEigen(const float4& v)
{
	return Vector4r<float>(v.x, v.y, v.z, v.w);
}

inline Vector3r<double> toEigen(const double3& v)
{
	return Vector3r<double>(v.x, v.y, v.z);
}
inline Vector4r<double> toEigen(const double4& v)
{
	return Vector4r<double>(v.x, v.y, v.z, v.w);
}

template <typename Derived, typename T = std::remove_cv_t<std::remove_reference_t<typename Eigen::DenseBase<Derived>::CoeffReturnType>>>
inline typename kernel::scalar_traits<T>::real3 fromEigen3(const Eigen::DenseBase<Derived>& v)
{
	using v3 = typename kernel::scalar_traits<T>::real3;
	CHECK(v.size() == 3);
	if (v.rows() == 1)
	{
		CHECK(v.cols() == 3);
		return v3{ v(0, 0), v(0, 1), v(0, 2) };
	}
	else
	{
		CHECK(v.rows() == 3);
		return v3{ v(0, 0), v(1), v(2, 0) };
	}
}

template <typename Derived, typename T = std::remove_cv_t<std::remove_reference_t<typename Eigen::DenseBase<Derived>::CoeffReturnType>>>
inline typename kernel::scalar_traits<T>::real4 fromEigen4(const Eigen::DenseBase<Derived>& v)
{
	using v4 = typename kernel::scalar_traits<T>::real4;
	CHECK(v.size() == 4);
	if (v.rows() == 1)
	{
		CHECK(v.cols() == 4);
		return v4{ v(0, 0), v(0, 1), v(0, 2), v(0, 3) };
	}
	else
	{
		CHECK(v.rows() == 4);
		return v4{ v(0, 0), v(1), v(2, 0), v(3, 0) };
	}
}

struct empty {};

/*
def fibonacci_sphere(N:int, *, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
  """
  Generates points on a sphere using the Fibonacci spiral
  :param N: the number of points
  :return: a tuple (pitch/latitude, yaw/longitude)
  """
  # Source: https://bduvenhage.me/geometry/2019/07/31/generating-equidistant-vectors.html
  gr = (np.sqrt(5.0)+1.0)/2.0 # golden ratio = 1.618...
  ga = (2-gr) * (2*np.pi)     # golden angle = 2.399...
  i = np.arange(1, N+1, dtype=dtype)
  lat = np.arcsin(-1 + 2*i/(N+1))
  lon = np.remainder(ga*i, 2*np.pi)
  #lon = np.arcsin(np.sin(ga*i))
  return lat, lon
 */

/**
 * Computes points on the sphere based on the Fibonacci spiral.
 * Returns a list of real3.
 */
template<typename T, typename V = typename kernel::scalar_traits<T>::real3>
std::vector<V> fibonacciSphere(int N, V center=V{0,0,0}, T radius=1)
{
	static const T gr = (sqrt(5.0) + 1.0) / 2.0; // golden ratio = 1.618...
	static const T ga = (2 - gr) * (2 * M_PI);      // golden angle = 2.399...
	
	std::vector<V> r(N);
	for (int i = 0; i < N; ++i)
	{
		T lat = asin(-1 + 2 * (i + 1) / T(N + 1));
		T lon = remainder(ga * (i + 1), 2 * M_PI);
		r[i].x = cos(lat) * cos(lon) * radius + center.x;
		r[i].y = sin(lat) * radius + center.y;
		r[i].z = cos(lat) * sin(lon) * radius + center.z;
	}
	return r;
}

class Histogram
{
	const double min_, max_;
	const int numBins_;
	std::vector<double> bins_;

public:
	/**
	 * Creates a histogram with 'numBins' discrete bins in the interval [min, max].
	 * The first bin starts at 'min', the last bin ends at 'max'.
	 */
	Histogram(double min, double max, int numBins)
		: min_(min), max_(max), numBins_(numBins), bins_(numBins, 0.0)
	{}

	void inc(double x, double val=1)
	{
		//convert x to [0,numBins-1]
		x = (x - min_) / (max_ - min_) * numBins_;
		int ix = clamp(int(x), 0, numBins_ - 1);
		bins_[ix] += val;
	}

	double sum() const
	{
		return std::accumulate(bins_.begin(), bins_.end(), 0.0);
	}

	void scale(double s)
	{
		for (auto& e : bins_)
			e *= s;
	}

	double getMin() const { return min_; }
	double getMax() const { return max_; }
	int getNumBins() const { return numBins_; }
	double value(int bin) const { return bins_[bin]; }
};

/**
 * Computes the distance between the two histograms.
 * The histogram must be in the same range and same number of bins.
 */
inline double distance(const Histogram& a, const Histogram& b)
{
	CHECK(fabs(a.getMin() - b.getMin()) < 1e-5);
	CHECK(fabs(a.getMax() - b.getMax()) < 1e-5);
	CHECK(a.getNumBins() == b.getNumBins());

	//manhattan distance
	double v = 0;
	for (int i=0; i<a.getNumBins(); ++i)
	{
		v += fabs(a.value(i) - b.value(i));
	}
	return v;
}

