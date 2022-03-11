#ifndef __TUM3D__DWT_FLOAT_FILTERS_CUH__
#define __TUM3D__DWT_FLOAT_FILTERS_CUH__


namespace cudaCompress {

namespace util {

// 0 : CDF 9/7
// 1 : CDF 5/3
// 2 : Daubechies 4 (with broken boundaries, because we use mirrored extension)
// 3 : Haar
#define CUDACOMPRESS_DWT_FLOAT_FILTER 0


#if CUDACOMPRESS_DWT_FLOAT_FILTER == 0
#define FILTER_LENGTH 9
#define FILTER_OFFSET 4
#elif CUDACOMPRESS_DWT_FLOAT_FILTER == 1
#define FILTER_LENGTH 5
#define FILTER_OFFSET 2
#elif CUDACOMPRESS_DWT_FLOAT_FILTER == 2
#define FILTER_LENGTH 5
#define FILTER_OFFSET 2
#elif CUDACOMPRESS_DWT_FLOAT_FILTER == 3
#define FILTER_LENGTH 3
#define FILTER_OFFSET 1
#endif

#define FILTER_OFFSET_RIGHT (FILTER_LENGTH - FILTER_OFFSET - 1)

#define COEFFICIENT_COUNT (2 * FILTER_LENGTH)

#define FILTER_OVERHEAD_LEFT_INV ((FILTER_OFFSET+1)/2*2)
#define FILTER_OVERHEAD_RIGHT_INV ((FILTER_OFFSET_RIGHT+1)/2*2)
#define FILTER_OVERHEAD_INV (FILTER_OVERHEAD_LEFT_INV + FILTER_OVERHEAD_RIGHT_INV)

// The forward filters are normalized so that the (nominal) analysis gain is 1 for both low-pass and high-pass.
// (according to final draft March 2000, JPEG2000 uses a gain of 2 in the high-pass filter and adjusts the quantization steps accordingly)
// (however, "JPEG 2000 image compression fundamentals, standards and practice" lists the numbers below)
#if CUDACOMPRESS_DWT_FLOAT_FILTER == 0
__device__ static const float g_ForwardFilterCoefficients[COEFFICIENT_COUNT] = {
    0.0267488f,-0.0168641f,-0.0782233f, 0.2668641f, 0.6029490f, 0.2668641f,-0.0782233f,-0.0168641f, 0.0267488f, // low-pass
    0.0f,       0.0456359f,-0.0287718f,-0.2956359f, 0.5575435f,-0.2956359f,-0.0287718f, 0.0456359f, 0.0f        // high-pass
};
__device__ static const float g_InverseFilterCoefficients[COEFFICIENT_COUNT] = {
    0.0f,       0.0337282f,-0.0575435f,-0.5337281f, 1.1150871f,-0.5337281f,-0.0575435f, 0.0337282f, 0.0f,       // even (interleaved lp and hp)
    0.0534975f,-0.0912718f,-0.1564465f, 0.5912718f, 1.2058980f, 0.5912718f,-0.1564465f,-0.0912718f, 0.0534975f  // odd  (interleaved hp and lp)
};
#elif CUDACOMPRESS_DWT_FLOAT_FILTER == 1
__device__ static const float g_ForwardFilterCoefficients[COEFFICIENT_COUNT] = {
    -0.125f, 0.25f, 0.75f, 0.25f,-0.125f,
     0.0f,  -0.25f, 0.5f, -0.25f, 0.0f
};
__device__ static const float g_InverseFilterCoefficients[COEFFICIENT_COUNT] = {
     0.0f,  -0.5f,  1.0f, -0.5f,  0.0f,
    -0.25f,  0.5f,  1.5f,  0.5f, -0.25f
};
#elif CUDACOMPRESS_DWT_FLOAT_FILTER == 2
__device__ static const float g_ForwardFilterCoefficients[COEFFICIENT_COUNT] = {
     0.0f,       0.34150635f, 0.59150635f,  0.15849365f, -0.091506f,
    -0.091506f, -0.15849365f, 0.59150635f, -0.34150635f,  0.0f
};
__device__ static const float g_InverseFilterCoefficients[COEFFICIENT_COUNT] = {
    -0.183012f, -0.6830127f, 1.1830127f, -0.3169873f,  0.0f,
     0.0f,       0.3169873f, 1.1830127f,  0.6830127f, -0.183012f
};
#elif CUDACOMPRESS_DWT_FLOAT_FILTER == 3
__device__ static const float g_ForwardFilterCoefficients[COEFFICIENT_COUNT] = {
     0.0f, 0.5f, 0.5f,
    -0.5f, 0.5f, 0.0f
};
__device__ static const float g_InverseFilterCoefficients[COEFFICIENT_COUNT] = {
     0.0f, 1.0f,-1.0f,
     1.0f, 1.0f, 0.0f
};
#endif


}

}


#endif
