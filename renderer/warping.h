#pragma once

#include "commons.h"
#include <cuMat/src/Matrix.h>

BEGIN_RENDERER_NAMESPACE

struct MY_API Warping
{
	typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 2, cuMat::ColumnMajor> FlowTensor;
	typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> DataTensor;

	/**
	 * \brief Warps the input image 'data' by the optical flow 'flow'.
	 * The input images data and flow are arranged with rows=height, cols=width.
	 * The optical flow contains in batch=0 the displacement along X (cols)
	 *  and in batch=1 the displacement along Y (rows).
	 * The displacement is in [-1,+1], where, e.g., +1 means that the pixel
	 * is moved completely from left to right.
	 * 
	 * Zero padding is used at the border
	 * 
	 * \param data the data tensor
	 * \param flow the flow tensor
	 * \return the warped data tensor
	 */
	static DataTensor warp(
		const DataTensor& data, const FlowTensor& flow);
};

END_RENDERER_NAMESPACE
