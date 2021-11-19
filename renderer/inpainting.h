#pragma once

#include "commons.h"
#include <cuMat/src/Matrix.h>

BEGIN_RENDERER_NAMESPACE

struct MY_API Inpainting
{
	typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::ColumnMajor> MaskTensor;
	typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> DataTensor;

	/**
	 * \brief Applies fast inpainting via down- and upsampling
	 *
	 * The mask is defined as:
	 *  - 1: non-empty pixel
	 *  - 0: empty pixel
	 *  To be precise, the input mask is checked by >=0.5.
	 * 
	 * \param mask the mask mask
	 * \param data the data matrix, channels are in the batch-dimension
	 * \return the inpainted data of the same shape as the data
	 */
	static DataTensor fastInpaint(
		const MaskTensor& mask,
		const DataTensor& data);

	/**
	 * \brief Applies fast inpainting via down- and upsampling
	 * with fractional masks. This has a similar effect as the adaptive smoothing.
	 *
	 * The mask is defined as:
	 *  - 1: non-empty pixel
	 *  - 0: empty pixel
	 *  and any fraction in between.
	 *
	 * \param mask the mask mask
	 * \param data the data matrix, channels are in the batch-dimension
	 * \return the inpainted data of the same shape as the data
	 */
	static DataTensor fastInpaintFractional(
		const MaskTensor& mask,
		const DataTensor& data);

};

END_RENDERER_NAMESPACE
