#pragma once

#include <cuda_runtime.h>
#include <GL/glew.h>

#include <volume.h>
#include <renderer.h>
#include "visualizer_commons.h"

/**
 * (partly) DEPREACTED
 */

namespace renderer {
	struct ShadingSettings;
}

namespace kernel
{

	/**
	 * \brief Performs fast inpainting
	 * \param input the rendered image
	 * \param maskChannel the channel that contains the mask
	 *  Entries with value <=0 are considered "empty".
	 * \param flowXChannel the channel that contains the flow along x
	 * \param flowYChannel the channel that contains the flow along x
	 * \return the inpainted tensor of shape Height * Width * 2
	 */
	FlowTensor inpaintFlow(
		const RENDERER_NAMESPACE::OutputTensor& input, 
		int maskChannel, int flowXChannel, int flowYChannel);

	/**
	 * \brief Warps the input image by the given flow field.
	 * \param input the input image
	 * \param flow the flow field (X,Y)
	 * \return the warped image
	 */
	RENDERER_NAMESPACE::OutputTensor warp(
		const RENDERER_NAMESPACE::OutputTensor& input,
		const FlowTensor& flow);

	/**
	 * \brief Extracts the minimal and maximal depth value from the input image
	 * \param input the input image
	 * \param depthChannel the channel number of the depth
	 * \return a pair with the minimal value in the first entry and the maximal value in the second.
	 */
	std::pair<float, float> extractMinMaxDepth(
		const RENDERER_NAMESPACE::OutputTensor& input, int depthChannel);

	/**
	 * Computes (1-v)*a + v*b
	 */
	RENDERER_NAMESPACE::OutputTensor lerp(
		const RENDERER_NAMESPACE::OutputTensor& a,
		const RENDERER_NAMESPACE::OutputTensor& b,
		float v);
}
