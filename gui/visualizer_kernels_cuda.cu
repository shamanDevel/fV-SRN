#include "visualizer_kernels.h"
#include <helper_math.cuh>
#include <cuMat/src/Context.h>

#include <errors.h>
#include <inpainting.h>
#include <warping.h>


FlowTensor kernel::inpaintFlow(const ::renderer::OutputTensor& input, int maskChannel, int flowXChannel,
	int flowYChannel)
{
	CHECK_ERROR(std::abs(flowXChannel - flowYChannel) == 1,
		"flowXChannel and flowYChannel must be consecutive, but are ",
		flowXChannel, " and ", flowYChannel);
	renderer::Inpainting::MaskTensor mask = input.slice(maskChannel);
	renderer::Inpainting::DataTensor flow = input.block(
		0, 0, std::min(flowXChannel, flowYChannel), input.rows(), input.cols(), 2);
	auto out = renderer::Inpainting::fastInpaintFractional(mask, flow);
	return out;
	//return flow;
}

::renderer::OutputTensor kernel::warp(const ::renderer::OutputTensor& input, const FlowTensor& flow)
{
	return renderer::Warping::warp(input, flow);
}

std::pair<float, float> kernel::extractMinMaxDepth(const ::renderer::OutputTensor& input, int depthChannel)
{
	auto depthValue = input.slice(depthChannel); //unevaluated
	float maxDepth = static_cast<float>(depthValue.maxCoeff());
	float minDepth = static_cast<float>((depthValue + (depthValue < 1e-5).cast<float>()).maxCoeff());
	return { minDepth, maxDepth };
}

::renderer::OutputTensor kernel::lerp(const ::renderer::OutputTensor& a, const ::renderer::OutputTensor& b, float v)
{
	return (1 - v)*a + v * b;
}

