#pragma once

#if RENDERER_OPENGL_SUPPORT==1

#include "commons.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <glm/glm.hpp>
#include <torch/types.h>

#include "renderer_tensor.cuh"


BEGIN_RENDERER_NAMESPACE

   class Framebuffer
{
	int width_, height_;
	GLuint fbo_;
	GLuint colorTexture_;
	GLuint depthTexture_;
	GLuint depthRbo_;
	cudaGraphicsResource_t colorTextureCuda_;
	cudaGraphicsResource_t depthTextureCuda_;

	GLuint prevBinding_;
	GLint prevViewport_[4];
	
	Framebuffer(Framebuffer const&) = delete;
	Framebuffer& operator=(Framebuffer const&) = delete;

public:
	Framebuffer(int width, int height);
	~Framebuffer();

	int width() const { return width_; }
	int height() const { return height_; }

	void bind();
	void unbind();

	void readRGBA(std::vector<float>& data);
	/**
	 * Copies the contents of the framebuffer into the specified PyTorch tensor.
	 * The CUDA tensor has shape (1, 5, height, width) of type float or double.
	 * The channels are (red, green, blue, alpha, depth in world-space).
	 */
	void copyToCuda(torch::Tensor& output);
};

END_RENDERER_NAMESPACE

namespace kernel
{
	void CopyFramebufferToCuda(
		cudaGraphicsResource_t colorTexture,
		cudaGraphicsResource_t depthTexture,
		Tensor4RW<float> output,
		cudaStream_t stream);
	void CopyFramebufferToCuda(
		cudaGraphicsResource_t colorTexture,
		cudaGraphicsResource_t depthTexture,
		Tensor4RW<double> output,
		cudaStream_t stream);
}

#endif
