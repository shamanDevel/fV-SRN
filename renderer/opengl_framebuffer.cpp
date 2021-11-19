#include "opengl_framebuffer.h"

#if RENDERER_OPENGL_SUPPORT==1

#include <c10/cuda/CUDAStream.h>

#include "opengl_utils.h"

#include <cuMat/src/Matrix.h>
#include <cuMat/src/Errors.h>

#include "opengl_mesh.h"
#include "renderer_tensor.cuh"
#include "pytorch_utils.h"

BEGIN_RENDERER_NAMESPACE


Framebuffer::Framebuffer(int width, int height)
	: width_(width)
    , height_(height)
    , fbo_(0)
    , colorTexture_(0)
    , depthTexture_(0)
    , depthRbo_(0)
    , prevBinding_(0)
{
	GLint oldRbo, oldFbo;
	glGetIntegerv(GL_RENDERBUFFER_BINDING, &oldRbo);
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, &oldFbo);
	checkOpenGLError();

	glGenFramebuffers(1, &fbo_);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);
	checkOpenGLError();

	glGenTextures(1, &colorTexture_);
	glBindTexture(GL_TEXTURE_2D, colorTexture_);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, colorTexture_, 0);
	checkOpenGLError();
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(&colorTextureCuda_, colorTexture_, GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsReadOnly));

	glGenTextures(1, &depthTexture_);
	glBindTexture(GL_TEXTURE_2D, depthTexture_);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, depthTexture_, 0);
	checkOpenGLError();
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(&depthTextureCuda_, depthTexture_, GL_TEXTURE_2D,
		cudaGraphicsRegisterFlagsReadOnly));

	glGenRenderbuffers(1, &depthRbo_);
	glBindRenderbuffer(GL_RENDERBUFFER, depthRbo_);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
	glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthRbo_);
	checkOpenGLError();

	auto status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	switch (status)
	{
	case GL_FRAMEBUFFER_COMPLETE:
		std::cout << "Framebuffer created successfully" << std::endl;
		break;
	case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
		std::cerr << "Framebuffer incomplete: Not all framebuffer attachment points are framebuffer attachment complete" << std::endl;
		throw std::runtime_error("Incomplete framebuffer");
	case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS_EXT:
		std::cerr << "Framebuffer incomplete: Not all attached images have the same width and height" << std::endl;
		throw std::runtime_error("Incomplete framebuffer");
	case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
		std::cerr << "Framebuffer incomplete: No images are attached to the framebuffer" << std::endl;
		throw std::runtime_error("Incomplete framebuffer");
	case GL_FRAMEBUFFER_UNSUPPORTED:
		std::cerr << "Framebuffer incomplete: The combination of internal formats of the attached images violates an implementation-dependent set of restrictions" << std::endl;
		throw std::runtime_error("Incomplete framebuffer");
	default:
		std::cerr << "Framebuffer incomplete: Unknown reason" << std::endl;
		throw std::runtime_error("Incomplete framebuffer");
	}

	glBindRenderbuffer(GL_RENDERBUFFER, oldRbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, oldFbo);
	checkOpenGLError();
}

Framebuffer::~Framebuffer()
{
	try {
		CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(colorTextureCuda_));
		CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(depthTextureCuda_));
		glDeleteFramebuffers(1, &fbo_);
		glDeleteRenderbuffers(1, &depthRbo_);
		glDeleteTextures(1, &depthTexture_);
		glDeleteTextures(1, &colorTexture_);
		checkOpenGLError();
	} catch (const cuMat::cuda_error& ex)
	{
		std::cerr << "Error on deconstructing Framebuffer: " << ex.what() << std::endl;
	}
}

void Framebuffer::bind()
{
	glGetIntegerv(GL_FRAMEBUFFER_BINDING, reinterpret_cast<int*>(&prevBinding_));
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo_);
	checkOpenGLError();
	glGetIntegerv(GL_VIEWPORT, prevViewport_);
	glViewport(0, 0, width_, height_);
	GLenum buffers[] = { GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
	glDrawBuffers(2, buffers);
	checkOpenGLError();
}

void Framebuffer::unbind()
{
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, prevBinding_);
	checkOpenGLError();
	glViewport(prevViewport_[0], prevViewport_[1], prevViewport_[2], prevViewport_[3]);
	checkOpenGLError();
	if (prevBinding_ > 0) {
		GLenum buffers[] = { GL_COLOR_ATTACHMENT0 };
		glDrawBuffers(1, buffers);
	}
	checkOpenGLError();
}

void Framebuffer::readRGBA(std::vector<float>& data)
{
	glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo_);
	data.resize(width_ * height_ * 4);
	glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, width_, height_, GL_RGBA, GL_FLOAT, data.data());
	checkOpenGLError();
	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
}

void Framebuffer::copyToCuda(torch::Tensor& output)
{
	CHECK_DIM(output, 4);
	CHECK_SIZE(output, 0, 1);
	CHECK_SIZE(output, 1, 5);
	CHECK_SIZE(output, 2, height_);
	CHECK_SIZE(output, 3, width_);
	CHECK_CUDA(output, true);

	CUstream stream = c10::cuda::getCurrentCUDAStream();
	c10::ScalarType scalarType = output.scalar_type();
	RENDERER_DISPATCH_FLOATING_TYPES(scalarType, "Framebuffer::copyToCuda", [&]()
		{
			const auto acc = accessor< ::kernel::Tensor4RW<scalar_t>>(output);
			kernel::CopyFramebufferToCuda(colorTextureCuda_, depthTextureCuda_, acc, stream);
			return true;
		});
}

END_RENDERER_NAMESPACE

#if 0

texture<float4, cudaTextureType2D, cudaReadModeElementType> framebufferTexRef;

namespace
{
	__global__ void FramebufferCopyToCudaIso(dim3 virtual_size,
		kernel::OutputTensor output)
	{
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size)
		{
			float4 rgba = tex2D(framebufferTexRef, i, j);
			float3 normal = make_float3(rgba) * 2 - 1;
			float mask = rgba.w;
			output.coeff(j, i, 0) = mask;
			output.coeff(j, i, 1) = normal.x;
			output.coeff(j, i, 2) = normal.y;
			output.coeff(j, i, 3) = normal.z;
#pragma unroll
			output.coeff(j, i, 4) = 0.0f; //depth
			output.coeff(j, i, 5) = 1.0f; //ao
			output.coeff(j, i, 6) = 0.0f; //flow x
			output.coeff(j, i, 7) = 0.0f; //flow y
		}
		CUMAT_KERNEL_2D_LOOP_END
	}

	__global__ void FramebufferCopyToCudaDvr(dim3 virtual_size,
		kernel::OutputTensor output)
	{
		CUMAT_KERNEL_2D_LOOP(i, j, virtual_size)
		{
			float4 rgba = tex2D(framebufferTexRef, i, j);
			output.coeff(j, i, 0) = rgba.x;
			output.coeff(j, i, 1) = rgba.y;
			output.coeff(j, i, 2) = rgba.z;
			output.coeff(j, i, 3) = rgba.w;
#pragma unroll
			for (int b = 4; b < 10; ++b)
				output.coeff(j, i, b) = 0.0f;
		}
		CUMAT_KERNEL_2D_LOOP_END
	}
}

void renderer::Framebuffer::copyToCudaIso(kernel::OutputTensor& output, cudaStream_t stream)
{
#if 1
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	
	assert(output.cols == width_);
	assert(output.rows == height_);
	cuMat::Context& ctx = cuMat::Context::current();

	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &colorTextureCuda_, stream));
	cudaArray_t array;
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, colorTextureCuda_, 0, 0));
	CUMAT_SAFE_CALL(cudaBindTextureToArray(framebufferTexRef, array));
	
	//copy kernel
	const auto cfg = ctx.createLaunchConfig2D(width_, height_, FramebufferCopyToCudaIso);
	FramebufferCopyToCudaIso
		<<< cfg.block_count, cfg.thread_per_block, 0, stream >>>
		(cfg.virtual_size, output);
	//CUMAT_CHECK_ERROR();
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	//CUMAT_SAFE_CALL(cudaDestroyTextureObject(tex));
	CUMAT_SAFE_CALL(cudaUnbindTexture(framebufferTexRef));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &colorTextureCuda_, stream));
#else
	//CPU-Path for debugging
	std::vector<float> rgba;
	readRGBA(rgba);
	const float4* rgba_v = reinterpret_cast<const float4*>(rgba.data());
	std::vector<float> hostMemory_v(output.rows * output.cols * output.batches);
	float* hostMemory = hostMemory_v.data();
	for (int x=0; x<output.cols; ++x) for (int y=0; y<output.rows; ++y)
	{
		float4 in = rgba_v[x + output.cols * y];
		float3 normal = safeNormalize(make_float3(in) * 2 - 1);
		float mask = in.w;
		hostMemory[output.idx(y, x, 0)] = mask;
		hostMemory[output.idx(y, x, 1)] = normal.x;
		hostMemory[output.idx(y, x, 2)] = normal.y;
		hostMemory[output.idx(y, x, 3)] = normal.z;
	}
	CUMAT_SAFE_CALL(cudaMemcpy(output.memory, hostMemory,
		sizeof(float) * hostMemory_v.size(), cudaMemcpyHostToDevice));
#endif
}

#endif

#endif
