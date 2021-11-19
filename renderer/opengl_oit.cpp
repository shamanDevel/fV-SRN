#include "opengl_oit.h"
#include "opengl_utils.h"

#if RENDERER_OPENGL_SUPPORT==1

#include <fstream>
#include <iostream>
#include <sstream>
#include <cuMat/src/Errors.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <tinyformat.h>

#include "camera.h"

//#include <cmrc/cmrc.hpp>
//CMRC_DECLARE(shaders);

BEGIN_RENDERER_NAMESPACE

OIT::OIT()
{
	//create quad buffer
	GLfloat verts[] = { -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f,
	1.0f, 1.0f, 0.0f, -1.0f, 1.0f, 0.0f };
	glGenBuffers(1, &quadBuf_);
	glBindBuffer(GL_ARRAY_BUFFER, quadBuf_);
	glBufferData(GL_ARRAY_BUFFER, 4 * 3 * sizeof(GLfloat), verts, GL_STATIC_DRAW);
	checkOpenGLError();

	// Set up the vertex array object
	glGenVertexArrays(1, &quadVao_);
	glBindVertexArray(quadVao_);
	checkOpenGLError();

	glBindBuffer(GL_ARRAY_BUFFER, quadBuf_);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);  // Vertex position
	checkOpenGLError();

	glBindVertexArray(0);
	checkOpenGLError();

	// The buffer for the head pointers and fragment lists
	glGenTextures(1, &oitHeadPtrTex_);
	glGenBuffers(4, oitBuffers_);
	glGenBuffers(1, &clearBuf_);

	// Our atomic counter
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, oitBuffers_[COUNTER_BUFFER]);
	glBufferData(GL_ATOMIC_COUNTER_BUFFER, sizeof(GLuint), NULL, GL_DYNAMIC_DRAW);
	checkOpenGLError();
}

OIT::~OIT()
{
	glDeleteVertexArrays(1, &quadVao_);
	glDeleteBuffers(1, &quadBuf_);

	glDeleteTextures(1, &oitHeadPtrTex_);
	glDeleteBuffers(4, oitBuffers_);
	glDeleteBuffers(1, &clearBuf_);
	checkOpenGLError();
}

void OIT::resizeScreen(int width, int height)
{
	width_ = width;
	height_ = height;

	glDeleteTextures(1, &oitHeadPtrTex_);
	glDeleteBuffers(1, &clearBuf_);
	glGenTextures(1, &oitHeadPtrTex_);
	glGenBuffers(1, &clearBuf_);
	checkOpenGLError();
	
	// The buffer for the head pointers, as an image texture
	glBindTexture(GL_TEXTURE_2D, oitHeadPtrTex_);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32UI, width, height);
	glBindImageTexture(0, oitHeadPtrTex_, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32UI);
	glBindTexture(GL_TEXTURE_2D, 0);
	checkOpenGLError();

	std::vector<GLuint> headPtrClearBuf(width * height, 0xffffffff);
	glGenBuffers(1, &clearBuf_);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, clearBuf_);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, headPtrClearBuf.size() * sizeof(GLuint),
		&headPtrClearBuf[0], GL_STATIC_COPY);
}

void OIT::resizeBuffer(int numFragments)
{
	// The buffer of linked lists
	numFragments_ = numFragments;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, oitBuffers_[LINKED_LIST_BUFFER_COLOR]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numFragments * 4*sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, oitBuffers_[LINKED_LIST_BUFFER_DEPTH]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numFragments * sizeof(float), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, oitBuffers_[LINKED_LIST_BUFFER_NEXT]);
	glBufferData(GL_SHADER_STORAGE_BUFFER, numFragments * sizeof(int), NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	checkOpenGLError();
}

void OIT::start()
{
	//clear buffers
	GLuint zero = 0;
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, oitBuffers_[COUNTER_BUFFER]);
	glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &zero);
	checkOpenGLError();

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, clearBuf_);
	glBindTexture(GL_TEXTURE_2D, oitHeadPtrTex_);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_, GL_RED_INTEGER,
		GL_UNSIGNED_INT, nullptr);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	checkOpenGLError();
	glBindTexture(GL_TEXTURE_2D, oitHeadPtrTex_);
	
	//bind further properties
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, oitBuffers_[LINKED_LIST_BUFFER_COLOR]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, oitBuffers_[LINKED_LIST_BUFFER_DEPTH]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, oitBuffers_[LINKED_LIST_BUFFER_NEXT]);
	checkOpenGLError();
	
}

void OIT::finish()
{
	glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
	checkOpenGLError();
}

void OIT::setShaderParams(Shader* shader)
{
	shader->setUnsignedInt("MaxNodes", numFragments_);
	checkOpenGLError();
}

int OIT::blend(Shader* blendingShader)
{
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	checkOpenGLError();

	//read number of fragments
	//TODO: find a way to do this asynchronously while the blending shader is running
	GLuint fragmentsRendered = 0;
	glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &fragmentsRendered);
	checkOpenGLError();
	if (fragmentsRendered > numFragments_)
		std::cout << "OIT Overflow! " << fragmentsRendered << " fragments rendered, but only "
		<< numFragments_ << " are available" << std::endl;
	//else
	//	std::cout << "OIT: " << fragmentsRendered << " fragments rendered of "
	//	<< numFragments_ << " available fragments" << std::endl;

	blendingShader->use();
	// Draw a screen filler
	glBindVertexArray(quadVao_);
	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	checkOpenGLError();
	glBindVertexArray(0);

	return fragmentsRendered;
}

void OIT::debug()
{
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	checkOpenGLError();

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0);
	checkOpenGLError();

	//read number of fragments
	GLuint fragmentsRendered = 0;
	glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, sizeof(GLuint), &fragmentsRendered);
	checkOpenGLError();

	//read head pointer texture
	std::vector<GLuint> headPtrBuf(width_ * height_);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RED_INTEGER, GL_UNSIGNED_INT, headPtrBuf.data());
	checkOpenGLError();
	
	//read fragment list buffer
	std::vector<float4> listBufColor(fragmentsRendered);
	std::vector<float> listBufDepth(fragmentsRendered);
	std::vector<unsigned int> listBufNext(fragmentsRendered);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, oitBuffers_[LINKED_LIST_BUFFER_COLOR]);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float4) * fragmentsRendered, listBufColor.data());
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, oitBuffers_[LINKED_LIST_BUFFER_DEPTH]);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) * fragmentsRendered, listBufDepth.data());
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, oitBuffers_[LINKED_LIST_BUFFER_NEXT]);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(int) * fragmentsRendered, listBufNext.data());
	checkOpenGLError();

	//print to console
	for (int x=0; x<width_; ++x) for (int y=0; y<height_; ++y)
	{
		unsigned int node = headPtrBuf[x + width_ * y];
		if (node == 0xffffffffu) continue;
		printf("(%04d, %04d): head=%d\n", x, y, int(node));
		while (node != 0xffffffffu)
		{
			float4 c = listBufColor[node];
			float d = listBufDepth[node];
			unsigned int n = listBufNext[node];
			node = n;
			printf("  rgba=(%.2f, %.2f, %.2f, %.2f), depth=%.5f\n",
				c.x, c.y, c.z, c.w, d);
		}
	}
}


END_RENDERER_NAMESPACE

#endif
