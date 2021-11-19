#pragma once

#if RENDERER_OPENGL_SUPPORT==1

#include "commons.h"

#include <GL/glew.h>
#include "opengl_shader.h"

BEGIN_RENDERER_NAMESPACE

/*
 * Helper class for order-independent transparency via fragment linked lists.
 * See shaders/OITIsoRendering.fs for an example on how to write into fragment lists,
 * and shaders/OITIsoBlending.fs for how to sort and blend the fragments.
 */
class OIT
{
	static constexpr int COUNTER_BUFFER = 0;
	static constexpr int LINKED_LIST_BUFFER_COLOR = 1;
	static constexpr int LINKED_LIST_BUFFER_DEPTH = 2;
	static constexpr int LINKED_LIST_BUFFER_NEXT = 3;
	
	int width_ = 0, height_ = 0, numFragments_ = 0;
	GLuint quadBuf_ = 0, quadVao_ = 0;
	GLuint oitBuffers_[4] = {0};
	GLuint oitHeadPtrTex_ = 0;
	GLuint clearBuf_ = 0;

	OIT(OIT const&) = delete;
	OIT& operator=(OIT const&) = delete;

public:
	OIT();
	~OIT();
	void resizeScreen(int width, int height);
	void resizeBuffer(int numFragments);
	int getNumFragments() const { return numFragments_; }

	void start(); //bind the buffers, fragments are now recorded
	void finish(); //unbind the buffers
	void setShaderParams(Shader* shader); //sets the uniforms for the rendering shader
	int blend(Shader* blendingShader); //blend the fragments to the screen, returns number of fragments
	void debug(); //print the fragment lists to console
};

END_RENDERER_NAMESPACE

#endif
