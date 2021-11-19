#pragma once

#if RENDERER_OPENGL_SUPPORT==1

#include "commons.h"

#include <GL/glew.h>
#include <iostream>

BEGIN_RENDERER_NAMESPACE

class OffscreenContext
{
public:
	static void setup();
	static void teardown();
};

/**
 * Checks for any opengl errors.
 * \param throw_ if true, an exception is throw. Else, only the message is printed
 * \return true on error, false if not
 */
bool checkOpenGLError(bool throw_ = false);

END_RENDERER_NAMESPACE

#endif
