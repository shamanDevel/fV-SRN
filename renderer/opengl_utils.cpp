#include "opengl_utils.h"

#if RENDERER_OPENGL_SUPPORT==1

#include <GLFW/glfw3.h>

namespace
{
	GLFWwindow* offscreenWindow = nullptr;
}
static void glfw_error_callback(int error, const char* description)
{
	fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

void renderer::OffscreenContext::setup()
{
	std::cout << "Setup OpenGL  offscreen context" << std::endl;
	if (offscreenWindow != nullptr)
	{
		std::cerr << "OpenGL offscreen context already created" << std::endl;
		return;
	}

	// Setup window
	glfwSetErrorCallback(glfw_error_callback);
	if (!glfwInit())
		throw std::runtime_error("Unable to initialize GLFW");

	// GL 3.0 + GLSL 130
	const char* glsl_version = "#version 130";
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
#if !defined(NDEBUG)
	glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#endif

	offscreenWindow = glfwCreateWindow(640, 480, "", NULL, NULL);
	if (offscreenWindow == nullptr)
		throw std::runtime_error("Unable to create offscreen window");
	glfwMakeContextCurrent(offscreenWindow);

	if (glewInit() != GLEW_OK)
	{
		std::cerr << "Unable to initialize GLEW" << std::endl;
		teardown();
		throw std::runtime_error("Unable to initialize GLEW");
	}
	std::cout << "OpenGL offscreen context created" << std::endl;
}

void renderer::OffscreenContext::teardown()
{
	if (offscreenWindow == nullptr) {
		std::cout << "OpenGL offscreen context already destroyed or none created" << std::endl;
	    return;
	}

	glfwDestroyWindow(offscreenWindow);
	glfwTerminate();

	offscreenWindow = nullptr;
	std::cout << "OpenGL offscreen context destroyed" << std::endl;
}

bool renderer::checkOpenGLError(bool throw_)
{
    bool hasError = false;
    GLenum err;
    while ((err = glGetError()) != GL_NO_ERROR)
    {
        static const char* EMPTY = "<Empty Message>";
        const char* msg = reinterpret_cast<const char*>(gluErrorString(err));
        if (!msg)
            msg = EMPTY;
        std::cerr << "OpenGL Error: " << msg << " (0x" << std::hex << err << std::dec << ")" << std::endl;
        hasError = true;
        //__debugbreak();
        if (err == GL_INVALID_OPERATION) break;
    }
    if (throw_ && hasError)
        throw std::runtime_error("OpenGL error, see stderr output");
    return !hasError;
}

#endif
