#include "opengl_shader.h"
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

#ifndef SHADER_RUNTIME_COMPILATION
#define SHADER_RUNTIME_COMPILATION 0
#endif

#if SHADER_RUNTIME_COMPILATION==0
#include <cmrc/cmrc.hpp>
CMRC_DECLARE(shaders);
#endif

BEGIN_RENDERER_NAMESPACE

renderer::Shader::Shader(const std::string& vertexPath, 
	const std::string& fragmentPath, const std::string& geometryPath,
	const std::string& preprocessorDefines)
	: vertexPath(vertexPath)
    , fragmentPath(fragmentPath)
    , geometryPath(geometryPath)
	, preprocessorDefines(preprocessorDefines)
{
	reload();
}

void Shader::free()
{
	if (id_)
	{
		glDeleteProgram(id_);
		id_ = 0;
		checkOpenGLError();
	}
}

static void checkCompileErrors(GLuint shader, std::string type)
{
	GLint success;
	GLchar infoLog[1024];
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			glGetShaderInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			glGetProgramInfoLog(shader, 1024, NULL, infoLog);
			std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << "\n" << infoLog << "\n -- --------------------------------------------------- -- " << std::endl;
		}
	}
}
void Shader::reload()
{
	free();
	bool hasGeometryShader = !geometryPath.empty();

	// 1. retrieve the vertex/fragment source code from filePath
	std::string vertexCode;
	std::string fragmentCode;
	std::string geometryCode;
#if SHADER_RUNTIME_COMPILATION==1
	//read from file
	std::ifstream vShaderFile;
	std::ifstream fShaderFile;
	std::ifstream gShaderFile;
	// ensure ifstream objects can throw exceptions:
	vShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	gShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	try
	{
		// open files
		vShaderFile.open(CUMAT_STR(RENDERER_SHADER_DIR) "/shaders/" + vertexPath);
		fShaderFile.open(CUMAT_STR(RENDERER_SHADER_DIR) "/shaders/" + fragmentPath);
		std::stringstream vShaderStream, fShaderStream;
		// read file's buffer contents into streams
		vShaderStream << vShaderFile.rdbuf();
		fShaderStream << fShaderFile.rdbuf();
		// close file handlers
		vShaderFile.close();
		fShaderFile.close();
		// convert stream into string
		vertexCode = vShaderStream.str();
		fragmentCode = fShaderStream.str();
		//repeat for geometry shader
		if (hasGeometryShader)
		{
			gShaderFile.open(CUMAT_STR(RENDERER_SHADER_DIR) "/shaders/" + geometryPath);
			std::stringstream gShaderStream;
			gShaderStream << gShaderFile.rdbuf();
			gShaderFile.close();
			geometryCode = gShaderStream.str();
		}
	}
	catch (std::ifstream::failure& e)
	{
		std::cerr << "Shader file not loaded successfully: " << e.what() << std::endl;
	}
#else
	//read from resources
	auto fs = cmrc::shaders::get_filesystem();
	auto vShaderFile = fs.open("shaders/" + vertexPath);
	vertexCode.resize(vShaderFile.size());
	memcpy(vertexCode.data(), vShaderFile.begin(), vShaderFile.size());
	auto fShaderFile = fs.open("shaders/" + fragmentPath);
	fragmentCode.resize(fShaderFile.size());
	memcpy(fragmentCode.data(), fShaderFile.begin(), fShaderFile.size());
	if (hasGeometryShader)
	{
		auto gShaderFile = fs.open("shaders/" + geometryPath);
		geometryCode.resize(gShaderFile.size());
		memcpy(geometryCode.data(), gShaderFile.begin(), gShaderFile.size());
	}
#endif

	const auto insertStr = [this](const std::string& str)
	{
		auto i1 = str.find("#version");
		auto i2 = str.find("\n", i1 + 1);
		return str.substr(0, i2 + 1) + preprocessorDefines + str.substr(i2);
	};
	vertexCode = insertStr(vertexCode);
	fragmentCode = insertStr(fragmentCode);
	if (hasGeometryShader)
		geometryCode = insertStr(geometryCode);

	// 2. compile shaders
	unsigned int vertex, fragment, geometry = 0;
	// vertex shader
	const char* vShaderCode = vertexCode.c_str();
	vertex = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertex, 1, &vShaderCode, NULL);
	glCompileShader(vertex);
	checkCompileErrors(vertex, "VERTEX");
	// fragment Shader
	const char* fShaderCode = fragmentCode.c_str();
	fragment = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragment, 1, &fShaderCode, NULL);
	glCompileShader(fragment);
	checkCompileErrors(fragment, "FRAGMENT");
	// geometry shader
	if (hasGeometryShader)
	{
		const char* gShaderCode = geometryCode.c_str();
		geometry = glCreateShader(GL_GEOMETRY_SHADER);
		glShaderSource(geometry, 1, &gShaderCode, NULL);
		glCompileShader(geometry);
		checkCompileErrors(geometry, "GEOMETRY");
	}
	// shader Program
	id_ = glCreateProgram();
	glAttachShader(id_, vertex);
	glAttachShader(id_, fragment);
	if (hasGeometryShader) glAttachShader(id_, geometry);
	glLinkProgram(id_);
	checkCompileErrors(id_, "PROGRAM");
	// delete the shaders as they're linked into our program now and no longer necessery
	glDeleteShader(vertex);
	glDeleteShader(fragment);
	if (hasGeometryShader) glDeleteShader(geometry);

	std::cout << "Shaders compiled" << std::endl;
}

void Shader::use()
{
	glUseProgram(id_);
	checkOpenGLError();
}

GLint Shader::getUniformLocation(const std::string& name) const
{
	GLint loc = glGetUniformLocation(id_, name.c_str());
	if (loc < 0)
		std::cerr << "No uniform with name \"" << name << "\" found" << std::endl;
	return loc;
}

END_RENDERER_NAMESPACE

#endif
