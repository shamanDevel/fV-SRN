#pragma once

#if RENDERER_OPENGL_SUPPORT==1

#include "commons.h"

#include <string>
#include <memory>
#include <GL/glew.h>
#include <glm/glm.hpp>

BEGIN_RENDERER_NAMESPACE

class Shader
{
	GLuint id_ = 0;
	std::string vertexPath, fragmentPath, geometryPath;
	std::string preprocessorDefines;

	Shader(Shader const&) = delete;
	Shader& operator=(Shader const&) = delete;
	
public:
	Shader(const std::string& vertexPath, const std::string& fragmentPath,
		const std::string& geometryPath = "",
		const std::string& preprocessorDefines = "");
	~Shader() { free(); };
	void free();
	void reload();
	void use();

	GLint getUniformLocation(const std::string& name) const;
	void setBool(const std::string& name, bool value) const
	{
		glUniform1i(getUniformLocation(name), (int)value);
	}
	void setInt(const std::string& name, int value) const
	{
		glUniform1i(getUniformLocation(name), value);
	}
	void setUnsignedInt(const std::string& name, unsigned int value) const
	{
		glUniform1ui(getUniformLocation(name), value);
	}
	void setFloat(const std::string& name, float value) const
	{
		glUniform1f(getUniformLocation(name), value);
	}
	void setVec2(const std::string& name, const glm::vec2& value) const
	{
		glUniform2fv(getUniformLocation(name), 1, &value[0]);
	}
	void setVec2(const std::string& name, float x, float y) const
	{
		glUniform2f(getUniformLocation(name), x, y);
	}
	void setVec3(const std::string& name, const glm::vec3& value) const
	{
		glUniform3fv(getUniformLocation(name), 1, &value[0]);
	}
	void setVec3(const std::string& name, float x, float y, float z) const
	{
		glUniform3f(getUniformLocation(name), x, y, z);
	}
	void setVec4(const std::string& name, const glm::vec4& value) const
	{
		glUniform4fv(getUniformLocation(name), 1, &value[0]);
	}
	void setVec4(const std::string& name, float x, float y, float z, float w)
	{
		glUniform4f(getUniformLocation(name), x, y, z, w);
	}
	void setMat2(const std::string& name, const glm::mat2& mat) const
	{
		glUniformMatrix2fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
	}
	void setMat3(const std::string& name, const glm::mat3& mat) const
	{
		glUniformMatrix3fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
	}
	void setMat4(const std::string& name, const glm::mat4& mat) const
	{
		glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &mat[0][0]);
	}
};
typedef std::shared_ptr<Shader> Shader_ptr;

END_RENDERER_NAMESPACE

#endif
