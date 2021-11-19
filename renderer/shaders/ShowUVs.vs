#version 330 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;

out vec3 Normal;
out vec3 FragPos;

uniform mat4 model;
uniform mat4 transpInvModel;
uniform mat4 view;
uniform mat4 projection;

void main()
{
    FragPos = vec3(model * position);
    Normal = mat3(transpInvModel) * vec3(normal);
    gl_Position = projection * view * vec4(FragPos, 1.0);
}