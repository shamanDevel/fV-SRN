#version 330 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;

out vec3 Normal;
out vec3 FragPos;

uniform mat4 modelMatrix;
uniform mat4 normalMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;

void main()
{
    FragPos = vec3(modelMatrix * position);
    Normal = vec3(normalMatrix * normal);
    gl_Position = projectionMatrix * viewMatrix * vec4(FragPos, 1.0);
}