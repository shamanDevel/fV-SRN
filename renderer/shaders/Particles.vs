#version 330 core

layout(location = 0) in vec4 position;
layout(location = 1) in vec4 normal;

out vec3 Position;
out vec3 Velocity;
out float Time;

void main()
{
    Position = vec3(position);
    Velocity = vec3(normal);
    Time = normal.w;
    gl_Position = vec4(Position, 1.0);
}