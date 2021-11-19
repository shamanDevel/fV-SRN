#version 420 core

layout(points) in;
layout(triangle_strip, max_vertices=12) out;

in vec3 Position[];
in vec3 Velocity[];
in float Time[];

out vec3 FragPos;
out vec3 Normal;

uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;
uniform mat4 normalMatrix;
uniform float speedMultiplier;
uniform float particleSize;

void main()
{
    float time = Time[0];
    if (time < 0) return;

    vec3 center = Position[0];
    vec3 front = Velocity[0];
    float speed = length(front);
    if (speed < 1e-4)
        front = vec3(1,0,0);
    else
        front = normalize(front);
    float arrowLength = 1 + speed * speedMultiplier;

    vec3 tangent1 = cross(front, vec3(0,1,0));
    if (length(tangent1) < 1e-5)
        tangent1 = cross(front, vec3(0,0,1));
    tangent1 = normalize(tangent1);
    vec3 tangent2 = normalize(cross(tangent1, front));

    const float sqrt4 = 0.35355339059; //sqrt(2)/4
    vec3 worldPositions[] = {
        center + arrowLength * particleSize * front,
        center + 0.5f * particleSize * tangent1,
        center - sqrt4 * particleSize * tangent1 - sqrt4 * particleSize * tangent2,
        center - sqrt4 * particleSize * tangent1 + sqrt4 * particleSize * tangent2,
    };
    vec4 screenPositions[4];
    for (int i=0; i<4; ++i)
        screenPositions[i] = projectionMatrix * viewMatrix * vec4(worldPositions[i],1.0);

    int indices[12] = {
        0,1,2,
        0,2,3,
        0,3,1,
        1,2,3
    };
    vec3 screenNormals[4];
    for (int i=0; i<4; ++i)
        screenNormals[i] = normalize((normalMatrix * vec4(
            cross(worldPositions[indices[3*i+0]]-worldPositions[indices[3*i+1]], worldPositions[indices[3*i+0]]-worldPositions[indices[3*i+2]]), 0.0)).xyz);

    for (int i=0; i<4; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            int idx = indices[3*i+j];
            FragPos = worldPositions[idx];
            Normal = screenNormals[i];
            gl_Position = screenPositions[idx];
            EmitVertex();
        }
        EndPrimitive();
    }
}