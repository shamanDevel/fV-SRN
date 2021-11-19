#version 330 core
out vec4 FragColor[2];

in vec3 Normal;  
in vec3 FragPos;  

uniform vec3 cameraOrigin;
  
void main()
{
    vec3 norm = normalize(Normal);
    vec3 result = norm * 0.5 + 0.5;
    FragColor[0] = vec4(result, 1.0);

    float d = distance(cameraOrigin, FragPos);
    FragColor[1] = vec4(d,d,d,1.0);
}
