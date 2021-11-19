#version 330 core
out vec4 FragColor[2];

in vec3 Normal;  
in vec3 FragPos;  

uniform vec3 ambientColor;
uniform vec3 diffuseColor;
uniform vec3 lightDirection;
uniform vec3 cameraOrigin;
  
void main()
{
    vec3 norm = normalize(Normal);

    float diffuse = clamp(abs(dot(norm, lightDirection)), 0, 1); //two-sided

    vec3 result = ambientColor + diffuse * diffuseColor;
    FragColor[0] = vec4(result, 1.0);
    float d = distance(cameraOrigin, FragPos);
    FragColor[1] = vec4(d,d,d,1.0);
}
