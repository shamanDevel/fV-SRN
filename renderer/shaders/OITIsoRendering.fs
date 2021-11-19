#version 430 core
layout (early_fragment_tests) in;

//Pass 1: write to fragment list

layout( location = 0 ) out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  
  
uniform vec3 lightDir; 
uniform vec3 viewPos; 
uniform vec3 ambientLightColor;
uniform vec3 diffuseLightColor;
uniform vec3 specularLightColor;
uniform int specularExponent;
uniform vec4 objectColor;
uniform bool useShading;

struct NodeType {
  vec4 color;
  float depth;
  uint next;
};

layout( binding = 0, r32ui) uniform uimage2D headPointers;
layout( binding = 0, offset = 0) uniform atomic_uint nextNodeCounter;
layout( binding = 0, std430 ) buffer linkedListsC { vec4 colors[]; };
layout( binding = 1, std430 ) buffer linkedListsD { float depths[]; };
layout( binding = 2, std430 ) buffer linkedListsN { uint nexts[]; };
uniform uint MaxNodes;

void main()
{
    vec4 result;
    if (useShading) {
        // ambient
        vec3 ambient = ambientLightColor * objectColor.rgb;
  	
        // diffuse 
        vec3 norm = normalize(Normal);
        float diff = abs(dot(norm, lightDir));
        vec3 diffuse = diff * diffuseLightColor * objectColor.rgb;
    
        // specular
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 reflectDir = reflect(-lightDir, norm);  
        float spec = pow(clamp(dot(viewDir, reflectDir), 0.0, 1.0), specularExponent);
        vec3 specular = spec * specularLightColor;  
        
        result = vec4(
            ambient + diffuse + specular,
            objectColor.a);
    }
    else
    {
        result = objectColor;
    }    

    // fragment linked list
    uint nodeIdx = atomicCounterIncrement(nextNodeCounter);
    if( nodeIdx < MaxNodes ) {
        uint prevHead = imageAtomicExchange(headPointers, ivec2(gl_FragCoord.xy), nodeIdx);
        colors[nodeIdx] = result;
        depths[nodeIdx] = distance(FragPos, viewPos); //gl_FragCoord.z;
        nexts[nodeIdx] = prevHead;
    }

}
