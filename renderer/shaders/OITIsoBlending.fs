#version 430 core
layout (early_fragment_tests) in;

#ifndef MAX_FRAGMENTS
#define MAX_FRAGMENTS 64
#endif

#define NO_SORT 0
#define INSERTION_SORT 1
#define SELECTION_SORT 2
#define BUBBLE_SORT 3
#define SORTING_ALGORITHM 1
  
struct NodeType {
  vec4 color;
  float depth;
};

layout( binding = 0, r32ui) uniform uimage2D headPointers;
layout( binding = 0, offset = 0) uniform atomic_uint nextNodeCounter;
layout( binding = 0, std430 ) buffer linkedListsC { vec4 colors[]; };
layout( binding = 1, std430 ) buffer linkedListsD { float depths[]; };
layout( binding = 2, std430 ) buffer linkedListsN { uint nexts[]; };

layout( location = 0 ) out vec4 FragColor;

vec3 xyzToRgb(vec3 xyz)
{
	float x = xyz.x / 100.0f;
	float y = xyz.y / 100.0f;
	float z = xyz.z / 100.0f;

	float r = x * 3.2404542f + y * -1.5371385f + z * -0.4985314f;
	float g = x * -0.9692660f + y * 1.8760108f + z * 0.0415560f;
	float b = x * 0.0556434f + y * -0.2040259f + z * 1.0572252f;

	r = (r > 0.0031308f) ? (1.055f * pow(r, 1.0f / 2.4f) - 0.055f) : (12.92f * r);
	g = (g > 0.0031308f) ? (1.055f * pow(g, 1.0f / 2.4f) - 0.055f) : (12.92f * g);
	b = (b > 0.0031308f) ? (1.055f * pow(b, 1.0f / 2.4f) - 0.055f) : (12.92f * b);

	return vec3(r, g, b);
}

void main()
{
    NodeType frags[MAX_FRAGMENTS];

    int count = 0;

    // Get the index of the head of the list
    uint n = imageLoad(headPointers, ivec2(gl_FragCoord.xy)).r;

    // Copy the linked list for this fragment into an array
    while( n != 0xffffffffu && count < MAX_FRAGMENTS) {
        frags[count].color = colors[n];
        frags[count].depth = depths[n];
        n = nexts[n];
        count++;
    }

#if SORTING_ALGORITHM == INSERTION_SORT
    for( uint i = 1; i < count; i++ )
    {
        NodeType toInsert = frags[i];
        uint j = i;
        while( j > 0 && toInsert.depth > frags[j-1].depth ) {
            frags[j] = frags[j-1];
            j--;
        }
        frags[j] = toInsert;
    }
#elif SORTING_ALGORITHM == SELECTION_SORT
    int max;
    NodeType tempNode;
    int j, i;
    for(j = 0; j < count - 1; j++)
    {
        max = j;
        for(  i = j + 1; i < count; i++)
        {   
            if(frags[i].depth > frags[max].depth)
            {
                max = i;
            }
        }
        if(max != j)
        {
            tempNode = frags[j];
            frags[j] = frags[max]; 
            frags[max] = tempNode;
        }
    }
#elif SORTING_ALGORITHM == BUBBLE_SORT
    int j, i;
    NodeType tempNode;
    for(i = 0; i < count - 1; i++)
    {
        for(j = 0; j < count - i - 1; j++)
        {
            if(frags[j].depth < frags[j+1].depth)
            {
                tempNode = frags[j];
                frags[j] = frags[j+1];
                frags[j+1] = tempNode;
            }
        }
    }
#endif

    // Traverse the array, and combine the colors using the alpha
    // channel. Back-to-front
    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    for( int i = 0; i < count; i++ )
    {
        float alpha = min(1.0, frags[i].color.a);
        color.rgb = mix( color.rgb, frags[i].color.rgb, alpha);
        color.a = color.a * (1 - alpha) + alpha;
    }
    color.rgb = xyzToRgb(color.rgb);
    color = clamp(color, 0.0, 1.0);

    // Output the final color
    FragColor = color;
}
