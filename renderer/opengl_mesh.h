#pragma once

#if RENDERER_OPENGL_SUPPORT==1

#include "commons.h"

#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>

#include "volume.h"

BEGIN_RENDERER_NAMESPACE

struct Vertex
{
	float4 position;
	float4 normals;
};

struct MeshCpu
{
	std::vector<Vertex> vertices;
	std::vector<GLuint> indices;

	static MeshCpu createSphere(int zSamples=16, int radialSamples=32, float radius=1, bool useEvenSlices=false);
	static MeshCpu createCube();
	//A wireframe cube that should be rendered with GL_LINES and index buffer.
	//Does not contain normals
	static MeshCpu createWireCube();
};

/**
 * Stores a single mesh.
 * Each vertex is represented by the structure "Vertex",
 * positions are bound to attribute index 0 and normals
 * to attribute index 1.
 */
class Mesh
{
	GLuint vbo = 0, vao = 0, ibo = 0;
	int numVertices = 0, numIndices = 0;
	int availableVertices = 0, availableIndices = 0;
	cudaGraphicsResource_t vboCuda = 0, iboCuda = 0;

	Mesh(Mesh const&) = delete;
	Mesh& operator=(Mesh const&) = delete;
	
public:
	static constexpr int POSITION_INDEX = 0;
	static constexpr int NORMAL_INDEX = 1;

	Mesh() = default;
	explicit Mesh(const MeshCpu& meshCpu) { copyFromCpu(meshCpu); }
	~Mesh();
	void free();
	//resize to exactly the given size
	void resize(int vertices, int indices);
	//reserves the given size, i.e. resize only if smaller
	void reserve(int vertices, int indices);
	//Draw with index buffer
	void drawIndexed(GLenum mode = GL_TRIANGLES);
	//Draw directly, without index buffer
	//mode: GL_POINTS, GL_LINES, GL_TRIANGLES, ...
	void drawArrays(GLenum mode);

	class Mapping
	{
		cudaGraphicsResource_t vboCuda_, iboCuda_;
		Vertex* vertices_;
		GLuint* indices_;
	private:
		Mapping(cudaGraphicsResource_t vboCuda, cudaGraphicsResource_t iboCuda);
		friend class Mesh;
	public:
		~Mapping();
		Vertex* vertices() { return vertices_; }
		GLuint* indices() { return indices_; }
	};
	std::unique_ptr<Mapping> cudaMap();

	void copyToCpu(MeshCpu& meshCpu);
	void copyFromCpu(const MeshCpu& meshCpu);

	bool isValid() const { return vbo != 0; }
	int getNumVertices() const { return numVertices; }
	int getNumIndices() const { return numIndices; }
	int getAvailableVertices() const { return availableVertices; }
	int getAvailableIndices() const { return availableIndices; }
	void setNumVertices(int num);
	void setNumIndices(int num);
};
typedef std::shared_ptr<Mesh> Mesh_ptr;

END_RENDERER_NAMESPACE

#endif
