#include "opengl_mesh.h"
#include "opengl_utils.h"

#if RENDERER_OPENGL_SUPPORT==1

#define _USE_MATH_DEFINES
#include <math.h>

#include <cuMat/src/Errors.h>
#include "helper_math.cuh"

BEGIN_RENDERER_NAMESPACE
MeshCpu MeshCpu::createSphere(int zSamples, int radialSamples, float radius, bool useEvenSlices)
{
	MeshCpu data;

    //TODO: properly port from
    //https://github.com/jMonkeyEngine/jmonkeyengine/blob/master/jme3-core/src/main/java/com/jme3/scene/shape/Sphere.java
#if 0

	int vertCount = (zSamples - 2) * (radialSamples + 1) + 2;
	data.vertices.resize(vertCount);

    // generate geometry
    float fInvRS = 1.0f / radialSamples;
    float fZFactor = 2.0f / (zSamples - 1);

    // Generate points on the unit circle to be used in computing the mesh
    // points on a sphere slice.
    std::vector<float> afSin(radialSamples + 1);
    std::vector<float> afCos(radialSamples + 1);
    for (int iR = 0; iR < radialSamples; iR++) {
        float fAngle = 2*M_PI * fInvRS * iR;
        afCos[iR] = std::cos(fAngle);
        afSin[iR] = std::sin(fAngle);
    }
    afSin[radialSamples] = afSin[0];
    afCos[radialSamples] = afCos[0];

    float4 tempVa;
    float4 tempVb;
    float4 tempVc;

    // generate the sphere itself
    int i = 0;
    for (int iZ = 1; iZ < (zSamples - 1); iZ++) {
        float fAFraction = M_PI_2 * (-1.0f + fZFactor * iZ); // in (-pi/2, pi/2)
        float fZFraction;
        if (useEvenSlices) {
            fZFraction = -1.0f + fZFactor * iZ; // in (-1, 1)
        }
        else {
            fZFraction = std::sin(fAFraction); // in (-1,1)
        }
        float fZ = radius * fZFraction;

        // compute center of slice
        float4 kSliceCenter = make_float4(0, 0, fZ, 1);

        // compute radius of slice
        float fSliceRadius = std::sqrt(std::abs(radius * radius - fZ * fZ));

        // compute slice vertices with duplication at end point
        Vector3f kNormal;
        int iSave = i;
        for (int iR = 0; iR < radialSamples; iR++) {
            float fRadialFraction = iR * fInvRS; // in [0,1)
            Vector3f kRadial = tempVc.set(afCos[iR], afSin[iR], 0);
            kRadial.mult(fSliceRadius, tempVa);
            posBuf.put(kSliceCenter.x + tempVa.x).put(
                kSliceCenter.y + tempVa.y).put(
                    kSliceCenter.z + tempVa.z);

            BufferUtils.populateFromBuffer(tempVa, posBuf, i);
            kNormal = tempVa;
            kNormal.normalizeLocal();
            if (!interior) // allow interior texture vs. exterior
            {
                normBuf.put(kNormal.x).put(kNormal.y).put(
                    kNormal.z);
            }
            else {
                normBuf.put(-kNormal.x).put(-kNormal.y).put(
                    -kNormal.z);
            }

            if (textureMode == TextureMode.Original) {
                texBuf.put(fRadialFraction).put(
                    0.5f * (fZFraction + 1.0f));
            }
            else if (textureMode == TextureMode.Projected) {
                texBuf.put(fRadialFraction).put(
                    FastMath.INV_PI
                    * (FastMath.HALF_PI + FastMath.asin(fZFraction)));
            }
            else if (textureMode == TextureMode.Polar) {
                float r = (FastMath.HALF_PI - FastMath.abs(fAFraction)) / FastMath.PI;
                float u = r * afCos[iR] + 0.5f;
                float v = r * afSin[iR] + 0.5f;
                texBuf.put(u).put(v);
            }

            i++;
        }

        BufferUtils.copyInternalVector3(posBuf, iSave, i);
        BufferUtils.copyInternalVector3(normBuf, iSave, i);

        if (textureMode == TextureMode.Original) {
            texBuf.put(1.0f).put(
                0.5f * (fZFraction + 1.0f));
        }
        else if (textureMode == TextureMode.Projected) {
            texBuf.put(1.0f).put(
                FastMath.INV_PI
                * (FastMath.HALF_PI + FastMath.asin(fZFraction)));
        }
        else if (textureMode == TextureMode.Polar) {
            float r = (FastMath.HALF_PI - FastMath.abs(fAFraction)) / FastMath.PI;
            texBuf.put(r + 0.5f).put(0.5f);
        }

        i++;
    }

    vars.release();

    // south pole
    posBuf.position(i * 3);
    posBuf.put(0f).put(0f).put(-radius);

    normBuf.position(i * 3);
    if (!interior) {
        normBuf.put(0).put(0).put(-1); // allow for inner
    } // texture orientation
    // later.
    else {
        normBuf.put(0).put(0).put(1);
    }

    texBuf.position(i * 2);

    if (textureMode == TextureMode.Polar) {
        texBuf.put(0.5f).put(0.5f);
    }
    else {
        texBuf.put(0.5f).put(0.0f);
    }

    i++;

    // north pole
    posBuf.put(0).put(0).put(radius);

    if (!interior) {
        normBuf.put(0).put(0).put(1);
    }
    else {
        normBuf.put(0).put(0).put(-1);
    }

    if (textureMode == TextureMode.Polar) {
        texBuf.put(0.5f).put(0.5f);
    }
    else {
        texBuf.put(0.5f).put(1.0f);
    }

    // allocate connectivity
    triCount = 2 * (zSamples - 2) * radialSamples;
    ShortBuffer idxBuf = BufferUtils.createShortBuffer(3 * triCount);
    setBuffer(Type.Index, 3, idxBuf);

    // generate connectivity
    int index = 0;
    for (int iZ = 0, iZStart = 0; iZ < (zSamples - 3); iZ++) {
        int i0 = iZStart;
        int i1 = i0 + 1;
        iZStart += (radialSamples + 1);
        int i2 = iZStart;
        int i3 = i2 + 1;
        for (int i = 0; i < radialSamples; i++, index += 6) {
            if (!interior) {
                idxBuf.put((short)i0++);
                idxBuf.put((short)i1);
                idxBuf.put((short)i2);
                idxBuf.put((short)i1++);
                idxBuf.put((short)i3++);
                idxBuf.put((short)i2++);
            }
            else { // inside view
                idxBuf.put((short)i0++);
                idxBuf.put((short)i2);
                idxBuf.put((short)i1);
                idxBuf.put((short)i1++);
                idxBuf.put((short)i2++);
                idxBuf.put((short)i3++);
            }
        }
    }

    // south pole triangles
    for (int i = 0; i < radialSamples; i++, index += 3) {
        if (!interior) {
            idxBuf.put((short)i);
            idxBuf.put((short)(vertCount - 2));
            idxBuf.put((short)(i + 1));
        }
        else { // inside view
            idxBuf.put((short)i);
            idxBuf.put((short)(i + 1));
            idxBuf.put((short)(vertCount - 2));
        }
    }

    // north pole triangles
    int iOffset = (zSamples - 3) * (radialSamples + 1);
    for (int i = 0; i < radialSamples; i++, index += 3) {
        if (!interior) {
            idxBuf.put((short)(i + iOffset));
            idxBuf.put((short)(i + 1 + iOffset));
            idxBuf.put((short)(vertCount - 1));
        }
        else { // inside view
            idxBuf.put((short)(i + iOffset));
            idxBuf.put((short)(vertCount - 1));
            idxBuf.put((short)(i + 1 + iOffset));
        }
    }

#endif

    return data;
}

MeshCpu MeshCpu::createCube()
{
    MeshCpu data;

    const auto addFace = [&data](float3 normal, float3 left)
    {
        int startIdx = static_cast<int>(data.vertices.size());
        float3 up = cross(normal, left);
        data.vertices.push_back({
            make_float4(0.5f * normal - 0.5f * left - 0.5f * up, 1),
            make_float4(normal, 0) });
        data.vertices.push_back({
            make_float4(0.5f * normal + 0.5f * left - 0.5f * up, 1),
            make_float4(normal, 0) });
        data.vertices.push_back({
            make_float4(0.5f * normal + 0.5f * left + 0.5f * up, 1),
            make_float4(normal, 0) });
        data.vertices.push_back({
            make_float4(0.5f * normal - 0.5f * left + 0.5f * up, 1),
            make_float4(normal, 0) });
        data.indices.push_back(startIdx + 0);
        data.indices.push_back(startIdx + 1);
        data.indices.push_back(startIdx + 2);
        data.indices.push_back(startIdx + 0);
        data.indices.push_back(startIdx + 2);
        data.indices.push_back(startIdx + 3);
    };
    addFace(make_float3(1, 0, 0), make_float3(0, 1, 0));
    addFace(make_float3(-1, 0, 0), make_float3(0, -1, 0));
    addFace(make_float3(0, 1, 0), make_float3(-1, 0, 0));
    addFace(make_float3(0, -1, 0), make_float3(1, 0, 0));
    addFace(make_float3(0, 0, 1), make_float3(0, 1, 0));
    addFace(make_float3(0, 0, -1), make_float3(0, -1, 0));

    return data;
}

MeshCpu MeshCpu::createWireCube()
{
    MeshCpu data;

    const auto addVertex = [&data](float x, float y, float z)
    {
        data.vertices.push_back({
        make_float4(x, y, z, 1),
        make_float4(normalize(make_float3(x,y,z)), 0) });
    };
    addVertex(-0.5f, -0.5f, -0.5f);
    addVertex(-0.5f, -0.5f, +0.5f);
    addVertex(-0.5f, +0.5f, -0.5f);
    addVertex(-0.5f, +0.5f, +0.5f);
    addVertex(+0.5f, -0.5f, -0.5f);
    addVertex(+0.5f, -0.5f, +0.5f);
    addVertex(+0.5f, +0.5f, -0.5f);
    addVertex(+0.5f, +0.5f, +0.5f);

    data.indices = {
        0, 1,
        2, 3,
        0, 2,
        1, 3,
        4, 5,
        6, 7,
        4, 6,
        5, 7,
        0, 4,
        1, 5,
        2, 6,
        3, 7
    };

    return data;
}

Mesh::~Mesh()
{
    try {
	    free();
    }
    catch (const cuMat::cuda_error& ex)
    {
        std::cerr << "Error on deconstructing Mesh: " << ex.what() << std::endl;
    }
}

void Mesh::free()
{
	if (vbo == 0) return; //already freed

    //CUMAT_SAFE_CALL_NO_THROW(cudaGraphicsUnregisterResource(vboCuda));
	//CUMAT_SAFE_CALL_NO_THROW(cudaGraphicsUnregisterResource(iboCuda));
    
    //No error checking. On program exit, when the context is already destroyed,
    //this throws tons of errors
    cudaGraphicsUnregisterResource(vboCuda);
    cudaGraphicsUnregisterResource(iboCuda);

	glBindVertexArray(vao);
	glDisableVertexAttribArray(POSITION_INDEX);
	glDisableVertexAttribArray(NORMAL_INDEX);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &ibo);
	glDeleteBuffers(1, &vbo);
	glBindVertexArray(0);
	glDeleteVertexArrays(1, &vao);
	//checkOpenGLError();

	vbo = 0;
	vao = 0;
	ibo = 0;
	numVertices = 0;
	numIndices = 0;
	vboCuda = 0;
	iboCuda = 0;
}

void Mesh::resize(int vertices, int indices)
{
	free();

	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	checkOpenGLError();

	glGenBuffers(1, &vbo); checkOpenGLError();
	glBindBuffer(GL_ARRAY_BUFFER, vbo); checkOpenGLError();
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices, nullptr, GL_DYNAMIC_DRAW); checkOpenGLError();

	glGenBuffers(1, &ibo); checkOpenGLError();
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); checkOpenGLError();
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint) * indices, nullptr, GL_DYNAMIC_DRAW); checkOpenGLError();

	glEnableVertexAttribArray(POSITION_INDEX); checkOpenGLError();
	glEnableVertexAttribArray(NORMAL_INDEX); checkOpenGLError();
	glVertexAttribPointer(POSITION_INDEX, 4, GL_FLOAT, false, 8 * sizeof(GLfloat), nullptr); checkOpenGLError();
	glVertexAttribPointer(NORMAL_INDEX, 4, GL_FLOAT, false, 8 * sizeof(GLfloat), (GLvoid*)(4 * sizeof(GLfloat))); checkOpenGLError();

	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&vboCuda, vbo, cudaGraphicsRegisterFlagsNone));
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&iboCuda, ibo, cudaGraphicsRegisterFlagsNone));
	
	glBindVertexArray(0); checkOpenGLError();

	numVertices = availableVertices = vertices;
	numIndices = availableIndices = indices;
}

void Mesh::reserve(int vertices, int indices)
{
	if (availableVertices < vertices || availableIndices < indices)
	{
		resize(std::max(availableVertices, vertices), std::max(availableIndices, indices));
	}
	numVertices = vertices;
	numIndices = indices;
}

void Mesh::drawIndexed(GLenum mode)
{
	if (numVertices == 0) return;
	glBindVertexArray(vao);
	checkOpenGLError();
	glDrawElements(mode, numIndices, GL_UNSIGNED_INT, nullptr);
	checkOpenGLError();
	glBindVertexArray(0);
}

void Mesh::drawArrays(GLenum mode)
{
    if (numVertices == 0) return;
    glBindVertexArray(vao);
    checkOpenGLError();
    glDrawArrays(mode, 0, numVertices);
    checkOpenGLError();
    glBindVertexArray(0);
}

Mesh::Mapping::Mapping(cudaGraphicsResource_t vboCuda, cudaGraphicsResource_t iboCuda)
    : vboCuda_(vboCuda)
    , iboCuda_(iboCuda)
{
    //glFinish();
    //checkOpenGLError();
    CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &vboCuda));
    CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &iboCuda));

    size_t s;
    CUMAT_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)(&vertices_), &s, vboCuda));
    CUMAT_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void**)(&indices_), &s, iboCuda));
    //CUMAT_SAFE_CALL(cudaDeviceSynchronize());
}

Mesh::Mapping::~Mapping()
{
    //CUMAT_SAFE_CALL(cudaDeviceSynchronize());
    CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &vboCuda_));
    CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &iboCuda_));
}

std::unique_ptr<Mesh::Mapping> Mesh::cudaMap()
{
    return std::unique_ptr<Mapping>(new Mapping(vboCuda, iboCuda));
}


void Mesh::copyToCpu(MeshCpu& meshCpu)
{
	meshCpu.vertices.resize(numVertices);
	meshCpu.indices.resize(numIndices);
    auto mapping = cudaMap();
	cudaMemcpy(meshCpu.vertices.data(), mapping->vertices(), sizeof(Vertex) * numVertices,
		cudaMemcpyDeviceToHost);
	cudaMemcpy(meshCpu.indices.data(), mapping->indices(), sizeof(GLuint) * numIndices,
		cudaMemcpyDeviceToHost);
}

void Mesh::copyFromCpu(const MeshCpu& meshCpu)
{
	reserve(meshCpu.vertices.size(), meshCpu.indices.size());

    //std::cout << "Mesh::copyFromCpu" << std::endl;
    //for (size_t i=0; i<meshCpu.vertices.size(); ++i)
    //{
    //    const auto& v = meshCpu.vertices[i];
    //    std::cout << " Vertex " << i << ": pos=(" << v.position.x << "," << v.position.y <<
    //        "," << v.position.z << "," << v.position.w << "), n=(" << v.normals.x << "," <<
    //        v.normals.y << "," << v.normals.z << "," << v.normals.w << ")\n";
    //}
    //for (size_t i=0; i<meshCpu.indices.size()/3; ++i)
    //{
    //    std::cout << " Indices " << i << ": [" <<
    //        meshCpu.indices[3 * i] << "," << meshCpu.indices[3 * i + 1] << "," <<
    //        meshCpu.indices[3 * i + 2] << "]\n";
    //}
    //std::cout << std::flush;

    auto mapping = cudaMap();
	cudaMemcpy(mapping->vertices(), meshCpu.vertices.data(), sizeof(Vertex) * meshCpu.vertices.size(),
		cudaMemcpyHostToDevice);
	cudaMemcpy(mapping->indices(), meshCpu.indices.data(), sizeof(GLuint) * meshCpu.indices.size(),
		cudaMemcpyHostToDevice);
}

void Mesh::setNumVertices(int num)
{
    if (num < 0 || num > availableVertices)
        throw std::runtime_error("num out of bounds: num<0 || num>=availableVertices");
    numVertices = num;
}

void Mesh::setNumIndices(int num)
{
    if (num < 0 || num >= availableIndices)
        throw std::runtime_error("num out of bounds: num<0 || num>=availableIndices");
    numIndices = num;
}


END_RENDERER_NAMESPACE

#endif
