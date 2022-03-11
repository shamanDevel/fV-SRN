#include "rasterization_meshes.h"

#include <glm/ext/matrix_transform.hpp>
#include <glm/gtx/string_cast.hpp>
#include <exception>

renderer::RasterizationMeshes::MeshInfo::MeshInfo(
    pyFloatArray vertices, pyFloatArray normals, pyIntArray indices, bool decouple)
    : ambientColor_(glm::vec3(0.5f))
    , diffuseColor_(glm::vec3(0.5f))
    , modelMatrix_(glm::identity<glm::mat4>())
{
#if RENDERER_OPENGL_SUPPORT==1
    if (vertices.ndim() != 2)
        throw std::runtime_error("Incompatible buffer dimension for the vertices, expected (N,3)!");
    if (vertices.shape(1) != 3)
        throw std::runtime_error("Incompatible buffer dimension for the vertices, expected (N,3)!");

    if (!decouple) {
        if (normals.ndim() != 2)
            throw std::runtime_error("Incompatible buffer dimension for the normals, expected (N,3)!");
        if (normals.shape(1) != 3)
            throw std::runtime_error("Incompatible buffer dimension for the normals, expected (N,3)!");
        if (vertices.shape(0) != normals.shape(0))
            throw std::runtime_error("Incompatible buffer dimension, vertices and normals must match!");
    }

    if (indices.ndim() != 2)
        throw std::runtime_error("Incompatible buffer dimension for the indices, expected (N,3)!");
    if (indices.shape(1) != 3)
        throw std::runtime_error("Incompatible buffer dimension for the indices, expected (N,3)!");

    MeshCpu cpu;
    const auto vertex = [&vertices](int i)
    {
        return make_float4(
            vertices.at(i, 0),
            vertices.at(i, 1),
            vertices.at(i, 2),
            1.0f);
    };
    if (!decouple) {
        for (ssize_t i = 0; i < vertices.shape(0); ++i)
        {
            auto position = vertex(i);
            auto normal = make_float4(
                normals.at(i, 0),
                normals.at(i, 1),
                normals.at(i, 2),
                0.0f);
            cpu.vertices.push_back({ position, normal });
        }
        for (ssize_t i = 0; i < indices.shape(0); ++i)
        {
            cpu.indices.push_back(indices.at(i, 0));
            cpu.indices.push_back(indices.at(i, 1));
            cpu.indices.push_back(indices.at(i, 2));
        }
    }
    else
    {
        for (ssize_t i = 0; i < indices.shape(0); ++i)
        {
            auto v1 = vertex(indices.at(i, 0));
            auto v2 = vertex(indices.at(i, 1));
            auto v3 = vertex(indices.at(i, 2));
            auto n = normalize(cross(make_float3(v3 - v1), make_float3(v2 - v1)));
            auto n4 = make_float4(-n, 0.0f);
            cpu.vertices.push_back({ v1, n4 });
            cpu.vertices.push_back({ v2, n4 });
            cpu.vertices.push_back({ v3, n4 });
            cpu.indices.push_back(3 * i);
            cpu.indices.push_back(3 * i + 1);
            cpu.indices.push_back(3 * i + 2);
        }
    }

    data_ = std::make_shared<Mesh>(cpu);
#else
    throw std::runtime_error("OpenGL disabled, can't create mesh!");
#endif
}

void renderer::RasterizationMeshes::MeshInfo::setModelMatrix(pyFloatArray matrix)
{
    if (matrix.ndim() != 2)
        throw std::runtime_error("Incompatible buffer dimension!");
    if (matrix.shape(0) != 4)
        throw std::runtime_error("Incompatible buffer size, expected (4,4)!");
    if (matrix.shape(1) != 4)
        throw std::runtime_error("Incompatible buffer size, expected (4,4)!");

    glm::mat4 m;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
        m[i][j] = matrix.at(i, j);
    modelMatrix_ = m;
}

void renderer::RasterizationMeshes::MeshInfo::setAmbientColor(glm::vec3 c)
{
    ambientColor_ = c;
}

void renderer::RasterizationMeshes::MeshInfo::setDiffuseColor(glm::vec3 c)
{
    diffuseColor_ = c;
}

renderer::RasterizationMeshes::RasterizationMeshes()
{
#if RENDERER_OPENGL_SUPPORT==1
    shader_ = std::make_shared<Shader>("PassThrough.vs", "SimpleDiffuse.fs");
#endif
}

std::string renderer::RasterizationMeshes::Name()
{
    return "Meshes";
}

std::string renderer::RasterizationMeshes::getName() const
{
    return Name();
}

bool renderer::RasterizationMeshes::drawUI(IModule::UIStorage_t& storage)
{
    //Nothing, this module is only supported in Python
    return false;
}

void renderer::RasterizationMeshes::drawExtraInfo(IModule::UIStorage_t& storage)
{
    IRasterization::drawExtraInfo(storage);
    //Nothing, this module is only supported in Python
}

bool renderer::RasterizationMeshes::updateUI(IModule::UIStorage_t& storage)
{
    return IRasterization::updateUI(storage);
    //Nothing, this module is only supported in Python
}

void renderer::RasterizationMeshes::load(const nlohmann::json& json, const ILoadingContext* fetcher)
{
    //Nothing, this module is only supported in Python
}

void renderer::RasterizationMeshes::save(nlohmann::json& json, const ISavingContext* context) const
{
    //Nothing, this module is only supported in Python
}

void renderer::RasterizationMeshes::performRasterization(const RasterizingContext* context)
{
#if RENDERER_OPENGL_SUPPORT==1
    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    shader_->use();

    //std::cout << "View:\n" << glm::to_string(context->view) << std::endl;
    //std::cout << "Projection:\n" << glm::to_string(context->projection) << std::endl;

    for (const auto& m : meshes_)
    {
        //std::cout << "Model:\n" << glm::to_string(m->modelMatrix_) << std::endl;
        glm::mat4 normalMatrix = transpose(inverse(context->view * m->modelMatrix_));
        //glm::mat4 normalMatrix = transpose(inverse(m->modelMatrix_));
        shader_->setMat4("modelMatrix", m->modelMatrix_);
        shader_->setMat4("normalMatrix", normalMatrix);
        shader_->setMat4("viewMatrix", context->view);
        shader_->setMat4("projectionMatrix", context->projection);
        shader_->setVec3("ambientColor", m->ambientColor_);
        shader_->setVec3("diffuseColor", m->diffuseColor_);
        shader_->setVec3("lightDirection", 0, 0, -1);
        shader_->setVec3("cameraOrigin", context->origin);

        m->data_->drawIndexed(GL_TRIANGLES);
    }
#else
    throw std::runtime_error("OpenGL-support disabled, RasterizationMeshes is not available");
#endif
}

void renderer::RasterizationMeshes::setMeshes(const std::vector<MeshInfo_ptr>& m)
{
    meshes_ = m;
}

void renderer::RasterizationMeshes::registerPybindModule(pybind11::module& m)
{
    IRasterization::registerPybindModule(m);

    //guard double registration
    static bool registered = false;
    if (registered) return;
    registered = true;

    namespace py = pybind11;
    py::class_<RasterizationMeshes, IRasterization, std::shared_ptr<RasterizationMeshes>> c(m, "RasterizationMeshes");

    py::class_<MeshInfo, MeshInfo_ptr> (c, "MeshInfo")
        .def("set_model_matrix", &MeshInfo::setModelMatrix,
            py::doc("Sets the model matrix, float buffer of shape (4,4)"))
        .def("set_ambient_color", [](MeshInfo* m, float r, float g, float b)
        {
                m->setAmbientColor(glm::vec3(r, g, b));
        }, py::doc("Sets the ambient color"),
            py::arg("r"), py::arg("g"), py::arg("b"))
        .def("set_diffuse_color", [](MeshInfo* m, float r, float g, float b)
        {
            m->setDiffuseColor(glm::vec3(r, g, b));
        }, py::doc("Sets the diffuse color"),
            py::arg("r"), py::arg("g"), py::arg("b"))
        ;

        c.def_static("create_mesh", &RasterizationMeshes::createMesh,
            py::doc(R"(
Constructs a new mesh from the given buffer
:param vertices: the vertex buffer, float buffer of shape (N,3)
:param normals: the vertex normal buffer, float buffer of shape (N,3)
:param indices: the index buffer, int buffer of shape (M,3)
)"), py::arg("vertices"), py::arg("normals"), py::arg("indices"), py::arg("decouple")=false)
        .def("set_meshes", &RasterizationMeshes::setMeshes,
            py::doc("Sets the meshes to render"), py::arg("meshes"))
        ;
}
