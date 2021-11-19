#pragma once

#include "irasterization.h"
#include "opengl_mesh.h"
#include "opengl_shader.h"
#include <pybind11/numpy.h>

BEGIN_RENDERER_NAMESPACE
/**
 * Allows to freely place meshes into the 3D scene.
 * This is used for debugging and video renderings from the python side.
 */
class RasterizationMeshes : public IRasterization
{
public:
    typedef py::array_t<int, py::array::c_style | py::array::forcecast> pyIntArray;
    typedef py::array_t<float, py::array::c_style | py::array::forcecast> pyFloatArray;
    class MeshInfo
    {
    public:
        /**
         * Constructs a new mesh from the given buffer
         * \param vertices the vertex buffer, float buffer of shape (N,3)
         * \param normals the vertex normal buffer, float buffer of shape (N,3)
         * \param indices the index buffer, int buffer of shape (M,3)
         * \param decouple if true, the vertices are no longer shared via the index buffer and
         *   the normals are recomputed from the triangle vertices
         */
        MeshInfo(pyFloatArray vertices, pyFloatArray normals, pyIntArray indices, bool decouple);
        /**
         * Sets the model matrix, float buffer of shape (4,4)
         */
        void setModelMatrix(pyFloatArray matrix);

        void setAmbientColor(glm::vec3 c);
        void setDiffuseColor(glm::vec3 c);

    private:
        friend class RasterizationMeshes;
        glm::vec3 ambientColor_;
        glm::vec3 diffuseColor_;
        glm::mat4 modelMatrix_;
        Mesh_ptr data_;
    };
    typedef std::shared_ptr<MeshInfo> MeshInfo_ptr;

public:
    RasterizationMeshes();
    static std::string Name();
    [[nodiscard]] std::string getName() const override;
    [[nodiscard]] bool drawUI(IModule::UIStorage_t& storage) override;
    void drawExtraInfo(IModule::UIStorage_t& storage) override;
    [[nodiscard]] bool updateUI(IModule::UIStorage_t& storage) override;
    void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
    void save(nlohmann::json& json, const ISavingContext* context) const override;
    void performRasterization(const RasterizingContext* context) override;

    static MeshInfo_ptr createMesh(pyFloatArray vertices, pyFloatArray normals, 
        pyIntArray indices, bool decouple)
    {
        return std::make_shared<MeshInfo>(vertices, normals, indices, decouple);
    }
    void setMeshes(const std::vector<MeshInfo_ptr>& m);

protected:
    void registerPybindModule(pybind11::module& m) override;

private:
    Shader_ptr shader_;
    std::vector<MeshInfo_ptr> meshes_;
};

END_RENDERER_NAMESPACE
