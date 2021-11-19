#pragma once

#include <chrono>

#include "irasterization.h"
#include "opengl_mesh.h"
#include "opengl_shader.h"

BEGIN_RENDERER_NAMESPACE
/**
 * Performs particle integration and renders the particles via rasterization.
 * Requires a volume that has velocities (3-channel features).
 */
class ParticleIntegration : public IRasterization
{
    //integration
    double particlesPerSecond_;
    double particleLifetimeSeconds_;
    double particleSpeed_;
    bool running_;
    int numActiveParticles_;
    int seedingIndex_;
    std::chrono::steady_clock::time_point lastTime_;

    //seeding
    double3 seedCenter_;
    double3 seedSize_;
#if RENDERER_OPENGL_SUPPORT==1
    Mesh_ptr seedBoxMesh_;
    Shader_ptr seedBoxShader_;
#endif
    unsigned short seedingTime_;
    double seedFraction_;

    //rendering
    double particleSize_;
    double particleVelocityScaling_;
    glm::vec3 particleColor_;
#if RENDERER_OPENGL_SUPPORT==1
    Mesh_ptr particleMesh_;
    Shader_ptr particleShader_;
#endif

public:
    ParticleIntegration();
    static std::string Name();
    [[nodiscard]] std::string getName() const override;
    [[nodiscard]] bool drawUI(IModule::UIStorage_t& storage) override;
    void drawExtraInfo(IModule::UIStorage_t& storage) override;
    [[nodiscard]] bool updateUI(IModule::UIStorage_t& storage) override;
    void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
    void save(nlohmann::json& json, const ISavingContext* context) const override;
    void reset();
    void performRasterization(const RasterizingContext* context) override;
protected:
    void registerPybindModule(pybind11::module& m) override;

protected:
    void particlesSeedAndAdvect(double deltaTime, const RasterizingContext* context);
    void drawSeedBox(const RasterizingContext* context) const;
    void drawParticles(const RasterizingContext* context) const;

};

END_RENDERER_NAMESPACE
