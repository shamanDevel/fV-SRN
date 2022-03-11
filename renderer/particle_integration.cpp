#include "particle_integration.h"

#include <c10/cuda/CUDAStream.h>
#include <cuMat/src/Macros.h>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/transform.hpp>
#include "kernel_loader.h"
#include "json_utils.h"

renderer::ParticleIntegration::ParticleIntegration()
    : particlesPerSecond_(32)
    , particleLifetimeSeconds_(4)
    , particleSpeed_(1/32.f)
    , running_(false)
    , numActiveParticles_(0)
    , seedingIndex_(0)
    , seedCenter_(make_double3(0,0,0))
    , seedSize_(make_double3(0.1, 0.1, 0.1))
    , seedingTime_(0)
    , seedFraction_(0)
    , particleSize_(1/128.f)
    , particleVelocityScaling_(1)
    , particleColor_(1,1,1)
{
    //meshes and shaders
#if RENDERER_OPENGL_SUPPORT==1
    seedBoxMesh_ = std::make_shared<Mesh>(MeshCpu::createWireCube());
    seedBoxShader_ = std::make_shared<Shader>("PassThrough.vs", "SimpleDiffuse.fs");

    particleMesh_ = nullptr; //created during rendering
    particleShader_ = std::make_shared<Shader>("Particles.vs", "Particles.fs", "Particles.gs");
#endif
}

std::string renderer::ParticleIntegration::Name()
{
    return "Particles";
}

std::string renderer::ParticleIntegration::getName() const
{
    return Name();
}

bool renderer::ParticleIntegration::updateUI(IModule::UIStorage_t& storage)
{
    return running_; //if running, always re-render
}

bool renderer::ParticleIntegration::drawUI(IModule::UIStorage_t& storage)
{
    bool changed = running_;
    ImGui::PushID("ParticleIntegration");

    if (ImGui::SliderDouble("Particles / second", &particlesPerSecond_, 1, 1024, "%.2f", 2))
    {
        changed = true;
        reset();
    }
    if (ImGui::SliderDouble("Particle Lifetime", &particleLifetimeSeconds_, 1/4.f, 64, "%.2f", 2))
    {
        changed = true;
        reset();
    }
    if (ImGui::SliderDouble("Particle Speed", &particleSpeed_, 1 / 256.f, 1.f, "%.5f", 2))
        changed = true;
    if (ImGui::SliderDouble3("Seed Center", &seedCenter_.x, -1, +1))
        changed = true;
    if (ImGui::SliderDouble3("Seed Size", &seedSize_.x, 0.001, +1))
        changed = true;
    if (ImGui::SliderDouble("Particle Size", &particleSize_, 1 / 1024.f, 1/32.f, "%.7f", 2))
        changed = true;
    if (ImGui::SliderDouble("Velocity Scaling", &particleVelocityScaling_, 1 / 16.f, 16.f, "%.4f", 2))
        changed = true;
    if (ImGui::ColorEdit3("Particle Color", &particleColor_.x,
        ImGuiColorEditFlags_RGB | ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_Float))
        changed = true;

    if (ImGui::Checkbox("RUN!", &running_))
    {
        changed = true;
        lastTime_ = std::chrono::steady_clock::now();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear"))
    {
        running_ = false;
        reset();
        changed = true;
    }

    ImGui::PopID();
    return changed;
}

void renderer::ParticleIntegration::drawExtraInfo(IModule::UIStorage_t& storage)
{
    IRasterization::drawExtraInfo(storage);
    ImGui::Text("#Particles: %d", numActiveParticles_);
}

void renderer::ParticleIntegration::load(const nlohmann::json& json, const ILoadingContext* fetcher)
{
    particlesPerSecond_ = json.value("particlesPerSecond", particlesPerSecond_);
    particleLifetimeSeconds_ = json.value("particleLifetimeSeconds", particleLifetimeSeconds_);
    particleSpeed_ = json.value("particleSpeed", particleSpeed_);
    seedCenter_ = json.value("seedCenter", seedCenter_);
    seedSize_ = json.value("seedSize", seedSize_);
    particleSize_ = json.value("particleSize", particleSize_);
    particleVelocityScaling_ = json.value("particleVelocityScaling", particleVelocityScaling_);
    particleColor_ = json.value("particleColor", particleColor_);
}

void renderer::ParticleIntegration::save(nlohmann::json& json, const ISavingContext* context) const
{
    json["particlesPerSecond"] = particlesPerSecond_;
    json["particleLifetimeSeconds"] = particleLifetimeSeconds_;
    json["particleSpeed"] = particleSpeed_;
    json["seedCenter"] = seedCenter_;
    json["seedSize"] = seedSize_;
    json["particleSize"] = particleSize_;
    json["particleVelocityScaling"] = particleVelocityScaling_;
    json["particleColor"] = particleColor_;
}

void renderer::ParticleIntegration::performRasterization(const RasterizingContext* context)
{
    try {
        if (running_) {
            auto now = std::chrono::steady_clock::now();
            double deltaTime = std::chrono::duration_cast<std::chrono::microseconds>(now - lastTime_).count() / 1'000'000.f;
            particlesSeedAndAdvect(deltaTime, context);
            lastTime_ = now;
        }

        drawSeedBox(context);
        drawParticles(context);
    } catch (const std::exception& ex)
    {
        running_ = false;
        std::cerr << "Unable to render/advect particles: " << ex.what();
    }
}

void renderer::ParticleIntegration::reset()
{
    numActiveParticles_ = 0;
    seedingIndex_ = 0;
}

void renderer::ParticleIntegration::particlesSeedAndAdvect(double deltaTime, const RasterizingContext* context)
{
#if RENDERER_OPENGL_SUPPORT==1
    //check max particles and resize mesh if needed
    const int maxParticles = std::ceil(particlesPerSecond_) * std::ceil(particleLifetimeSeconds_);
    if (particleMesh_==nullptr)
        particleMesh_ = std::make_shared<Mesh>();
    if (particleMesh_->getAvailableVertices() < maxParticles)
    {
        particleMesh_->resize(maxParticles, 1);
        reset();
    }

    CUstream stream = c10::cuda::getCurrentCUDAStream();

    //assemble kernel source
    IVolumeInterpolation_ptr volume = getSelectedVolume(context);
    IKernelModule::GlobalSettings s;
    s.root = context->root;
    s.interpolationInObjectSpace = false;
    s.scalarType = IKernelModule::GlobalSettings::kFloat;
    s.volumeShouldProvideNormals = false;
    s.volumeOutput = IKernelModule::GlobalSettings::Velocity;
    volume->prepareRendering(s);
    std::stringstream defines;
    std::stringstream includes;
    std::vector<std::string> constantNames;
    defines << "#define PARTICLE_INTEGRATION__VOLUME_INTERPOLATION_T " <<
        volume->getPerThreadType(s) << "\n";
    defines << "#define KERNEL_DOUBLE_PRECISION "
        << (s.scalarType == IKernelModule::GlobalSettings::kDouble ? 1 : 0)
        << "\n";
    defines <<
        "// " << volume->getTag() << " : " << volume->getName() << "\n"
        << volume->getDefines(s) << "\n";
    includes <<
        "// " << volume->getTag() << " : " << volume->getName() << "\n";
    for (const auto& i : volume->getIncludeFileNames(s))
        includes << "#include \"" << i << "\"\n";
    const auto c = volume->getConstantDeclarationName(s);
    if (!c.empty())
        constantNames.push_back(c);
    includes << "#include \"renderer_particle_integration_kernels.cuh\"\n";

    std::stringstream sourceFile;
    sourceFile << "// DEFINES:\n" << defines.str() << "\n";
    volume->fillExtraSourceCode(s, sourceFile);
    sourceFile << "\n// INCLUDES:\n" << includes.str() << "\n";

    //create kernel
    const auto seedKernel = KernelLoader::Instance().getKernelFunction(
        "ParticleIntegrateSeed", sourceFile.str(), constantNames, false, false);
    const auto advectionKernel = KernelLoader::Instance().getKernelFunction(
        "ParticleIntegrationAdvect", sourceFile.str(), constantNames, false, false);
    if (!seedKernel.has_value())
        throw std::runtime_error("Error compiling seeding kernel");
    if (!advectionKernel.has_value())
        throw std::runtime_error("Error compiling advection kernel");
    volume->fillConstantMemory(s, advectionKernel.value().constant(c), stream);

    //bind particles to CUDA
    auto mapping = particleMesh_->cudaMap();
    Vertex* particles = mapping->vertices();

    //seed 'particlesToSeed' particles starting at index seedingIndex_ modulo maxParticles
    seedFraction_ += deltaTime * particlesPerSecond_;
    int particlesToSeed = static_cast<int>(seedFraction_);
    seedFraction_ -= particlesToSeed;
    if (particlesToSeed>0) {
        //invoke seeding kernel
        int minGridSize = std::min(
            int(CUMAT_DIV_UP(particlesToSeed, seedKernel.value().bestBlockSize())),
            seedKernel.value().minGridSize());
        dim3 virtual_size{
            static_cast<unsigned int>(particlesToSeed),
            static_cast<unsigned int>(1),
            static_cast<unsigned int>(1) };
        unsigned int time = seedingTime_++;
        float3 seedMin = make_float3(seedCenter_ - seedSize_ * 0.5);
        float3 seedSize = make_float3(seedSize_);
        const void* args[] = {
            &virtual_size, &particles, &seedingIndex_, &maxParticles,
            &seedMin, &seedSize, &time };
        auto result = cuLaunchKernel(
            seedKernel.value().fun(), minGridSize, 1, 1, seedKernel.value().bestBlockSize(), 1, 1,
            0, stream, const_cast<void**>(args), NULL);
        if (result != CUDA_SUCCESS) {
            printError(result, "ParticleIntegrateSeed");
            throw std::runtime_error("CUDA error: unable to seed particles");
        }

        //increment seeding index and number of active particles
        seedingIndex_ = (seedingIndex_ + particlesToSeed) % maxParticles;
        numActiveParticles_ = std::min(numActiveParticles_ + particlesToSeed, maxParticles);
    }

    //advect
    float speed = static_cast<float>(deltaTime * particleSpeed_);
    if (speed>0 && numActiveParticles_>0)
    {
        //invoke advection kernel
        int minGridSize = std::min(
            int(CUMAT_DIV_UP(numActiveParticles_, advectionKernel.value().bestBlockSize())),
            advectionKernel.value().minGridSize());
        dim3 virtual_size{
            static_cast<unsigned int>(numActiveParticles_),
            static_cast<unsigned int>(1),
            static_cast<unsigned int>(1) };
        const void* args[] = {
            &virtual_size, &particles, &speed };
        auto result = cuLaunchKernel(
            advectionKernel.value().fun(), minGridSize, 1, 1, advectionKernel.value().bestBlockSize(), 1, 1,
            0, stream, const_cast<void**>(args), NULL);
        if (result != CUDA_SUCCESS) {
            printError(result, "ParticleIntegrationAdvect");
            throw std::runtime_error("CUDA error: unable to advect particles");
        }
    }
#else
throw std::runtime_error("OpenGL-support disabled, ParticleIntegration is not available");
#endif
}

void renderer::ParticleIntegration::drawSeedBox(const RasterizingContext* context) const
{
#if RENDERER_OPENGL_SUPPORT==1
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    seedBoxShader_->use();

    std::cout << "View:\n" << glm::to_string(context->view) << std::endl;
    std::cout << "Projection:\n" << glm::to_string(context->projection) << std::endl;

    glm::mat4 modelMatrix = 
        glm::translate(glm::vec3(seedCenter_.x, seedCenter_.y, seedCenter_.z))
        * glm::scale(glm::vec3(seedSize_.x, seedSize_.y, seedSize_.z));
    glm::mat4 normalMatrix = transpose(inverse(context->view * modelMatrix));
    seedBoxShader_->setMat4("modelMatrix", modelMatrix);
    seedBoxShader_->setMat4("normalMatrix", normalMatrix);
    seedBoxShader_->setMat4("viewMatrix", context->view);
    seedBoxShader_->setMat4("projectionMatrix", context->projection);
    seedBoxShader_->setVec3("ambientColor", 0,0,1);
    seedBoxShader_->setVec3("diffuseColor", 0,0,0);
    seedBoxShader_->setVec3("lightDirection", 1, 0, 0);
    seedBoxShader_->setVec3("cameraOrigin", context->origin);

    seedBoxMesh_->drawIndexed(GL_LINES);
#else
    throw std::runtime_error("OpenGL-support disabled, ParticleIntegration is not available");
#endif
}

void renderer::ParticleIntegration::drawParticles(const RasterizingContext* context) const
{
#if RENDERER_OPENGL_SUPPORT==1
    if (!particleMesh_ || numActiveParticles_==0) return;
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    particleShader_->use();

    glm::mat4 normalMatrix = transpose(inverse(context->view));
    particleShader_->setMat4("viewMatrix", context->view);
    particleShader_->setMat4("projectionMatrix", context->projection);
    particleShader_->setMat4("normalMatrix", normalMatrix);
    particleShader_->setVec3("ambientColor", 0.5f * particleColor_);
    particleShader_->setVec3("diffuseColor", 0.5f * particleColor_);
    particleShader_->setVec3("lightDirection", 0, 0, 1);
    particleShader_->setVec3("cameraOrigin", context->origin);
    particleShader_->setFloat("speedMultiplier", particleVelocityScaling_);
    particleShader_->setFloat("particleSize", particleSize_);

    particleMesh_->setNumVertices(numActiveParticles_);
    particleMesh_->drawArrays(GL_POINTS);
#else
throw std::runtime_error("OpenGL-support disabled, ParticleIntegration is not available");
#endif
}

void renderer::ParticleIntegration::registerPybindModule(pybind11::module& m)
{
    IRasterization::registerPybindModule(m);

    //guard double registration
    static bool registered = false;
    if (registered) return;
    registered = true;

    namespace py = pybind11;
    py::class_<ParticleIntegration, IRasterization, std::shared_ptr<ParticleIntegration>>(m, "ParticleIntegration")
        .def_readwrite("particles_per_second", &ParticleIntegration::particlesPerSecond_)
        .def_readwrite("particle_lifetime_seconds", &ParticleIntegration::particleLifetimeSeconds_)
        .def_readwrite("particle_speed", &ParticleIntegration::particleSpeed_)
        .def_readwrite("running", &ParticleIntegration::running_)
        .def_readonly("num_active_particles", &ParticleIntegration::numActiveParticles_)
        .def_readwrite("seed_center", &ParticleIntegration::seedCenter_)
        .def_readwrite("seed_size", &ParticleIntegration::seedSize_)
        .def_readwrite("particle_size", &ParticleIntegration::particleSize_)
        .def_readwrite("particle_velocity_scaling", &ParticleIntegration::particleVelocityScaling_)
        .def_readwrite("particle_color", &ParticleIntegration::particleColor_)
        .def("reset", &ParticleIntegration::reset)
        ;
}
