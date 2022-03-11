#include "ray_evaluation_stepping.h"

#include <magic_enum.hpp>
#include <c10/cuda/CUDAStream.h>
#include <cuMat/src/Errors.h>
#include <lodepng.h>

#include "helper_math.cuh"
#include "image_evaluator_simple.h"
#include "module_registry.h"
#include "renderer_tensor.cuh"
#include "pytorch_utils.h"
#include "volume_interpolation_grid.h"

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(shaders);

bool renderer::IRayEvaluationStepping::stepsizeCanBeInObjectSpace(IVolumeInterpolation_ptr volume)
{
    return volume && volume->supportsObjectSpaceIndexing();
}

double renderer::IRayEvaluationStepping::getStepsizeWorld()
{
    return stepsize_;
}

bool renderer::IRayEvaluationStepping::drawStepsizeUI(UIStorage_t& storage)
{
    bool changed = false;
    IVolumeInterpolation_ptr volume = get_or(
        storage, ImageEvaluatorSimple::UI_KEY_SELECTED_VOLUME, IVolumeInterpolation_ptr());
    double maxSize = 1;
    if (volume && volume->supportsObjectSpaceIndexing())
    {
        
        double voxelSize = max_coeff(volume->voxelSize());
        double stepsMin = 0.1, stepsMax = 10.0;
        double stepsPerVoxel = voxelSize / stepsize_;
        if (ImGui::SliderDouble("Steps/voxel##IRayEvaluationStepping", &stepsPerVoxel, stepsMin, stepsMax, "%.2f", 2))
        {
            changed = true;
            stepsize_ = voxelSize / stepsPerVoxel;
        }
    }
    else
    {
        if (volume)
            maxSize = max_coeff(volume->boxSize());
        double stepsMin = 10, stepsMax = 1000.0;
        double stepsPerRay = maxSize / stepsize_;
        if (ImGui::SliderDouble("Steps/ray##IRayEvaluationStepping", &stepsPerRay, stepsMin, stepsMax, "%.2f", 2))
        {
            changed = true;
            stepsize_ = maxSize / stepsPerRay;
        }
    }

    return changed;
}

void renderer::IRayEvaluationStepping::load(const nlohmann::json& json, const ILoadingContext* context)
{
    IRayEvaluation::load(json, context);
    stepsize_ = json.value("stepsize", 0.005);
    if (json.value("stepsizeIsObjectSpace", false))
    {
        //fix for old config files that stored the stepsize in object space:
        //assume a grid of 256^3 for simplicity
        stepsize_ /= 256;
    }
}

void renderer::IRayEvaluationStepping::save(nlohmann::json& json, const ISavingContext* context) const
{
    IRayEvaluation::save(json, context);
    json["stepsize"] = stepsize_;
}

void renderer::IRayEvaluationStepping::registerPybindModule(pybind11::module& m)
{
    IRayEvaluation::registerPybindModule(m);

    //guard double registration
    static bool registered = false;
    if (registered) return;
    registered = true;
    
    namespace py = pybind11;
    py::class_<IRayEvaluationStepping, IRayEvaluation, std::shared_ptr<IRayEvaluationStepping>>(m, "IRayEvaluationStepping")
        .def_readwrite("stepsize", &IRayEvaluationStepping::stepsize_)
    ;
}

const char* renderer::RayEvaluationSteppingIso::SurfaceFeatureNames[] = {
    "Off",
    "first curvature k1",
    "second curvature k2",
    "mean curvature (k1+k2)/2",
    "gaussian curvature k1*k2",
    "curvature texture",
};

renderer::RayEvaluationSteppingIso::RayEvaluationSteppingIso()
    : isovalue_(0.5)
    , binarySearchSteps_(0)
    , selectedSurfaceFeature_(SurfaceFeatures::OFF)
    , numIsocontours_(5)
    , isocontourRange_(5)
{
    //setup texture for surface feature
    auto desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    CUMAT_SAFE_CALL(cudaMallocArray(&isocontourTextureArray_, &desc, ISOCONTOUR_TEXTURE_RESOLUTION));
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = isocontourTextureArray_;
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;
    CUMAT_SAFE_CALL(cudaCreateTextureObject(&isocontourTextureObject_, &resDesc, &texDesc, NULL));
    isocontourTextureCpu_.resize(ISOCONTOUR_TEXTURE_RESOLUTION);

    updateIsocontourTexture();

    //load 2D curvature texture
    const auto TEXTURE_PATH = "shaders/curvature-texture.png";
    auto fs = cmrc::shaders::get_filesystem();
    auto textureFile = fs.open(TEXTURE_PATH);
    unsigned char* textureData;
    unsigned textureWidth;
    unsigned textureHeight;
    unsigned ret = lodepng_decode_memory(
        &textureData, &textureWidth, &textureHeight,
        reinterpret_cast<const unsigned char*>(textureFile.begin()), textureFile.size(),
        LodePNGColorType::LCT_RGBA, 8);
    if (ret != 0) throw std::runtime_error("Unable to load curvature texture");

    desc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    CUMAT_SAFE_CALL(cudaMallocArray(&curvatureTextureArray_, &desc, textureWidth, textureHeight));
    memset(&resDesc, 0, sizeof(cudaResourceDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = curvatureTextureArray_;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;
    CUMAT_SAFE_CALL(cudaCreateTextureObject(&curvatureTextureObject_, &resDesc, &texDesc, NULL));
    CUMAT_SAFE_CALL(cudaMemcpyToArray(
        curvatureTextureArray_, 0, 0, textureData,
        textureWidth * textureHeight * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    free(textureData);
}

renderer::RayEvaluationSteppingIso::~RayEvaluationSteppingIso()
{
    CUMAT_SAFE_CALL_NO_THROW(cudaDestroyTextureObject(isocontourTextureObject_));
    CUMAT_SAFE_CALL_NO_THROW(cudaFreeArray(isocontourTextureArray_));
    CUMAT_SAFE_CALL_NO_THROW(cudaDestroyTextureObject(curvatureTextureObject_));
    CUMAT_SAFE_CALL_NO_THROW(cudaFreeArray(curvatureTextureArray_));
}

void renderer::RayEvaluationSteppingIso::updateIsocontourTexture()
{
    static const float3 COLOR_NEGATIVE = make_float3(0.96f, 0.0f, 0.5f); //magenta
    static const float3 COLOR_ZERO = make_float3(1, 1, 1); //white
    static const float3 COLOR_POSITIVE = make_float3(0, 1, 0); //green
    static const float3 COLOR_ISO_ZERO = make_float3(0, 0, 0.5); //dark blue
    static const float3 COLOR_ISO_OTHER = make_float3(0, 0, 0); //black
    static const float ISO_WIDTH = 20; //fraction of the colormap filled by the isovalue

    for (int i=0; i<ISOCONTOUR_TEXTURE_RESOLUTION; ++i)
    {
        //convert i to [-1,+1]
        float v1 = 2 * i / (ISOCONTOUR_TEXTURE_RESOLUTION - 1.0f) - 1;
        //base color based on the color ramp
        float3 rgb;
        if (v1 < 0)
            rgb = lerp(COLOR_NEGATIVE, COLOR_ZERO, v1 + 1);
        else
            rgb = lerp(COLOR_ZERO, COLOR_POSITIVE, v1);
        //isocontours
        if (numIsocontours_ > 0) {
            float v2 = v1 * numIsocontours_;
            int closestIsovalue = static_cast<int>(std::round(v2));
            float3 isocolor = closestIsovalue == 0 ? COLOR_ISO_ZERO : COLOR_ISO_OTHER;
            float distanceToIso = std::abs(v2 - closestIsovalue);
            float isoBlend = std::max(0.0f, 1.0f - ISO_WIDTH * distanceToIso);
            rgb = lerp(rgb, isocolor, isoBlend);
        }
        //fill texture
        isocontourTextureCpu_[i] = make_float4(rgb, 1.0f);
    }
    CUMAT_SAFE_CALL(cudaMemcpyToArray(
        isocontourTextureArray_, 0, 0, isocontourTextureCpu_.data(), 
        ISOCONTOUR_TEXTURE_RESOLUTION * 4 * sizeof(float), cudaMemcpyHostToDevice));
}

torch::Tensor renderer::RayEvaluationSteppingIso::performShading(const torch::Tensor& rayDirections,
    const torch::Tensor& normals, const std::optional<torch::Tensor>& curvatureOpt, CUstream stream)
{
    if (!curvatureOpt.has_value() && selectedSurfaceFeature_ != SurfaceFeatures::OFF)
    {
        throw std::runtime_error("A surface feature requiring curvature is needed, but no curvature tensor is passed");
    }

    CHECK_CUDA(rayDirections, true);
    CHECK_DIM(rayDirections, 2);
    CHECK_SIZE(rayDirections, 1, 3);
    int N = static_cast<int>(rayDirections.size(0));
    CHECK_CUDA(normals, true);
    CHECK_DIM(normals, 2);
    CHECK_SIZE(normals, 0, N);
    CHECK_SIZE(normals, 1, 3);
    CHECK_MATCHING_DTYPE(rayDirections, normals);
    if (curvatureOpt.has_value())
    {
        const torch::Tensor curvature = curvatureOpt.value();
        CHECK_CUDA(curvature, true);
        CHECK_DIM(curvature, 2);
        CHECK_SIZE(curvature, 0, N);
        CHECK_SIZE(curvature, 1, 2);
        CHECK_MATCHING_DTYPE(rayDirections, curvature);
    }

    //kernel
    const std::string kernelName = "RayEvaluationIsoShadingCurvature";
    std::vector<std::string> constantNames; //stays empty
    std::stringstream extraSource;
    extraSource << "#define KERNEL_DOUBLE_PRECISION "
        << (rayDirections.scalar_type() == GlobalSettings::kDouble ? 1 : 0)
        << "\n";
    extraSource << "#define RAY_EVALUATION_STEPPING__SURFACE_FEATURE "
        << static_cast<int>(selectedSurfaceFeature_)
        << "\n";
    extraSource << "#include \"renderer_ray_evaluation_stepping_iso_kernels.cuh\"\n";
    const auto fun0 = KernelLoader::Instance().getKernelFunction(
        kernelName, extraSource.str(), constantNames, false, false);
    if (!fun0.has_value())
        throw std::runtime_error("Unable to compile kernel");
    const auto fun = fun0.value();

    //output tensors
    auto colors = torch::empty({ N, 4 },
        at::TensorOptions().dtype(normals.scalar_type()).device(c10::kCUDA));

    //launch kernel
    int blockSize = fun.bestBlockSize();
    int minGridSize = std::min(
        int(CUMAT_DIV_UP(N, blockSize)),
        fun.minGridSize());
    dim3 virtual_size{
        static_cast<unsigned int>(N), 1, 1 };
    bool success = RENDERER_DISPATCH_FLOATING_TYPES(normals.scalar_type(), "RayEvaluationSteppingIso::performShading", [&]()
        {
            const auto accRayDirections = accessor< ::kernel::Tensor2Read<scalar_t>>(rayDirections);
            const auto accNormals = accessor< ::kernel::Tensor2Read<scalar_t>>(normals);
            const auto accCurvature = curvatureOpt.has_value()
                ? accessor< ::kernel::Tensor2Read<scalar_t>>(curvatureOpt.value())
                : ::kernel::Tensor2Read<scalar_t>();
            const auto accColors = accessor< ::kernel::Tensor2RW<scalar_t>>(colors);
            cudaTextureObject_t tex = selectedSurfaceFeature() == SurfaceFeatures::CURVATURE_TEXTURE
                ? curvatureTextureObject_ : isocontourTextureObject_;
            const void* args[] = { &virtual_size,
                &accNormals, &accCurvature, &accRayDirections, &accColors,
                &this->isocontourRange_, &tex};
            auto result = cuLaunchKernel(
                fun.fun(), minGridSize, 1, 1, blockSize, 1, 1,
                0, stream, const_cast<void**>(args), NULL);
            if (result != CUDA_SUCCESS)
                return printError(result, kernelName);
            return true;
        });

    if (!success) throw std::runtime_error("Error during rendering!");

    return colors;
}

std::string renderer::RayEvaluationSteppingIso::getName() const
{
    return "Iso";
}

void renderer::RayEvaluationSteppingIso::prepareRendering(GlobalSettings& s) const
{
    s.volumeShouldProvideNormals = true;
    if (selectedSurfaceFeature_ != SurfaceFeatures::OFF)
        s.volumeShouldProvideCurvature = true;
}

std::optional<int> renderer::RayEvaluationSteppingIso::getBatches(const GlobalSettings& s) const
{
    bool batched = std::holds_alternative<torch::Tensor>(isovalue_.value);
    if (batched)
    {
        torch::Tensor t = std::get<torch::Tensor>(isovalue_.value);
        CHECK_CUDA(t, true);
        CHECK_DIM(t, 1);
        return t.size(0);
    }
    return {};
}

std::string renderer::RayEvaluationSteppingIso::getDefines(const GlobalSettings& s) const
{
    auto volume = getSelectedVolume(s);
    if (!volume) return "";

    std::stringstream ss;
    ss << "#define RAY_EVALUATION_STEPPING__VOLUME_INTERPOLATION_T "
        << volume->getPerThreadType(s)
        << "\n";
    bool batched = std::holds_alternative<torch::Tensor>(isovalue_.value);
    if (batched)
        ss << "#define RAY_EVALUATION_STEPPING__ISOVALUE_BATCHED\n";
    ss << "#define RAY_EVALUATION_STEPPING__SURFACE_FEATURE " << int(selectedSurfaceFeature_) << "\n";
    return ss.str();
}

std::vector<std::string> renderer::RayEvaluationSteppingIso::getIncludeFileNames(const GlobalSettings& s) const
{
    return { "renderer_ray_evaluation_stepping_iso.cuh" };
}

std::string renderer::RayEvaluationSteppingIso::getConstantDeclarationName(const GlobalSettings& s) const
{
    return "rayEvaluationSteppingIsoParameters";
}

std::string renderer::RayEvaluationSteppingIso::getPerThreadType(const GlobalSettings& s) const
{
    return "::kernel::RayEvaluationSteppingIso";
}

void renderer::RayEvaluationSteppingIso::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
    auto volume = getSelectedVolume(s);
    if (!volume) throw std::runtime_error("No volume specified!");

    bool batched = std::holds_alternative<torch::Tensor>(isovalue_.value);
    if (batched)
    {
        torch::Tensor t = std::get<torch::Tensor>(isovalue_.value);
        CHECK_CUDA(t, true);
        CHECK_DIM(t, 1);
        CHECK_DTYPE(t, s.scalarType);
        RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "RayEvaluationSteppingIso", [&]()
            {
                struct Parameters
                {
                    scalar_t stepsize;
                    int binarySearchSteps;
                    ::kernel::Tensor1Read<scalar_t> isovalue;
                    scalar_t isocontourRange;
                    cudaTextureObject_t isocontourTexture;
                } p;
                p.stepsize = static_cast<scalar_t>(getStepsizeWorld());
                p.binarySearchSteps = binarySearchSteps_;
                p.isovalue = accessor<::kernel::Tensor1Read<scalar_t>>(t);
                p.isocontourRange = static_cast<scalar_t>(isocontourRange_);
                p.isocontourTexture = selectedSurfaceFeature() == SurfaceFeatures::CURVATURE_TEXTURE
                    ? curvatureTextureObject_ : isocontourTextureObject_;
                CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
            });
    }
    else
    {
        double value = std::get<double>(isovalue_.value);
        RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "RayEvaluationSteppingIso", [&]()
            {
                struct Parameters
                {
                    scalar_t stepsize;
                    int binarySearchSteps;
                    scalar_t isovalue;
                    scalar_t isocontourRange;
                    cudaTextureObject_t isocontourTexture;
                } p;
                p.stepsize = static_cast<scalar_t>(getStepsizeWorld());
                p.binarySearchSteps = binarySearchSteps_;
                p.isovalue = static_cast<scalar_t>(value);
                p.isocontourRange = static_cast<scalar_t>(isocontourRange_);
                p.isocontourTexture = selectedSurfaceFeature() == SurfaceFeatures::CURVATURE_TEXTURE
                    ? curvatureTextureObject_ : isocontourTextureObject_;
                CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
            });
    }
}

bool renderer::RayEvaluationSteppingIso::drawUI(UIStorage_t& storage)
{
    bool changed = IRayEvaluation::drawUI(storage);
    bool textureChanged = false;

    if (ImGui::CollapsingHeader("Renderer##IRayEvaluation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (drawStepsizeUI(storage))
            changed = true;
        if (ImGui::SliderInt("Binary Search##RayEvaluationSteppingIso",
            &binarySearchSteps_, 0, 20))
            changed = true;

        //get min, max density from storage
        float minDensity = 0, maxDensity = 1;
        if (const auto it = storage.find(VolumeInterpolationGrid::UI_KEY_MIN_DENSITY);
            it != storage.end())
        {
            minDensity = std::any_cast<float>(it->second);
        }
        if (const auto it = storage.find(VolumeInterpolationGrid::UI_KEY_MAX_DENSITY);
            it != storage.end())
        {
            maxDensity = std::any_cast<float>(it->second);
        }

        if (ImGui::SliderDouble("Isovalue##RayEvaluationSteppingIso",
            enforceAndGetScalar<double>(isovalue_), minDensity, maxDensity))
            changed = true;

        if (ImGui::Combo(
            "Surface Feature##RayEvaluationSteppingIso",
            reinterpret_cast<int*>(&selectedSurfaceFeature_),
            SurfaceFeatureNames,
            int(SurfaceFeatures::_NUM_FEATURES_)))
        {
            textureChanged = true;
        }

        if (selectedSurfaceFeature_ != SurfaceFeatures::OFF)
        {
            if (ImGui::SliderInt("Num Isocontours##RayEvaluationSteppingIso",
                &numIsocontours_, 0, 10))
                textureChanged = true;
            if (ImGui::SliderFloat("Isocontour Range##RayEvaluationSteppingIso",
                &isocontourRange_, 0.01f, 500.0f, "%.3f", 10))
                textureChanged = true;
        }
    }

    if (textureChanged)
    {
        updateIsocontourTexture();
        changed = true;
    }

    return changed;
}

void renderer::RayEvaluationSteppingIso::load(const nlohmann::json& json, const ILoadingContext* context)
{
    IRayEvaluationStepping::load(json, context);
    *enforceAndGetScalar<double>(isovalue_) = json.value("isovalue", 0.5);
    binarySearchSteps_ = json.value("binarySearchSteps", 0);
    selectedSurfaceFeature_ = magic_enum::enum_cast<SurfaceFeatures>(
        json.value("selectedSurfaceFeature", "")).value_or(SurfaceFeatures::OFF);
    numIsocontours_ = json.value("numIsocontours", 5);
    isocontourRange_ = json.value("isocontourRange", 5.0f);

    updateIsocontourTexture();
}

void renderer::RayEvaluationSteppingIso::save(nlohmann::json& json, const ISavingContext* context) const
{
    IRayEvaluationStepping::save(json, context);
    json["isovalue"] = *getScalarOrThrow<double>(isovalue_);
    json["binarySearchSteps"] = binarySearchSteps_;
    json["selectedSurfaceFeature"] = magic_enum::enum_name(selectedSurfaceFeature_);
    json["numIsocontours"] = numIsocontours_;
    json["isocontourRange"] = isocontourRange_;
}

void renderer::RayEvaluationSteppingIso::registerPybindModule(pybind11::module& m)
{
    IRayEvaluationStepping::registerPybindModule(m);

    //guard double registration
    static bool registered = false;
    if (registered) return;
    registered = true;
    
    namespace py = pybind11;
    py::class_<RayEvaluationSteppingIso, IRayEvaluationStepping, std::shared_ptr<RayEvaluationSteppingIso>> c(m, "RayEvaluationSteppingIso");
    py::enum_<SurfaceFeatures>(c, "SurfaceFeatures")
        .value("OFF", SurfaceFeatures::OFF)
        .value("FIRST_PRINCIPAL_CURVATURE", SurfaceFeatures::FIRST_PRINCIPAL_CURVATURE)
        .value("SECOND_PRINCIPAL_CURVATURE", SurfaceFeatures::SECOND_PRINCIPAL_CURVATURE)
        .value("MEAN_CURVATURE", SurfaceFeatures::MEAN_CURVATURE)
        .value("GAUSSIAN_CURVATURE", SurfaceFeatures::GAUSSIAN_CURVATURE)
        .export_values();
    c.def(py::init<>())
        .def_readonly("isovalue", &RayEvaluationSteppingIso::isovalue_,
            py::doc("double with the isovalue (possible batched as (B,) tensor)"))
        .def_property("binary_search_steps",
            &RayEvaluationSteppingIso::binarySearchSteps,
            &RayEvaluationSteppingIso::setBinarySearchSteps,
            py::doc("The number of binary search steps to refine the hit"))
        .def_property("surface_feature",
            &RayEvaluationSteppingIso::selectedSurfaceFeature,
            &RayEvaluationSteppingIso::setSelectedSurfaceFeature,
            py::doc("The surface feature to use for coloring"))
        .def_property("num_isocontours",
            &RayEvaluationSteppingIso::numIsocontours,
            &RayEvaluationSteppingIso::setNumIsocontours,
            py::doc("The number of isocontours to include in the surface feature mapping"))
        .def_property("isocontour_range",
            &RayEvaluationSteppingIso::isocontourRange,
            &RayEvaluationSteppingIso::setIsocontourRange,
            py::doc("The range of values [-v,+v] for the surface feature mapping"))
        .def("perform_shading", [](RayEvaluationSteppingIso* self, 
            const torch::Tensor& rayDirections, const torch::Tensor& normals, const std::optional<torch::Tensor>& curvature)
            {
                return self->performShading(rayDirections, normals, curvature, c10::cuda::getCurrentCUDAStream());
            },
            py::doc(R"(
                Performs shading based on the current curvature settings.
                :param rayDirections: tensor of shape (N,3) with the ray direction
                :param normals: Tensor of shape (N,3) with the surface normals.
                  The values will be normalized automatically.
                :param curvature: Tensor of shape (N,2) with the curvature
                :returns: A tensor of shape (N,4) with the blended color.
                )"),
            py::arg("ray_directions"), py::arg("normals"), py::arg("curvature"))
    ;
    
}


renderer::RayEvaluationSteppingDvr::RayEvaluationSteppingDvr()
    : alphaEarlyOut_(1.0 - 1e-5)
    , enableEarlyOut_(true)
    , minDensity_(0)
    , maxDensity_(1)
{
    const auto bx = ModuleRegistry::Instance().getModulesForTag(
        std::string(Blending::TAG));
    TORCH_CHECK(bx.size() == 1, "There should only be a single blending instance registered");
    blending_ = std::dynamic_pointer_cast<Blending>(bx[0].first);
}

std::string renderer::RayEvaluationSteppingDvr::getName() const
{
    return "DVR";
}

void renderer::RayEvaluationSteppingDvr::prepareRendering(GlobalSettings& s) const
{
    IRayEvaluation::prepareRendering(s);
    //nothing to do otherwise (for now)
}

std::string renderer::RayEvaluationSteppingDvr::getDefines(const GlobalSettings& s) const
{
    auto volume = getSelectedVolume(s);
    if (!volume) throw std::runtime_error("No volume loaded!");

    std::stringstream ss;
    ss << "#define RAY_EVALUATION_STEPPING__VOLUME_INTERPOLATION_T "
        << volume->getPerThreadType(s)
        << "\n";
    ss << "#define RAY_EVALUATION_STEPPING__TRANSFER_FUNCTION_T "
        << getSelectedTF()->getPerThreadType(s)
        << "\n";
    ss << "#define RAY_EVALUATION_STEPPING__BLENDING_T "
        << getSelectedBlending()->getPerThreadType(s)
        << "\n";
    ss << "#define RAY_EVALUATION_STEPPING__BRDF_T "
        << getSelectedBRDF()->getPerThreadType(s)
        << "\n";
    ss << "#define RAY_EVALUATION_STEPPING__ENABLE_EARLY_OUT "
        << (enableEarlyOut_ ? "1" : "0")
        << "\n";
    if (volume->outputType() == GlobalSettings::VolumeOutput::Density)
        ss << "#define RAY_EVALUATION_STEPPING__SKIP_TRANSFER_FUNCTION 0\n";
    else if (volume->outputType() == GlobalSettings::VolumeOutput::Color)
        ss << "#define RAY_EVALUATION_STEPPING__SKIP_TRANSFER_FUNCTION 1\n";
    else
        throw std::runtime_error("Output mode of the volume is unrecognized in RayEvaluationSteppingDvr");
    return ss.str();
}

std::vector<std::string> renderer::RayEvaluationSteppingDvr::getIncludeFileNames(const GlobalSettings& s) const
{
    return { "renderer_ray_evaluation_stepping_dvr.cuh" };
}

std::string renderer::RayEvaluationSteppingDvr::getConstantDeclarationName(const GlobalSettings& s) const
{
    return "rayEvaluationSteppingDvrParameters";
}

std::string renderer::RayEvaluationSteppingDvr::getPerThreadType(const GlobalSettings& s) const
{
    return "::kernel::RayEvaluationSteppingDvr";
}

void renderer::RayEvaluationSteppingDvr::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
    auto volume = getSelectedVolume(s);
    if (!volume) throw std::runtime_error("No volume loaded!");
    
    RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "RayEvaluationSteppingDvr", [&]()
        {
            struct Parameters
            {
                scalar_t stepsize;
                scalar_t alphaEarlyOut;
                scalar_t densityMin;
                scalar_t densityMax;
            } p;
            p.stepsize = static_cast<scalar_t>(getStepsizeWorld());
            p.alphaEarlyOut = static_cast<scalar_t>(alphaEarlyOut_);
            p.densityMin = static_cast<scalar_t>(minDensity_);
            p.densityMax = static_cast<scalar_t>(maxDensity_);
            CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
        });
}

bool renderer::RayEvaluationSteppingDvr::drawUI(UIStorage_t& storage)
{
    bool changed = IRayEvaluation::drawUI(storage);

    //TF
    const auto& tfs =
        ModuleRegistry::Instance().getModulesForTag(ITransferFunction::Tag());
    if (!tf_)
        tf_ = std::dynamic_pointer_cast<ITransferFunction>(tfs[0].first);
    if (ImGui::CollapsingHeader("Transfer Function##IRayEvaluation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        for (int i = 0; i < tfs.size(); ++i) {
            const auto& name = tfs[i].first->getName();
            if (ImGui::RadioButton(name.c_str(), tfs[i].first == tf_)) {
                tf_ = std::dynamic_pointer_cast<ITransferFunction>(tfs[i].first);
                changed = true;
            }
            if (i < tfs.size() - 1) ImGui::SameLine();
        }
        if (tf_->drawUI(storage))
            changed = true;
    }

    //rendering parameters
    if (ImGui::CollapsingHeader("Renderer##IRayEvaluation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        if (drawStepsizeUI(storage))
            changed = true;
        if (ImGui::Checkbox("Early Out##IRayEvaluation", &enableEarlyOut_))
            changed = true;

        //get min, max density from storage
        float minDensity = 0, maxDensity = 1;
        if (const auto it = storage.find(VolumeInterpolationGrid::UI_KEY_MIN_DENSITY);
            it != storage.end())
        {
            minDensity = std::any_cast<float>(it->second);
        }
        if (const auto it = storage.find(VolumeInterpolationGrid::UI_KEY_MAX_DENSITY);
            it != storage.end())
        {
            maxDensity = std::any_cast<float>(it->second);
        }

        //minDensity_ = fmax(minDensity_, minDensity);
        //maxDensity_ = fmin(maxDensity_, maxDensity);
        if (ImGui::SliderDouble("Min Density", &minDensity_, minDensity, maxDensity))
            changed = true;
        if (ImGui::SliderDouble("Max Density", &maxDensity_, minDensity, maxDensity))
            changed = true;
        storage[UI_KEY_SELECTED_MIN_DENSITY] = static_cast<double>(minDensity_);
        storage[UI_KEY_SELECTED_MAX_DENSITY] = static_cast<double>(maxDensity_);

        if (blending_->drawUI(storage))
            changed = true;
    }

    //BRDF
    const auto& brdfs =
        ModuleRegistry::Instance().getModulesForTag(IBRDF::Tag());
    if (!brdf_)
        brdf_ = std::dynamic_pointer_cast<IBRDF>(brdfs[0].first);
    if (ImGui::CollapsingHeader("BRDF##IRayEvaluation", ImGuiTreeNodeFlags_DefaultOpen))
    {
        for (int i = 0; i < brdfs.size(); ++i) {
            const auto& name = brdfs[i].first->getName();
            if (ImGui::RadioButton(name.c_str(), brdfs[i].first == brdf_)) {
                brdf_ = std::dynamic_pointer_cast<IBRDF>(brdfs[i].first);
                changed = true;
            }
            if (i < brdfs.size() - 1) ImGui::SameLine();
        }
        if (brdf_->drawUI(storage))
            changed = true;
    }

    return changed;
}

renderer::IModule_ptr renderer::RayEvaluationSteppingDvr::getSelectedModuleForTag(const std::string& tag) const
{
    if (tag == Blending::TAG)
        return blending_;
    else if (tag == ITransferFunction::TAG)
        return tf_;
    else if (tag == IBRDF::TAG)
        return brdf_;
    else
        return IRayEvaluation::getSelectedModuleForTag(tag);
}

std::vector<std::string> renderer::RayEvaluationSteppingDvr::getSupportedTags() const
{
    std::vector<std::string> tags = IRayEvaluation::getSupportedTags();
    tags.push_back(blending_->getTag());
    if (IModuleContainer_ptr mc = std::dynamic_pointer_cast<IModuleContainer>(blending_))
    {
        const auto& t = mc->getSupportedTags();
        tags.insert(tags.end(), t.begin(), t.end());
    }
    tags.push_back(tf_->getTag());
    if (IModuleContainer_ptr mc = std::dynamic_pointer_cast<IModuleContainer>(tf_))
    {
        const auto& t = mc->getSupportedTags();
        tags.insert(tags.end(), t.begin(), t.end());
    }
    tags.push_back(brdf_->getTag());
    if (IModuleContainer_ptr mc = std::dynamic_pointer_cast<IModuleContainer>(brdf_))
    {
        const auto& t = mc->getSupportedTags();
        tags.insert(tags.end(), t.begin(), t.end());
    }
    return tags;
}

void renderer::RayEvaluationSteppingDvr::load(const nlohmann::json& json, const ILoadingContext* context)
{
    IRayEvaluationStepping::load(json, context);
    
    std::string tfName = json.value("selectedTF", "");
    tf_ = std::dynamic_pointer_cast<ITransferFunction>(
        context->getModule(ITransferFunction::Tag(), tfName));

    std::string brdfName = json.value("selectedBRDF", "");
    brdf_ = std::dynamic_pointer_cast<IBRDF>(
        context->getModule(IBRDF::Tag(), brdfName));

    minDensity_ = json.value("minDensity", 0.0);
    maxDensity_ = json.value("maxDensity", 1.0);
    enableEarlyOut_ = json.value("earlyOut", true);
}

void renderer::RayEvaluationSteppingDvr::save(nlohmann::json& json, const ISavingContext* context) const
{
    IRayEvaluationStepping::save(json, context);
    json["selectedTF"] = tf_ ? tf_->getName() : "";
    json["selectedBRDF"] = brdf_? brdf_->getName() : "";
    json["minDensity"] = minDensity_;
    json["maxDensity"] = maxDensity_;
    json["earlyOut"] = enableEarlyOut_;
}

void renderer::RayEvaluationSteppingDvr::convertToTextureTF()
{
    if (!tf_) return;
    if (tf_->getName() == TransferFunctionTexture::Name()) 
        return; //already a texture TF

    auto newTF = std::make_shared<TransferFunctionTexture>();
    if (newTF->canPaste(tf_))
        newTF->doPaste(tf_);
    else
        std::cerr << "Copying to texture TF not supported from source TF (" << tf_->getName() << ")" << std::endl;
    tf_ = newTF;
}

void renderer::RayEvaluationSteppingDvr::registerPybindModule(pybind11::module& m)
{
    IRayEvaluationStepping::registerPybindModule(m);

    //guard double registration
    static bool registered = false;
    if (registered) return;
    registered = true;
    
    namespace py = pybind11;
    py::class_<RayEvaluationSteppingDvr, IRayEvaluationStepping, std::shared_ptr<RayEvaluationSteppingDvr>>(m, "RayEvaluationSteppingDvr")
        .def(py::init<>())
        .def_readwrite("min_density", &RayEvaluationSteppingDvr::minDensity_)
        .def_readwrite("max_density", &RayEvaluationSteppingDvr::maxDensity_)
        .def_readwrite("early_out", &RayEvaluationSteppingDvr::enableEarlyOut_)
        .def_readonly("blending", &RayEvaluationSteppingDvr::blending_)
        .def_readwrite("tf", &RayEvaluationSteppingDvr::tf_)
        .def_readwrite("brdf", &RayEvaluationSteppingDvr::brdf_)
        .def("convert_to_texture_tf", &RayEvaluationSteppingDvr::convertToTextureTF)
        ;
}
