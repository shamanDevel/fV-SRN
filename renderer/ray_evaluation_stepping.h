#pragma once

#include "ray_evaluation.h"

BEGIN_RENDERER_NAMESPACE

/**
 * Parent class for all ray evaluators that traverse the volume
 * with a constant step size.
 */
class IRayEvaluationStepping : public IRayEvaluation
{
protected:
    double stepsize_ = 0.005;

    bool stepsizeCanBeInObjectSpace(IVolumeInterpolation_ptr volume);

public:
    double getStepsizeWorld();

    void load(const nlohmann::json& json, const ILoadingContext* context) override;
    void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
    bool drawStepsizeUI(UIStorage_t& storage);
    void registerPybindModule(pybind11::module& m) override;
};

class RayEvaluationSteppingIso : public IRayEvaluationStepping
{
    //Scalar or tensor of shape (B,)
    Parameter<double> isovalue_;
    int binarySearchSteps_;

    //curvature lines
    //Kindlmann 2013 Curvature-based Transfer Functions
    enum class SurfaceFeatures
    {
        OFF,
        FIRST_PRINCIPAL_CURVATURE,
        SECOND_PRINCIPAL_CURVATURE,
        MEAN_CURVATURE,
        GAUSSIAN_CURVATURE,
        CURVATURE_TEXTURE,
        _NUM_FEATURES_
    };
    static const char* SurfaceFeatureNames[int(SurfaceFeatures::_NUM_FEATURES_)];
    SurfaceFeatures selectedSurfaceFeature_;
    //isocontour mapping
    static constexpr int ISOCONTOUR_TEXTURE_RESOLUTION = 1024;
    cudaArray_t isocontourTextureArray_{ 0 };
    cudaTextureObject_t isocontourTextureObject_{ 0 };
    std::vector<float4> isocontourTextureCpu_;
    cudaArray_t curvatureTextureArray_{ 0 };
    cudaTextureObject_t curvatureTextureObject_{ 0 };
    int numIsocontours_;
    float isocontourRange_;

    //TODO: shading

public:
    RayEvaluationSteppingIso();
    virtual ~RayEvaluationSteppingIso();
    void updateIsocontourTexture();

    std::string getName() const override;
    void prepareRendering(GlobalSettings& s) const override;
    std::optional<int> getBatches(const GlobalSettings& s) const override;
    std::string getDefines(const GlobalSettings& s) const override;
    std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
    std::string getConstantDeclarationName(const GlobalSettings& s) const override;
    std::string getPerThreadType(const GlobalSettings& s) const override;
    void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;
    bool drawUI(UIStorage_t& storage) override;
    void load(const nlohmann::json& json, const ILoadingContext* context) override;
    void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
    void registerPybindModule(pybind11::module& m) override;

public:
    /**
     * For isosurface tracing from PyTorch:
     * Performs shading based on the current curvature settings.
     * \param rayDirections tensor of shape (N,3) with the ray direction
     * \param normals Tensor of shape (N,3) with the surface normals.
     *   The values will be normalized automatically.
     * \param curvature Tensor of shape (N,2) with the curvature
     * \return A tensor of shape (N,4) with the blended color.
     */
    torch::Tensor performShading(
        const torch::Tensor& rayDirections,
        const torch::Tensor& normals,
        const std::optional<torch::Tensor>& curvature,
        CUstream stream);

    [[nodiscard]] int binarySearchSteps() const
    {
        return binarySearchSteps_;
    }

    void setBinarySearchSteps(const int binary_search_steps)
    {
        TORCH_CHECK(binary_search_steps >= 0, "binary_search_steps must be non-negative");
        binarySearchSteps_ = binary_search_steps;
    }

    [[nodiscard]] SurfaceFeatures selectedSurfaceFeature() const
    {
        return selectedSurfaceFeature_;
    }

    void setSelectedSurfaceFeature(const SurfaceFeatures selected_surface_feature)
    {
        selectedSurfaceFeature_ = selected_surface_feature;
    }

    [[nodiscard]] int numIsocontours() const
    {
        return numIsocontours_;
    }

    void setNumIsocontours(const int num_isocontours)
    {
        TORCH_CHECK(num_isocontours>0, "num_isocontours must be positive")
        numIsocontours_ = num_isocontours;
        updateIsocontourTexture();
    }

    [[nodiscard]] float isocontourRange() const
    {
        return isocontourRange_;
    }

    void setIsocontourRange(const float isocontour_range)
    {
        TORCH_CHECK(isocontour_range > 0, "isocontour_range must be positive")
        isocontourRange_ = isocontour_range;
        updateIsocontourTexture();
    }
};

class RayEvaluationSteppingDvr : public IRayEvaluationStepping
{
    const double alphaEarlyOut_;
    bool enableEarlyOut_;
    double minDensity_;
    double maxDensity_;

    Blending_ptr blending_;
    ITransferFunction_ptr tf_;
    IBRDF_ptr brdf_;

public:
    RayEvaluationSteppingDvr();

    [[nodiscard]] Blending_ptr getSelectedBlending() const { return blending_; }
    [[nodiscard]] ITransferFunction_ptr getSelectedTF() const { return tf_; }
    [[nodiscard]] IBRDF_ptr getSelectedBRDF() const { return brdf_; }

    std::string getName() const override;
    void prepareRendering(GlobalSettings& s) const override;
    std::string getDefines(const GlobalSettings& s) const override;
    std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
    std::string getConstantDeclarationName(const GlobalSettings& s) const override;
    std::string getPerThreadType(const GlobalSettings& s) const override;
    void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;
    bool drawUI(UIStorage_t& storage) override;
    IModule_ptr getSelectedModuleForTag(const std::string& tag) const override;
    std::vector<std::string> getSupportedTags() const override;
    void load(const nlohmann::json& json, const ILoadingContext* context) override;
    void save(nlohmann::json& json, const ISavingContext* context) const override;

    /**
     * For python: Converts the current TF to a texture TF.
     * This allows for pre-integration
     */
    void convertToTextureTF();
protected:
    void registerPybindModule(pybind11::module& m) override;
};

END_RENDERER_NAMESPACE
