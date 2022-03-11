#pragma once

#include "transfer_function.h"
#include "volume_interpolation.h"
#include "lru_cache.h"

BEGIN_RENDERER_NAMESPACE

class TransferFunctionTex2D : public ITransferFunction
{
public:
    TransferFunctionTex2D();
    ~TransferFunctionTex2D() override;

    [[nodiscard]] std::string getName() const override;
    [[nodiscard]] bool drawUI(UIStorage_t& storage) override;
    void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
    void save(nlohmann::json& json, const ISavingContext* context) const override;
    void prepareRendering(GlobalSettings& s) const override;
    [[nodiscard]] std::optional<int> getBatches(const GlobalSettings& s) const override;
    [[nodiscard]] std::string getDefines(const GlobalSettings& s) const override;
    [[nodiscard]] std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
    [[nodiscard]] std::string getConstantDeclarationName(const GlobalSettings& s) const override;
    [[nodiscard]] std::string getPerThreadType(const GlobalSettings& s) const override;
protected:
    void fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr, double stepsize, CUstream stream) override;
    void registerPybindModule(pybind11::module& m) override;
public:
    bool canPaste(std::shared_ptr<ITransferFunction> other) override;
    void doPaste(std::shared_ptr<ITransferFunction> other) override;
    double4 evaluate(double density) const override;
    double getMaxAbsorption() const override;
    bool requiresGradients() const override;

private:
    //TF
    static constexpr int RESOLUTION_DENSITY = 256;
    static constexpr int RESOLUTION_GRADIENT = 256;
    cudaArray_t tfTextureArray_;
    cudaTextureObject_t tfTextureObject_;
    std::vector<float> tfTextureCpu_;

    //configuration
    double maxGradient_;
    double globalAbsorptionScaling_;
    struct Point
    {
        float density;
        float gradientMagnitude;
    };
    struct Material
    {
        Point edges[4];
        ImVec4 color;
        float lowerAbsorption;
        float upperAbsorption;
    };
    std::vector<Material> material_;

    //UI
    IVolumeInterpolation_ptr currentVolume_;
    //per-volume cached info
    class VolumeCache
    {
        float maxGradient_;
        cudaArray_t histogramTextureArray_;
        cudaTextureObject_t histogramTextureObject_;
    public:
        VolumeCache(IVolumeInterpolation_ptr volume);
        ~VolumeCache();
        float maxGradient() const { return maxGradient_; }
        cudaTextureObject_t texture() const { return histogramTextureObject_; }
    };
    typedef std::shared_ptr<VolumeCache> VolumeCache_ptr;
    LRUCache<IVolumeInterpolation*, VolumeCache_ptr> volumeInfoCache_;
    //point editing
    int selectedMaterial_;
    int selectedEdge_;
    bool isDragging_;
};
typedef std::shared_ptr<TransferFunctionTex2D> TransferFunctionTex2D_ptr;

END_RENDERER_NAMESPACE
