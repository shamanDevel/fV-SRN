#pragma once

#include <GL/glew.h>
#include <torch/types.h>
#include <cuda.h>

#include "iimage_evaluator.h"
#include "camera.h"
#include "ray_evaluation.h"
#include "opengl_framebuffer.h"
#include "irasterization.h"

BEGIN_RENDERER_NAMESPACE

/**
 * Simple image evaluator that stores a camera and ray evaluator.
 * No super- or subsampling, no postprocessing like SR networks.
 */
class ImageEvaluatorSimple : public IImageEvaluator
{
public:
	static const std::string UI_KEY_SELECTED_VOLUME;
	static const std::string NAME;

protected:
	ICamera_ptr selectedCamera_;
	IVolumeInterpolation_ptr selectedVolume_;
	IRayEvaluation_ptr selectedRayEvaluator_;
	RasterizationContainer_ptr rasterizationContainer_;
	bool isUImode_;

	int samplesPerIterationLog2_;
	unsigned int currentTime_;
	unsigned int refiningCounter_;

	bool useTonemapping_;
	float lastMaxExposure_;
	float tonemappingShoulder_;
	bool fixMaxExposure_;
	float fixedMaxExposure_;

#if RENDERER_OPENGL_SUPPORT==1
	std::unique_ptr<Framebuffer> framebuffer_;
#endif

public:
	ImageEvaluatorSimple();

    [[nodiscard]] virtual ICamera_ptr getSelectedCamera() const
        { return selectedCamera_; }
    [[nodiscard]] virtual IRayEvaluation_ptr getSelectedRayEvaluator() const
        { return selectedRayEvaluator_; }
	[[nodiscard]] virtual IVolumeInterpolation_ptr getSelectedVolume() const
        { return selectedVolume_; }
	[[nodiscard]] virtual RasterizationContainer_ptr getRasterizations() const
	    { return rasterizationContainer_; }

	void setSelectedVolume(IVolumeInterpolation_ptr v) { selectedVolume_ = v; }

	static const std::string& Name();
	std::string getName() const override;
	bool drawUI(UIStorage_t& storage) override;
	IModule_ptr getSelectedModuleForTag(const std::string& tag) const override;
	std::vector<std::string> getSupportedTags() const override;
	bool isIterativeRefining() const override;
	torch::Tensor render(int width, int height, CUstream stream, bool refine, const torch::Tensor& out) override;
	float getExposureForTonemapping() const;
    void extractColor(const torch::Tensor& inputTensor, tensor_or_texture_t output, ChannelMode channel, CUstream stream) override;
    static torch::Tensor extractColorTorch(const torch::Tensor& rawInputTensor, bool useTonemapping, float maxExposure, ChannelMode channel);
	void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
};
typedef std::shared_ptr<ImageEvaluatorSimple> ImageEvaluatorSimple_ptr;


END_RENDERER_NAMESPACE
