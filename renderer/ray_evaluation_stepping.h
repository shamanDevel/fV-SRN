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
	bool stepsizeIsObjectSpace_ = true;
	double stepsize_ = 0.5;

	bool stepsizeCanBeInObjectSpace(IVolumeInterpolation_ptr volume);

public:
	double getStepsizeWorld(IVolumeInterpolation_ptr volume);

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

	//TODO: shading

public:
	RayEvaluationSteppingIso();

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
};

class RayEvaluationSteppingDvr : public IRayEvaluationStepping
{
	const double alphaEarlyOut_;
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
