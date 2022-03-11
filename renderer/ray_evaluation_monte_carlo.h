#pragma once

#include "ray_evaluation.h"
#include "image_evaluator_simple.h"
//#include "brdf.h"
#include "opengl_mesh.h"
#include "opengl_shader.h"
#include "phase_function.h"

BEGIN_RENDERER_NAMESPACE

class RayEvaluationMonteCarlo : public IRayEvaluation
{
protected:
	double minDensity_;
	double maxDensity_;
	//double scatteringFactor_; //TODO: make Parameter<double>
	
	ITransferFunction_ptr tf_;
	double colorScaling_;
	//IBRDF_ptr brdf_;
	IPhaseFunction_ptr phaseFunction_;

	int numBounces_;
	
	//light
	double3 lightPitchYawDistance_;
	double lightRadius_;
	double lightIntensity_;
	//light UI
	bool showLight_;
#if RENDERER_OPENGL_SUPPORT==1
	Mesh lightMesh_;
	Shader lightShader_;
#endif

public:
	RayEvaluationMonteCarlo();

	bool requiresSampler() const override { return true; }
	bool shouldSupersample() const override { return true; }
	bool isIterativeRefining() const override { return true; }
	
	[[nodiscard]] ITransferFunction_ptr getSelectedTF() const { return tf_; }
	//[[nodiscard]] IBRDF_ptr getSelectedBRDF() const { return brdf_; }
	[[nodiscard]] IPhaseFunction_ptr getSelectedPhaseFunction() const { return phaseFunction_; }

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

    bool hasRasterizing() const override;
    void performRasterization(const RasterizingContext* context) override;

	static torch::Tensor SampleLight(IImageEvaluator_ptr context, int numSamples, unsigned int time, CUstream stream);
	static torch::Tensor EvalBackground(IImageEvaluator_ptr context, const torch::Tensor& rayStart, const torch::Tensor& rayDir, unsigned int time, CUstream stream);
	static std::tuple<torch::Tensor, torch::Tensor> NextDirection(
		IImageEvaluator_ptr context,
		const torch::Tensor& rayStart, const torch::Tensor& rayDir, unsigned int time, CUstream stream);
	static torch::Tensor PhaseFunctionProbability(IImageEvaluator_ptr context, 
		const torch::Tensor& dirIn, const torch::Tensor& dirOut, const torch::Tensor& position, CUstream stream);

protected:
	void registerPybindModule(pybind11::module& m) override;

private:
	double3 getLightPosition(IModuleContainer_ptr root) const;
};

END_RENDERER_NAMESPACE
