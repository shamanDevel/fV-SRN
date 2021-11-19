#pragma once

#include "imodule.h"
#include "parameter.h"
#include "volume_interpolation.h"
#include "transfer_function.h"
#include "blending.h"
#include "brdf.h"

BEGIN_RENDERER_NAMESPACE

/**
 * \brief Evaluates a single ray through the volume. Called via:
 * <code>
 *  struct RayEvaluationOutput
 *	{
 *		real4 color;
 *		real3 normal;
 *		real_t depth;
 *	};
 *  RayEvaluationOutput color = rayeval.eval(real3 rayStart, real3 rayDir, real_t tmax, int batch);
 * </code>
 * Additionally, if \ref requiresSampler() is true, the evaluation method shall
 * accept an additional input by reference with the sampler. The type of this
 * random number sampler is unspecified -> template.
 *
 * If \ref shouldSupersample(), the positions in the pixel should
 * be jittered and the ray evaluation should be called multiple times.
 *
 * Subclasses:
 * IRayEvaluation
 *  |- IRayEvaluationStepping (fixed per-ray or global step size)
 *  |   |- RayEvaluationSteppingIso
 *  |   |- RayEvaluationSteppingDvr
 *  |- IRayEvaluationAnalytic (analytic voxel traversal and integration)
 *  |   |- TODO
 *  |- RayEvaluatorMonteCarlo (monte carlo volume rendering, can spawn secondary rays)
 */
class IRayEvaluation : public IKernelModule, public IModuleContainer
{
public:
	static const std::string UI_KEY_SELECTED_MIN_DENSITY;
	static const std::string UI_KEY_SELECTED_MAX_DENSITY;

public:
	IRayEvaluation();
	virtual ~IRayEvaluation() = default;

	static constexpr std::string_view TAG = "RayEvaluation";
	std::string getTag() const override { return std::string(TAG); }
	static std::string Tag() { return std::string(TAG); }

	/**
	 * Returns if this ray evaluation requires a random sampler as additional input.
	 */
	virtual bool requiresSampler() const { return false; }
	/**
	 * Returns true if this ray evaluation is stochastic that
	 * requires multiple samples within the pixel for convergence.
	 */
	virtual bool shouldSupersample() const { return false; }

	/**
	 * Returns the iff this image evaluator iteratively refines the image.
	 * In that case, \render is called with \c refine=true
	 * if there was no UI interaction to allow for an iterative refinement.
	 */
	virtual bool isIterativeRefining() const { return false; }
	
	bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* context) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
public:
	IModule_ptr getSelectedModuleForTag(const std::string& tag) const override;
	std::vector<std::string> getSupportedTags() const override;
protected:
	IVolumeInterpolation_ptr getSelectedVolume(const GlobalSettings& s) const;
};
typedef std::shared_ptr<IRayEvaluation> IRayEvaluation_ptr;

END_RENDERER_NAMESPACE
