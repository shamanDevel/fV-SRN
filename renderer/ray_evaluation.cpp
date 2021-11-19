#include "ray_evaluation.h"

#include "image_evaluator_simple.h"
#include "module_registry.h"

renderer::IRayEvaluation::IRayEvaluation()
{
}

const std::string renderer::IRayEvaluation::UI_KEY_SELECTED_MIN_DENSITY = "IRayEvaluation::MinDensity";
const std::string renderer::IRayEvaluation::UI_KEY_SELECTED_MAX_DENSITY = "IRayEvaluation::MaxDensity";
bool renderer::IRayEvaluation::drawUI(UIStorage_t& storage)
{
	return false;
}

void renderer::IRayEvaluation::load(const nlohmann::json& json, const ILoadingContext* context)
{
	//old version: volumes are saved per ray-evaluation, not at the image evaluator
	//In newer save files, the volumes are stored at the image evaluator,
	// and this code does nothing
	std::string selectionName = json.value("selectedVolume", "");
	auto selectedVolume = std::dynamic_pointer_cast<IVolumeInterpolation>(
		context->getModule(IVolumeInterpolation::Tag(), selectionName));
	if (selectedVolume)
	{
	    //find parent
		auto parent = context->getModule(
			ImageEvaluatorSimple::Tag(), ImageEvaluatorSimple::Name());
		if (parent)
			std::dynamic_pointer_cast<ImageEvaluatorSimple>(parent)->setSelectedVolume(selectedVolume);
	}
}

void renderer::IRayEvaluation::save(nlohmann::json& json, const ISavingContext* context) const
{
	//nothing to save
}

void renderer::IRayEvaluation::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<IRayEvaluation, IRayEvaluation_ptr>(m, "IRayEvaluation");
	//Breaking change: the volume is now stored in the image evaluator
}

renderer::IModule_ptr renderer::IRayEvaluation::getSelectedModuleForTag(const std::string& tag) const
{
	return nullptr;
}

std::vector<std::string> renderer::IRayEvaluation::getSupportedTags() const
{
	return {};
}

renderer::IVolumeInterpolation_ptr renderer::IRayEvaluation::getSelectedVolume(const GlobalSettings& s) const
{
	return std::dynamic_pointer_cast<IVolumeInterpolation>(
		s.root->getSelectedModuleForTag(IVolumeInterpolation::Tag()));
}
