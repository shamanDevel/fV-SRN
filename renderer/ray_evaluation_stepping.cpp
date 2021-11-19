#include "ray_evaluation_stepping.h"

#include "helper_math.cuh"
#include "image_evaluator_simple.h"
#include "module_registry.h"
#include "renderer_tensor.cuh"
#include "pytorch_utils.h"
#include "volume_interpolation_grid.h"

bool renderer::IRayEvaluationStepping::stepsizeCanBeInObjectSpace(IVolumeInterpolation_ptr volume)
{
	return volume && volume->supportsObjectSpaceIndexing();
}

double renderer::IRayEvaluationStepping::getStepsizeWorld(IVolumeInterpolation_ptr volume)
{
	if (stepsizeIsObjectSpace_ && stepsizeCanBeInObjectSpace(volume))
	{
		return stepsize_ * max_coeff(volume->voxelSize());
	}
	return stepsize_;
}

bool renderer::IRayEvaluationStepping::drawStepsizeUI(UIStorage_t& storage)
{
	IVolumeInterpolation_ptr volume = get_or(
		storage, ImageEvaluatorSimple::UI_KEY_SELECTED_VOLUME, IVolumeInterpolation_ptr());
	bool changed = false;
	float stepMin = 0.01f, stepMax = 1.0f;
	if (ImGui::SliderDouble("Stepsize##IRayEvaluationStepping", &stepsize_, stepMin, stepMax, "%.5f", 2))
		changed = true;

	if (stepsizeCanBeInObjectSpace(volume))
	{
		if (ImGui::Checkbox("Object space##IRayEvaluationStepping", &stepsizeIsObjectSpace_))
			changed = true;
	}

	return changed;
}

void renderer::IRayEvaluationStepping::load(const nlohmann::json& json, const ILoadingContext* context)
{
	IRayEvaluation::load(json, context);
	stepsize_ = json.value("stepsize", 0.5);
	stepsizeIsObjectSpace_ = json.value("stepsizeIsObjectSpace", true);
}

void renderer::IRayEvaluationStepping::save(nlohmann::json& json, const ISavingContext* context) const
{
	IRayEvaluation::save(json, context);
	json["stepsize"] = stepsize_;
	json["stepsizeIsObjectSpace"] = stepsizeIsObjectSpace_;
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
		.def_readwrite("stepsizeIsObjectSpace", &IRayEvaluationStepping::stepsizeIsObjectSpace_);
}


renderer::RayEvaluationSteppingIso::RayEvaluationSteppingIso()
	: isovalue_(0.5)
{
}

std::string renderer::RayEvaluationSteppingIso::getName() const
{
	return "Iso";
}

void renderer::RayEvaluationSteppingIso::prepareRendering(GlobalSettings& s) const
{
	s.volumeShouldProvideNormals = true;
	//std::cout << "Isosurface rendering, request that the volume provides normals" << std::endl;
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
					::kernel::Tensor1Read<scalar_t> isovalue;
				} p;
				p.stepsize = static_cast<scalar_t>(getStepsizeWorld(volume));
				p.isovalue = accessor<::kernel::Tensor1Read<scalar_t>>(t);
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
					scalar_t isovalue;
				} p;
				p.stepsize = static_cast<scalar_t>(getStepsizeWorld(volume));
				p.isovalue = static_cast<scalar_t>(value);
				CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
			});
	}
}

bool renderer::RayEvaluationSteppingIso::drawUI(UIStorage_t& storage)
{
	bool changed = IRayEvaluation::drawUI(storage);

	if (ImGui::CollapsingHeader("Renderer##IRayEvaluation", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (drawStepsizeUI(storage))
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
	}

	return changed;
}

void renderer::RayEvaluationSteppingIso::load(const nlohmann::json& json, const ILoadingContext* context)
{
	IRayEvaluationStepping::load(json, context);
	*enforceAndGetScalar<double>(isovalue_) = json.value("isovalue", 0.5);
}

void renderer::RayEvaluationSteppingIso::save(nlohmann::json& json, const ISavingContext* context) const
{
	IRayEvaluationStepping::save(json, context);
	json["isovalue"] = *getScalarOrThrow<double>(isovalue_);
}

void renderer::RayEvaluationSteppingIso::registerPybindModule(pybind11::module& m)
{
	IRayEvaluationStepping::registerPybindModule(m);

	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<RayEvaluationSteppingIso, IRayEvaluationStepping, std::shared_ptr<RayEvaluationSteppingIso>>(m, "RayEvaluationSteppingIso")
		.def(py::init<>())
		.def_readonly("isovalue", &RayEvaluationSteppingIso::isovalue_,
			py::doc("double with the isovalue (possible batched as (B,) tensor)"));
}


renderer::RayEvaluationSteppingDvr::RayEvaluationSteppingDvr()
	: alphaEarlyOut_(1.0 - 1e-5)
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
			p.stepsize = static_cast<scalar_t>(getStepsizeWorld(volume));
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
}

void renderer::RayEvaluationSteppingDvr::save(nlohmann::json& json, const ISavingContext* context) const
{
	IRayEvaluationStepping::save(json, context);
	json["selectedTF"] = tf_ ? tf_->getName() : "";
	json["selectedBRDF"] = brdf_? brdf_->getName() : "";
	json["minDensity"] = minDensity_;
	json["maxDensity"] = maxDensity_;
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
		.def_readonly("blending", &RayEvaluationSteppingDvr::blending_)
		.def_readwrite("tf", &RayEvaluationSteppingDvr::tf_)
		.def_readwrite("brdf", &RayEvaluationSteppingDvr::brdf_)
		.def("convert_to_texture_tf", &RayEvaluationSteppingDvr::convertToTextureTF)
		;
}
