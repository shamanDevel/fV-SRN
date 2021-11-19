#include "irasterization.h"
#include "image_evaluator_simple.h"
#include "particle_integration.h"
#include "rasterization_meshes.h"

#include <pybind11/stl.h>

void renderer::IRasterization::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;

	namespace py = pybind11;
	py::class_<IRasterization, IRasterization_ptr>(m, "IRasterization");
}

renderer::IVolumeInterpolation_ptr renderer::IRasterization::getSelectedVolume(const IModule::UIStorage_t& s) const
{
    return IModule::get_or(
        s, ImageEvaluatorSimple::UI_KEY_SELECTED_VOLUME, IVolumeInterpolation_ptr());
}

renderer::IVolumeInterpolation_ptr renderer::IRasterization::getSelectedVolume(const RasterizingContext* c) const
{
    return std::dynamic_pointer_cast<IVolumeInterpolation>(
        c->root->getSelectedModuleForTag(IVolumeInterpolation::Tag()));
}

std::unordered_map<std::string, renderer::RasterizationContainer::Factory_t> renderer::RasterizationContainer::Implementations_;

void renderer::RasterizationContainer::RegisterImplementation()
{
    static bool registered = false;
    if (!registered)
    {
        Implementations_.emplace(
			ParticleIntegration::Name(), 
			[]()->IRasterization_ptr {return std::make_shared<ParticleIntegration>(); });
		Implementations_.emplace(
			RasterizationMeshes::Name(),
			[]()->IRasterization_ptr {return std::make_shared<RasterizationMeshes>(); });
        registered = true;
    }
}

renderer::RasterizationContainer::RasterizationContainer()
{
    RegisterImplementation();
}


bool renderer::RasterizationContainer::drawUI(UIStorage_t& storage)
{
	bool changed = false;
	//add
	ImGui::Text("Add");
	for (const auto& e : Implementations_)
	{
		ImGui::SameLine();
		if (ImGui::Button(e.first.c_str()))
		{
			rasterizations_.push_back(e.second());
			changed = true;
		}
	}
	//show
	for (auto it = rasterizations_.begin(); it != rasterizations_.end(); )
	{
		auto m = *it;
		bool open = true;
		if (ImGui::CollapsingHeader(m->getName().c_str(), &open, ImGuiTreeNodeFlags_DefaultOpen))
		{
			if (m->drawUI(storage))
				changed = true;
		}
		if (!open) {
			rasterizations_.erase(it++);
			changed = true;
		}
		else
			++it;
	}
	return changed;
}

void renderer::RasterizationContainer::drawExtraInfo(UIStorage_t& storage)
{
	for (const auto& r : rasterizations_)
		r->drawExtraInfo(storage);
}

bool renderer::RasterizationContainer::updateUI(UIStorage_t& storage)
{
	bool changed = false;
	for (const auto& r : rasterizations_)
		if (r->updateUI(storage))
			changed = true;
	return changed;
}

void renderer::RasterizationContainer::load(const nlohmann::json& json, const ILoadingContext* fetcher)
{
	auto a = json["rasterizations"];
	rasterizations_.clear();
	int numRasterizations = a.size();
	for (int i=0; i<numRasterizations; ++i)
	{
		auto obj = a[i];
		std::string name = obj.at("type").get<std::string>();
		auto data = obj.at("data");
		auto instance = Implementations_[name]();
		instance->load(data, fetcher);
		rasterizations_.push_back(instance);
	}
}

void renderer::RasterizationContainer::save(nlohmann::json& json, const ISavingContext* context) const
{
	auto a = nlohmann::json::array();
	for (const auto& r : rasterizations_)
	{
		auto data = nlohmann::json::object();
		r->save(data, context);
		a.push_back(nlohmann::json::object({
			{"type", r->getName()},
			{"data", data}
			}));
	}
	json["rasterizations"] = a;
}

bool renderer::RasterizationContainer::hasRasterizing() const
{
	return !rasterizations_.empty();
}

void renderer::RasterizationContainer::performRasterization(const RasterizingContext* context)
{
	for (const auto& m : rasterizations_)
		m->performRasterization(context);
}

std::vector<std::string> renderer::RasterizationContainer::ImplementationNames()
{
    std::vector<std::string> v;
    for (const auto& e : Implementations_) v.push_back(e.first);
    return v;
}

renderer::IRasterization_ptr renderer::RasterizationContainer::addImplementation(const std::string& name)
{
	auto it = Implementations_.find(name);
	if (it == Implementations_.end())
		throw std::runtime_error("No implementation found");
	auto v = it->second();
	rasterizations_.push_back(v);
	return v;
}

void renderer::RasterizationContainer::registerPybindModule(pybind11::module& m)
{
	for (const auto& e : Implementations_)
		e.second()->registerPybindModule(m);

	namespace py = pybind11;
	py::class_<RasterizationContainer, RasterizationContainer_ptr>(m, "RasterizationContainer")
		.def("num_rasterizations", &RasterizationContainer::numRasterizations)
		.def_readonly("rasterizations", &RasterizationContainer::rasterizations_)
		.def_static("Implementation_names", &RasterizationContainer::ImplementationNames)
		.def("add_implementation", &RasterizationContainer::addImplementation)
		;
}
