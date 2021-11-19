#include "blending.h"

#include <magic_enum.hpp>

std::string renderer::Blending::getName() const
{
	return "blending";
}

bool renderer::Blending::drawUI(UIStorage_t& storage)
{
	const auto num_enums = magic_enum::enum_count<BlendMode>();
	const auto current_name = magic_enum::enum_name(blendMode_);
	if (ImGui::SliderInt("Blend Mode##Blending", reinterpret_cast<int*>(&blendMode_),
		0, num_enums - 1, current_name.data()))
		return true;
	return false;
}

void renderer::Blending::load(const nlohmann::json& json, const ILoadingContext* context)
{
	blendMode_ = magic_enum::enum_cast<BlendMode>(json.value("blending", "")).
		value_or(BlendMode::BeerLambert);
}

void renderer::Blending::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["blending"] = magic_enum::enum_name(blendMode_);
}

void renderer::Blending::registerPybindModule(pybind11::module& m)
{
	namespace py = pybind11;
	py::class_<Blending> c(m, "Blending");
	py::enum_<BlendMode>(c, "BlendMode")
		.value("Alpha", BlendMode::Alpha)
		.value("BeerLambert", BlendMode::BeerLambert)
		.export_values();
	c.def(py::init<>())
		.def_property("blendMode",
			&Blending::blendMode,
			&Blending::setBlendMode);
}

std::optional<int> renderer::Blending::getBatches(const GlobalSettings& s) const
{
	return {};
}

std::string renderer::Blending::getDefines(const GlobalSettings& s) const
{
	switch (blendMode_)
	{
	case BlendMode::Alpha: return "#define BLENDING_MODE 0\n";
	case BlendMode::BeerLambert: return "#define BLENDING_MODE 1\n";
	default: throw std::runtime_error("Unknown blend mode");
	}
}

std::vector<std::string> renderer::Blending::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_blending.cuh" };
}

std::string renderer::Blending::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "";
}

std::string renderer::Blending::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::Blending";
}

void renderer::Blending::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
	//nothing to do
}
