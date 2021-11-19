#include "phase_function.h"

#include "pytorch_utils.h"
#include "renderer_tensor.cuh"

void renderer::IPhaseFunction::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<IPhaseFunction, std::shared_ptr<IPhaseFunction>> c(m, "IPhaseFunction");
}

renderer::PhaseFunctionHenyeyGreenstein::PhaseFunctionHenyeyGreenstein()
	: g_(0.0)
{
}

std::string renderer::PhaseFunctionHenyeyGreenstein::getName() const
{
	return "Henyey-Greenstein";
}

bool renderer::PhaseFunctionHenyeyGreenstein::drawUI(UIStorage_t& storage)
{
	bool changed = false;
	double* g = getScalarOrThrow<double>(g_);
	if (ImGui::SliderDouble("g##HenyeyGreenstein", g, -0.99, +0.99))
		changed = true;
	return changed;
}

void renderer::PhaseFunctionHenyeyGreenstein::load(const nlohmann::json& json, const ILoadingContext* context)
{
	*enforceAndGetScalar<double>(g_) = json.value("g", 1.0);
}

void renderer::PhaseFunctionHenyeyGreenstein::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["g"] = *getScalarOrThrow<double>(g_);
}

void renderer::PhaseFunctionHenyeyGreenstein::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;

	IPhaseFunction::registerPybindModule(m);
	
	namespace py = pybind11;
	py::class_<PhaseFunctionHenyeyGreenstein, IPhaseFunction,
		std::shared_ptr<PhaseFunctionHenyeyGreenstein>> c(m, "PhaseFunctionHenyeyGreenstein");
	c.def(py::init())
		.def_readonly("g", &PhaseFunctionHenyeyGreenstein::g_,
			py::doc("The scattering factor 'g' in (-1,+1)."));
}

void renderer::PhaseFunctionHenyeyGreenstein::prepareRendering(GlobalSettings& s) const
{
	//do nothing
}

std::optional<int> renderer::PhaseFunctionHenyeyGreenstein::getBatches(const GlobalSettings& s) const
{
	if (std::holds_alternative<double>(g_.value))
		return {};
	else
	{
		torch::Tensor g = std::get<torch::Tensor>(g_.value);
		const auto dtype = g.scalar_type();
		TORCH_CHECK(dtype == c10::kFloat || dtype == c10::kDouble, "dtype must be float or double, but is ", dtype);
		CHECK_CUDA(g, true);
		CHECK_DIM(g, 1);
		return g.size(0);
	}
}

std::string renderer::PhaseFunctionHenyeyGreenstein::getDefines(const GlobalSettings& s) const
{
	if (std::holds_alternative<torch::Tensor>(g_.value))
		return "#define PHASE_FUNCTION_HENYEY_GREENSTEIN_BATCHED\n";
	return "";
}

std::vector<std::string> renderer::PhaseFunctionHenyeyGreenstein::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_phase_function.cuh" };
}

std::string renderer::PhaseFunctionHenyeyGreenstein::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "phaseFunctionHenyeyGreensteinParameters";
}

std::string renderer::PhaseFunctionHenyeyGreenstein::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::PhaseFunctionHenyeyGreenstein";
}

void renderer::PhaseFunctionHenyeyGreenstein::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr,
	CUstream stream)
{
	if (std::holds_alternative<double>(g_.value))
	{
		//scalar
		double g = *getScalarOrThrow<double>(g_);
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "PhaseFunctionHenyeyGreenstein", [&]()
			{
				struct Parameters
				{
					scalar_t g;
				} p;
				p.g = static_cast<scalar_t>(g);
				CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
			});
	}
	else
	{
		const torch::Tensor g = std::get<torch::Tensor>(g_.value);
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "PhaseFunctionHenyeyGreenstein", [&]()
			{
				struct Parameters
				{
					kernel::Tensor1Read<scalar_t> g;
				} p;
				p.g = accessor<kernel::Tensor1Read<scalar_t>>(g);
				CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
			});
	}
}

renderer::PhaseFunctionRayleigh::PhaseFunctionRayleigh()
{
}

std::string renderer::PhaseFunctionRayleigh::getName() const
{
	return "Rayleigh";
}

bool renderer::PhaseFunctionRayleigh::drawUI(UIStorage_t& storage)
{
	return false; //nothing to do
}

void renderer::PhaseFunctionRayleigh::load(const nlohmann::json& json, const ILoadingContext* fetcher)
{
	//nothing to do
}

void renderer::PhaseFunctionRayleigh::save(nlohmann::json& json, const ISavingContext* context) const
{
	//nothing to do
}

void renderer::PhaseFunctionRayleigh::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;

	IPhaseFunction::registerPybindModule(m);

	namespace py = pybind11;
	py::class_<PhaseFunctionRayleigh, IPhaseFunction,
		std::shared_ptr<PhaseFunctionRayleigh>> c(m, "PhaseFunctionRayleigh");
	c.def(py::init());
}

void renderer::PhaseFunctionRayleigh::prepareRendering(GlobalSettings& s) const
{
	//nothing to do
}

std::optional<int> renderer::PhaseFunctionRayleigh::getBatches(const GlobalSettings& s) const
{
	return {}; //nothing to do
}

std::string renderer::PhaseFunctionRayleigh::getDefines(const GlobalSettings& s) const
{
	return ""; //nothing to do
}

std::vector<std::string> renderer::PhaseFunctionRayleigh::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_phase_function.cuh" };
}

std::string renderer::PhaseFunctionRayleigh::getConstantDeclarationName(const GlobalSettings& s) const
{
	return ""; //nothing to do
}

std::string renderer::PhaseFunctionRayleigh::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::PhaseFunctionRayleigh";
}

void renderer::PhaseFunctionRayleigh::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
	//nothing to do
}
