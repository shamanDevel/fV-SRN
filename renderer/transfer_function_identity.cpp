#include "transfer_function.h"

#include <vector_functions.h>
#include "renderer_tensor.cuh"
#include "renderer_commons.cuh"
#include "pytorch_utils.h"

renderer::TransferFunctionIdentity::TransferFunctionIdentity()
	: scaleAbsorptionEmission_(make_double2(10,1))
{
}

renderer::TransferFunctionIdentity::~TransferFunctionIdentity()
{
}

std::string renderer::TransferFunctionIdentity::getName() const
{
	return "Identity";
}

bool renderer::TransferFunctionIdentity::drawUI(UIStorage_t& storage)
{
	bool changed = false;
	if (ImGui::SliderDouble("Absorption##TransferFunctionIdentity",
		&enforceAndGetScalar<double2>(scaleAbsorptionEmission_)->x, 0, 100, "%.3f", 2))
		changed = true;
	if (ImGui::SliderDouble("Emission##TransferFunctionIdentity",
		&enforceAndGetScalar<double2>(scaleAbsorptionEmission_)->y, 0, 100, "%.3f", 2))
		changed = true;
	return changed;
}

void renderer::TransferFunctionIdentity::load(const nlohmann::json& json, const ILoadingContext* context)
{
	enforceAndGetScalar<double2>(scaleAbsorptionEmission_)->x = json.value("absorptionScaling", 1.0);
	enforceAndGetScalar<double2>(scaleAbsorptionEmission_)->y = json.value("emissionScaling", 1.0);
}

void renderer::TransferFunctionIdentity::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["absorptionScaling"] = getScalarOrThrow<double2>(scaleAbsorptionEmission_)->x;
	json["emissionScaling"] = getScalarOrThrow<double2>(scaleAbsorptionEmission_)->y;
}

void renderer::TransferFunctionIdentity::registerPybindModule(pybind11::module& m)
{
	ITransferFunction::registerPybindModule(m);

	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<TransferFunctionIdentity, ITransferFunction, std::shared_ptr<TransferFunctionIdentity>>(m, "TransferFunctionIdentity")
		.def(py::init<>())
		.def_readonly("absorption_emission", &TransferFunctionIdentity::scaleAbsorptionEmission_,
			py::doc("double2 where x=scale absorption, y=scale emission"));
}

std::optional<int> renderer::TransferFunctionIdentity::getBatches(const GlobalSettings& s) const
{
	bool batched = std::holds_alternative<torch::Tensor>(scaleAbsorptionEmission_.value);
	if (batched)
	{
		torch::Tensor tf = std::get<torch::Tensor>(scaleAbsorptionEmission_.value);
		CHECK_CUDA(tf, true);
		CHECK_DIM(tf, 3);
		CHECK_SIZE(tf, 1, 1);
		CHECK_SIZE(tf, 2, 2);
		return tf.size(0);
	}
	return {};
}

std::vector<std::string> renderer::TransferFunctionIdentity::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_tf_identity.cuh" };
}

std::string renderer::TransferFunctionIdentity::getDefines(const GlobalSettings& s) const
{
	bool batched = std::holds_alternative<torch::Tensor>(scaleAbsorptionEmission_.value);
	if (batched)
		return "#define TRANSFER_FUNCTION_IDENTITY__BATCHED\n";
	else
		return "";
}

std::string renderer::TransferFunctionIdentity::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "transferFunctionIdentityParameters";
}

std::string renderer::TransferFunctionIdentity::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::TransferFunctionIdentity";
}

void renderer::TransferFunctionIdentity::fillConstantMemoryTF(
	const GlobalSettings& s, CUdeviceptr ptr, double stepsize, CUstream stream)
{
	bool batched = std::holds_alternative<torch::Tensor>(scaleAbsorptionEmission_.value);
	if (batched)
	{
		torch::Tensor tf = std::get<torch::Tensor>(scaleAbsorptionEmission_.value);
		CHECK_CUDA(tf, true);
		CHECK_DIM(tf, 3);
		CHECK_SIZE(tf, 1, 1);
		CHECK_SIZE(tf, 2, 2);
		CHECK_DTYPE(tf, s.scalarType);
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "TransferFunctionIdentity", [&]()
			{
				struct Parameters
				{
					::kernel::Tensor3Read<scalar_t> scaleAbsorptionEmission;
				} p;
				p.scaleAbsorptionEmission = accessor<::kernel::Tensor3Read<scalar_t>>(tf);
				CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
			});
	} else
	{
		double2 value = std::get<double2>(scaleAbsorptionEmission_.value);
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "TransferFunctionIdentity", [&]()
			{
				using real2 = typename ::kernel::scalar_traits<scalar_t>::real2;
				struct Parameters
				{
					scalar_t scaleAbsorption;
					scalar_t scaleEmission;
				} p;
				p.scaleAbsorption = static_cast<scalar_t>(value.x);
				p.scaleEmission = static_cast<scalar_t>(value.y);
				CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
			});
	}
}

double4 renderer::TransferFunctionIdentity::evaluate(double density) const
{
	double2 value = std::get<double2>(scaleAbsorptionEmission_.value);
	double scaleAbsorption = value.x;
	double scaleEmission = value.y;
	return make_double4(
		density * scaleEmission, //red
		density * scaleEmission, //green
		density * scaleEmission, //blue
		density * scaleAbsorption);
}

double renderer::TransferFunctionIdentity::getMaxAbsorption() const
{
	bool batched = std::holds_alternative<torch::Tensor>(scaleAbsorptionEmission_.value);
	if (batched)
	{
		torch::Tensor tf = std::get<torch::Tensor>(scaleAbsorptionEmission_.value);
		return torch::max(tf.slice(1, 0, 1)).item().toDouble();
	}
	else
	{
		double2 value = std::get<double2>(scaleAbsorptionEmission_.value);
		return value.x;
	}
}

