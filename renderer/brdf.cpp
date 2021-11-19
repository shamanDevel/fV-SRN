#include "brdf.h"
#include <magic_enum.hpp>
#include <sstream>
#include <glm/gtx/transform.hpp>

#include "camera.h"
#include "helper_math.cuh"
#include "json_utils.h"
#include "pytorch_utils.h"
#include "renderer_tensor.cuh"
#include "renderer_commons.cuh"

renderer::BRDFLambert::BRDFLambert()
	: enableMagnitudeScaling_(false)
	, enablePhong_(false)
	, lightFollowsCamera_(true)
	, lightType_(LightType::Directional)
	, showLight_(false)
#if RENDERER_OPENGL_SUPPORT==1
	, lightMesh_(MeshCpu::createCube())
	, lightShader_("PassThrough.vs", "SimpleDiffuse.fs")
#endif
{
}

std::string renderer::BRDFLambert::getName() const
{
	return "Lambert";
}

bool renderer::BRDFLambert::drawUI(UIStorage_t& storage)
{
	bool changed = false;
	
	//UI
	ImGui::PushID("BRDFLambert");
	
	if (ImGui::Checkbox("Magnitude Scaling", &enableMagnitudeScaling_))
		changed = true;
	if (enableMagnitudeScaling_)
	{
		double* magnitudeScaling = enforceAndGetScalar<double>(magnitudeScaling_, 1.0);
		if (ImGui::SliderDouble("Scaling", magnitudeScaling, 0.0, 10.0, "%.6f", 10))
			changed = true;
	}

	if (ImGui::Checkbox("Phong Shading", &enablePhong_))
		changed = true;
	if (enablePhong_)
	{
		double* ambient = enforceAndGetScalar<double>(ambient_, 1.0);
		if (ImGui::SliderDouble("Ambient", ambient, 0.0, 1.0))
			changed = true;

		double* specular = enforceAndGetScalar<double>(specular_, 1.0);
		if (ImGui::SliderDouble("Specular", specular, 0.0, 1.0))
			changed = true;

		double* magnitudeCenter = enforceAndGetScalar<double>(magnitudeCenter_, 1.0);
		if (ImGui::SliderDouble("Magnitude Center", magnitudeCenter, 0.0, 1.0))
			changed = true;

		double* magnitudeRadius = enforceAndGetScalar<double>(magnitudeRadius_, 1.0);
		if (ImGui::SliderDouble("Magnitude Radius", magnitudeRadius, 0.0, 1.0))
			changed = true;

		int* specularExponent = enforceAndGetScalar<int>(specularExponent_, 1);
		if (ImGui::SliderInt("Specular Exponent", specularExponent, 1, 32))
			changed = true;

		const auto& lightTypeName = magic_enum::enum_name(lightType_);
		const auto lightTypeCount = magic_enum::enum_count<LightType>();
		if (ImGui::SliderInt("Light", reinterpret_cast<int*>(&lightType_), 0, lightTypeCount - 1,
			lightTypeName.data()))
			changed = true;
		if (ImGui::Checkbox("Follow Camera", &lightFollowsCamera_))
			changed = true;
		int flags = lightFollowsCamera_ ? ImGuiInputTextFlags_ReadOnly : 0;
		if (lightType_ == LightType::Point) {
			double3* lightPosition = enforceAndGetScalar<double3>(lightPosition_, make_double3(0));
			if (ImGui::InputDouble3("Position", &lightPosition->x, "%.3f", flags))
				changed = true;
		}
		if (lightType_ == LightType::Directional) {
			double3* lightDirection = enforceAndGetScalar<double3>(lightDirection_, make_double3(0));
			if (ImGui::InputDouble3("Direction", &lightDirection->x, "%.3f", flags))
				changed = true;
		}
		if (ImGui::Checkbox("Show Light", &showLight_))
			changed = true;
	}

	ImGui::PopID();
	return changed;
}

void renderer::BRDFLambert::load(const nlohmann::json& json, const ILoadingContext* context)
{
	enableMagnitudeScaling_ = json.value("enableMagnitudeScaling", false);
	*enforceAndGetScalar<double>(magnitudeScaling_) = json.value("magnitudeScaling", 1.0);
	
	enablePhong_ = json.value("enablePhong", false);
	*enforceAndGetScalar<double>(ambient_) = json.value("ambient", 1.0);
	*enforceAndGetScalar<double>(specular_) = json.value("specular", 1.0);
	*enforceAndGetScalar<double>(magnitudeCenter_) = json.value("magnitudeCenter", 1.0);
	*enforceAndGetScalar<double>(magnitudeRadius_) = json.value("magnitudeRadius", 1.0);
	*enforceAndGetScalar<int>(specularExponent_) = json.value("specularExponent", 1);

	lightFollowsCamera_ = json.value("lightFollowsCamera", true);
	lightType_ = magic_enum::enum_cast<LightType>(json.value("lightType", "")).
		value_or(LightType::Directional);
	*enforceAndGetScalar<double3>(lightPosition_) = json.value("lightPosition", make_double3(0));
	*enforceAndGetScalar<double3>(lightDirection_) = json.value("lightDirection", make_double3(0));
}

void renderer::BRDFLambert::save(nlohmann::json& json, const ISavingContext* context) const
{
	
	json["enableMagnitudeScaling"] = enableMagnitudeScaling_;
	json["magnitudeScaling"] = *getScalarOrThrow<double>(magnitudeScaling_);
	
	json["enablePhong"] = enablePhong_;
	json["ambient"] = *getScalarOrThrow<double>(ambient_);
	json["specular"] = *getScalarOrThrow<double>(specular_);
	json["magnitudeCenter"] = *getScalarOrThrow<double>(magnitudeCenter_);
	json["magnitudeRadius"] = *getScalarOrThrow<double>(magnitudeRadius_);
	json["specularExponent"] = *getScalarOrThrow<int>(specularExponent_);

	json["lightFollowsCamera"] = lightFollowsCamera_;
	json["lightType"] = magic_enum::enum_name(lightType_);
	json["lightPosition"] = *getScalarOrThrow<double3>(lightPosition_);
	json["lightDirection"] = *getScalarOrThrow<double3>(lightDirection_);
}

void renderer::BRDFLambert::registerPybindModule(pybind11::module& m)
{
	namespace py = pybind11;
	py::class_<BRDFLambert, std::shared_ptr<BRDFLambert>> c(m, "BRDFLambert");
	py::enum_<LightType>(c, "LightType")
		.value("Point", LightType::Point)
		.value("Directional", LightType::Directional)
		.export_values();
	c.def(py::init<>())
		.def_readwrite("enable_magnitude_scaling", &BRDFLambert::enableMagnitudeScaling_)
		.def_readonly("magnitude_scaling", &BRDFLambert::enableMagnitudeScaling_)
		.def_readwrite("enable_phong", &BRDFLambert::enablePhong_)
		.def_readonly("ambient", &BRDFLambert::ambient_)
		.def_readonly("specular", &BRDFLambert::specular_)
		.def_readonly("magnitude_center", &BRDFLambert::magnitudeCenter_)
		.def_readonly("magnitude_radius", &BRDFLambert::magnitudeRadius_)
		.def_readonly("specular_exponent", &BRDFLambert::specularExponent_)
		.def_readwrite("light_follows_camera", &BRDFLambert::lightFollowsCamera_)
		.def_readwrite("light_type", &BRDFLambert::lightType_)
		.def_readonly("light_position", &BRDFLambert::lightPosition_)
		.def_readonly("light_direction", &BRDFLambert::lightDirection_);
}

void renderer::BRDFLambert::prepareRendering(GlobalSettings& s) const
{
	if (enableMagnitudeScaling_ || enablePhong_) {
		s.volumeShouldProvideNormals = true;
		//std::cout << "BRDF: magnitude scaling or phong enabled, request that the volume provides normals" << std::endl;
	}
}

std::optional<int> renderer::BRDFLambert::getBatches(const GlobalSettings& s) const
{
	//either all tensors or all scalars
	if (std::holds_alternative<double>(magnitudeScaling_.value))
	{
		//scalar
		TORCH_CHECK(std::holds_alternative<double>(ambient_.value),
			"Either all tensors or all scalars: magnitudeScaling is scalar, so all other parameters must be as well");
		TORCH_CHECK(std::holds_alternative<double>(specular_.value),
			"Either all tensors or all scalars: magnitudeScaling is scalar, so all other parameters must be as well");
		TORCH_CHECK(std::holds_alternative<double>(magnitudeCenter_.value),
			"Either all tensors or all scalars: magnitudeScaling is scalar, so all other parameters must be as well");
		TORCH_CHECK(std::holds_alternative<double>(magnitudeRadius_.value),
			"Either all tensors or all scalars: magnitudeScaling is scalar, so all other parameters must be as well");
		TORCH_CHECK(std::holds_alternative<int>(specularExponent_.value),
			"Either all tensors or all scalars: magnitudeScaling is scalar, so all other parameters must be as well");
		TORCH_CHECK(std::holds_alternative<double3>(lightPosition_.value),
			"Either all tensors or all scalars: magnitudeScaling is scalar, so all other parameters must be as well");
		TORCH_CHECK(std::holds_alternative<double3>(lightDirection_.value),
			"Either all tensors or all scalars: magnitudeScaling is scalar, so all other parameters must be as well");
		return {};
	} else
	{
		//tensor
		int batch = 1;
		std::string lastBatchName;
		const auto dtype = std::get<torch::Tensor>(magnitudeScaling_.value).scalar_type();
		TORCH_CHECK(dtype == c10::kFloat || dtype == c10::kDouble, "dtype must be float or double, but is ", dtype);

#define CHECK_BATCH(param, dt)	\
		do	\
		{	\
			TORCH_CHECK(std::holds_alternative<torch::Tensor>(param.value),	\
				"Either all tensors or all scalars: magnitudeScaling is a tensor, so ", C10_STRINGIZE(param), " must be as well");	\
			const torch::Tensor& t = std::get<torch::Tensor>(param.value);	\
			CHECK_CUDA(t, true);	\
			CHECK_DIM(t, 1);			\
			CHECK_DTYPE(t, dt);	\
			int b = t.size(0);		\
			if (b>1)					\
			{							\
				if (batch > 1 && batch != b)	\
					TORCH_CHECK(false, "Batch sizes must agree. Batch size of ", C10_STRINGIZE(param), " is ", b, ", but ", lastBatchName, " already has a batch size of ", batch);	\
				batch = b;	\
				lastBatchName = C10_STRINGIZE(param);	\
			}	\
		} while (false)

		CHECK_BATCH(magnitudeScaling_, dtype);
		CHECK_BATCH(ambient_, dtype);
		CHECK_BATCH(specular_, dtype);
		CHECK_BATCH(magnitudeCenter_, dtype);
		CHECK_BATCH(magnitudeRadius_, dtype);
		CHECK_BATCH(specularExponent_, c10::kInt);

#define CHECK_BATCH3(param)	\
		do    \
		{	  \
			TORCH_CHECK(std::holds_alternative<torch::Tensor>(param.value),	\
			            "Either all tensors or all scalars: magnitudeScaling is a tensor, so ",	\
			            C10_STRINGIZE(param), " must be as well");	\
			const torch::Tensor& t = std::get<torch::Tensor>(param.value);	\
			CHECK_CUDA(t, true);		\
			CHECK_DIM(t, 2);			\
			CHECK_DTYPE(t, dtype);	\
			CHECK_SIZE(t, 1, 3);	\
			int b = t.size(0);			\
			if (b > 1)					\
			{							\
				if (batch > 1 && batch != b)	\
					TORCH_CHECK(false, "Batch sizes must agree. Batch size of ", C10_STRINGIZE(param), " is ", b,	\
				            ", but ", lastBatchName, " already has a batch size of ", batch);	\
				batch = b;	\
				lastBatchName = C10_STRINGIZE(param);	\
			}	\
		} while (false)

		CHECK_BATCH3(lightPosition_);
		CHECK_BATCH3(lightDirection_);

#undef CHECK_BATCH
#undef CHECK_BATCH3

		if (batch > 1)
			return batch;
		return {};
	}
}

std::string renderer::BRDFLambert::getDefines(const GlobalSettings& s) const
{
	std::stringstream ss;

	if (std::holds_alternative<torch::Tensor>(magnitudeScaling_.value))
		ss << "#define BRDF_LAMBERT__BATCHED\n";
	if (enableMagnitudeScaling_)
		ss << "#define BRDF_LAMBERT_ENABLE_MAGNITUDE_SCALING\n";
	if (enablePhong_)
		ss << "#define BRDF_LAMBERT_ENABLE_PHONG\n";
	switch (lightType_)
	{
	case LightType::Point:
	{
		ss << "#define BRDF_LAMBERT_LIGHT_TYPE 0\n";
	} break;
	case LightType::Directional:
	{
		ss << "#define BRDF_LAMBERT_LIGHT_TYPE 1\n";
	} break;
	}
	
	return ss.str();
}

std::vector<std::string> renderer::BRDFLambert::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_brdf_lambert.cuh" };
}

std::string renderer::BRDFLambert::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "brdfLambertParameters";
}

std::string renderer::BRDFLambert::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::BRDFLambert";
}

void renderer::BRDFLambert::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
	if (lightFollowsCamera_)
		updateLightFromCamera(s);

	if (std::holds_alternative<double>(magnitudeScaling_.value))
	{
		//scalar
		double magnitudeScaling = *getScalarOrThrow<double>(magnitudeScaling_);
		double ambient = *getScalarOrThrow<double>(ambient_);
		double specular = *getScalarOrThrow<double>(specular_);
		double magnitudeCenter = *getScalarOrThrow<double>(magnitudeCenter_);
		double magnitudeRadius = *getScalarOrThrow<double>(magnitudeRadius_);
		int specularExponent = *getScalarOrThrow<int>(specularExponent_);
		double3 lightPosition = *getScalarOrThrow<double3>(lightPosition_);
		double3 lightDirection = *getScalarOrThrow<double3>(lightDirection_);
		double3 lightParameter = lightType_ == LightType::Point ? lightPosition : lightDirection;
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "BRDFLambert", [&]()
			{
				using scalar3 = typename ::kernel::scalar_traits<scalar_t>::real3;
				struct Parameters
				{
					scalar_t magnitudeScaling;
					scalar_t ambient;
					scalar_t specular;
					scalar_t magnitudeCenter;
					scalar_t magnitudeRadius;
					int    specularExponent;
					scalar3  lightParameter;
				} p;
				p.magnitudeScaling = static_cast<scalar_t>(magnitudeScaling);
				p.ambient = static_cast<scalar_t>(ambient);
				p.specular = static_cast<scalar_t>(specular);
				p.magnitudeCenter = static_cast<scalar_t>(magnitudeCenter);
				p.magnitudeRadius = static_cast<scalar_t>(magnitudeRadius);
				p.specularExponent = specularExponent;
				p.lightParameter = kernel::cast3<scalar_t>(lightParameter);
				CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
			});
	}
	else {
		//tensor
		const torch::Tensor magnitudeScaling = std::get<torch::Tensor>(magnitudeScaling_.value);
		const torch::Tensor ambient = std::get<torch::Tensor>(ambient_.value);
		const torch::Tensor specular = std::get<torch::Tensor>(specular_.value);
		const torch::Tensor magnitudeCenter = std::get<torch::Tensor>(magnitudeCenter_.value);
		const torch::Tensor magnitudeRadius = std::get<torch::Tensor>(magnitudeRadius_.value);
		const torch::Tensor specularExponent = std::get<torch::Tensor>(specularExponent_.value);
		const torch::Tensor lightPosition = std::get<torch::Tensor>(lightPosition_.value);
		const torch::Tensor lightDirection = std::get<torch::Tensor>(lightDirection_.value);
		const torch::Tensor lightParameter = lightType_ == LightType::Point ? lightPosition : lightDirection;
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "BRDFLambert", [&]()
			{
				using scalar3 = typename ::kernel::scalar_traits<scalar_t>::real3;
				struct Parameters
				{
					kernel::Tensor1Read<scalar_t> magnitudeScaling;
					kernel::Tensor1Read<scalar_t> ambient;
					kernel::Tensor1Read<scalar_t> specular;
					kernel::Tensor1Read<scalar_t> magnitudeCenter;
					kernel::Tensor1Read<scalar_t> magnitudeRadius;
					kernel::Tensor1Read<int>    specularExponent;
					kernel::Tensor2Read<scalar_t> lightParameter;
				} p;
				p.magnitudeScaling = accessor<kernel::Tensor1Read<scalar_t>>(magnitudeScaling);
				p.ambient = accessor<kernel::Tensor1Read<scalar_t>>(ambient);
				p.specular = accessor<kernel::Tensor1Read<scalar_t>>(specular);
				p.magnitudeCenter = accessor<kernel::Tensor1Read<scalar_t>>(magnitudeCenter);
				p.magnitudeRadius = accessor<kernel::Tensor1Read<scalar_t>>(magnitudeRadius);
				p.specularExponent = accessor<kernel::Tensor1Read<int>>(specularExponent);
				p.lightParameter = accessor<kernel::Tensor2Read<scalar_t>>(lightParameter);
				CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
			});
	}
}

bool renderer::BRDFLambert::updateLightFromCamera(const GlobalSettings& s)
{
	ICamera_ptr cam = std::dynamic_pointer_cast<ICamera>(s.root->getSelectedModuleForTag(ICamera::Tag()));
	if (!cam)
	{
		throw std::runtime_error("BRDFLambert: No camera found, light can't follow the camera");
	}
	double3 newPosition = cam->getOrigin(0);
	double3 newDirection = cam->getFront(0);
	double3* pos = enforceAndGetScalar<double3>(lightPosition_);
	double3* dir = enforceAndGetScalar<double3>(lightDirection_);
	bool changed = any(newPosition != *pos) || any(newDirection != *dir);
	*pos = newPosition;
	*dir = newDirection;
	return changed;
}

bool renderer::BRDFLambert::hasRasterizing() const
{
	return showLight_ || IBRDF::hasRasterizing();
}

void renderer::BRDFLambert::performRasterization(const RasterizingContext* context)
{
    IBRDF::performRasterization(context);

	//render light bulb
	if (showLight_)
	{
#if RENDERER_OPENGL_SUPPORT==1
		double3 lightPos;
		if (lightType_ == LightType::Point)
			lightPos = *enforceAndGetScalar<double3>(lightPosition_);
		else
			lightPos = -*enforceAndGetScalar<double3>(lightDirection_) * 1.5f;
		float lightRadius = 0.1f;

		glDisable(GL_CULL_FACE);
		lightShader_.use();

		glm::mat4 modelMatrix = glm::translate(glm::vec3(lightPos.x, lightPos.y, lightPos.z))
			* glm::scale(glm::vec3(lightRadius, lightRadius, lightRadius));
		lightShader_.setMat4("model", modelMatrix);
		lightShader_.setMat4("view", context->view);
		lightShader_.setMat4("projection", context->projection);
		lightShader_.setVec3("ambientColor", 0.5f, 0.5, 0.5f);
		lightShader_.setVec3("diffuseColor", 0.5f, 0.5f, 0.5f);
		lightShader_.setVec3("lightDirection", 1, 0, 0);
		lightShader_.setVec3("cameraOrigin", context->origin);

		lightMesh_.drawIndexed();
#else
		throw std::runtime_error("OpenGL-support disabled, can't render light source visualization");
#endif
	}
}
