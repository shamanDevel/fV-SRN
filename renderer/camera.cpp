#include "camera.h"

#include <glm/glm.hpp>
#include <magic_enum.hpp>
#include <c10/cuda/CUDAStream.h>
#include <cuMat/src/Macros.h>

#include "json_utils.h"
#include "helper_math.cuh"
#include "kernel_loader.h"
#include "pytorch_utils.h"
#include "renderer_tensor.cuh"
#include "renderer_commons.cuh"

#include "glm/gtc/matrix_transform.hpp"

const char* renderer::CameraOnASphere::OrientationNames[6] = {
	"Xp", "Xm", "Yp", "Ym", "Zp", "Zm"
};
const float3 renderer::CameraOnASphere::OrientationUp[6] = {
	float3{1,0,0}, float3{-1,0,0},
	float3{0,1,0}, float3{0,-1,0},
	float3{0,0,1}, float3{0,0,-1}
};
const int3 renderer::CameraOnASphere::OrientationPermutation[6] = {
	int3{2,-1,-3}, int3{-2, 1, 3},
	int3{1,2,3}, int3{-1,-2,-3},
	int3{-3,-1,2}, int3{3,1,-2}
};
const bool renderer::CameraOnASphere::OrientationInvertYaw[6] = {
	false, true, true, false, true, false
};
const bool renderer::CameraOnASphere::OrientationInvertPitch[6] = {
	false, false, false, false, false, false
};

std::tuple<torch::Tensor, torch::Tensor> renderer::ICamera::generateRays(
	int width, int height, bool doublePrecision, CUstream stream)
{
	GlobalSettings s{};
	s.scalarType = doublePrecision ? GlobalSettings::kDouble : GlobalSettings::kFloat;

	//kernel
	this->setAspectRatio(double(width) / height);
	this->prepareRendering(s);
	const std::string kernelName = "CameraGenerateRayKernel";
	std::vector<std::string> constantNames;
	if (const auto c = getConstantDeclarationName(s); !c.empty())
		constantNames.push_back(c);
	std::stringstream extraSource;
	extraSource << "#define KERNEL_DOUBLE_PRECISION "
		<< (s.scalarType == IKernelModule::GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
	extraSource << getDefines(s) << "\n";
	for (const auto& i : getIncludeFileNames(s))
		extraSource << "#include \"" << i << "\"\n";
	extraSource << "#define IMAGE_EVALUATOR__CAMERA_T " <<
		getPerThreadType(s) << "\n";
	extraSource << "#include \"renderer_camera_kernels.cuh\"\n";
	const auto fun = KernelLoader::Instance().getKernelFunction(
		kernelName, extraSource.str(), constantNames, false, false).value();
	if (auto c = getConstantDeclarationName(s); !c.empty())
	{
		CUdeviceptr ptr = fun.constant(c);
		fillConstantMemory(s, ptr, stream);
	}
	
	//output tensors
	int batches = getBatches(s).value_or(1);
	auto rayStart = torch::empty({ batches, height, width, 3 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	auto rayDir = torch::empty({ batches, height, width, 3 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));

	//launch kernel
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(width * height * batches, fun.bestBlockSize())),
		fun.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(width),
		static_cast<unsigned int>(height),
		static_cast<unsigned int>(batches) };
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "ICamera::generateRays", [&]()
		{
			const auto accStart = accessor< ::kernel::Tensor4RW<scalar_t>>(rayStart);
			const auto accDir = accessor< ::kernel::Tensor4RW<scalar_t>>(rayDir);
			const void* args[] = { &virtual_size, &accStart, &accDir};
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, fun.bestBlockSize(), 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return printError(result, kernelName);
			return true;
		});
	if (!success) throw std::runtime_error("Error during rendering!");

	return std::make_tuple(rayStart, rayDir);
}

void renderer::ICamera::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;

	////trampoline class
	//class PyCamera : public ICamera
	//{
	//public:
	//	using ICamera::ICamera;

	//};

	namespace py = pybind11;
	py::class_<ICamera, std::shared_ptr<ICamera>>(m, "ICamera")
		.def_readonly("aspect_ratio", &ICamera::aspectRatio_)
		.def("get_origin", &ICamera::getOrigin)
		.def("get_front", &ICamera::getFront)
		.def("generate_rays", [](ICamera* self, int width, int height, bool doublePrecision)
			{
				return self->generateRays(width, height, doublePrecision, c10::cuda::getCurrentCUDAStream());
			},
			py::doc("Generates the ray start and direction for the current camera."),
				py::arg("width"), py::arg("height"), py::arg("double_precision") = false)
		.def("get_parameters", &ICamera::getParameters)
		.def("set_parameters", &ICamera::setParameters)
		//.def("get_parameters", [](ICamera* self) {return self->getParameters(); })
		//.def("set_parameters", [](ICamera* self, const torch::Tensor& t) {self->setParameters(t); })
		;
	
}

renderer::CameraOnASphere::CameraOnASphere()
	: orientation_(Ym)
	, center_(make_double3(0))
	, pitchYawDistance_(make_double3(0, 0, 0))
	, fovYradians_(glm::radians(45.0))
	, matrixFromExternalSource_(false)
{
}


torch::Tensor renderer::CameraOnASphere::getParameters()
{
	updateCameraMatrix(c10::kFloat);
	return cachedCameraMatrix_;
}

void renderer::CameraOnASphere::setParameters(const torch::Tensor& parameters)
{
	if (!parameters.defined() || parameters.numel()==0)
	{
		//revert to internal parameters
		matrixFromExternalSource_ = false;
		return;
	}

	TORCH_CHECK(parameters.dim() == 3, "camera matrix must be of shape B,3,3, but is of shape", parameters.sizes());
	TORCH_CHECK(parameters.size(1) == 3, "camera matrix must be of shape B,3,3, but is of shape", parameters.sizes());
	TORCH_CHECK(parameters.size(2) == 3, "camera matrix must be of shape B,3,3, but is of shape", parameters.sizes());
	TORCH_CHECK(parameters.is_cuda(), "camera matrix must reside in CUDA memory");

	cachedCameraMatrix_ = parameters;
	matrixFromExternalSource_ = true;
}


std::string renderer::CameraOnASphere::getName() const
{
	return "Sphere";
}

bool renderer::CameraOnASphere::updateUI(UIStorage_t& storage)
{
	bool changed = false;
	double& currentPitch = enforceAndGetScalar<double3>(pitchYawDistance_)->x;
	double& currentYaw = enforceAndGetScalar<double3>(pitchYawDistance_)->y;

	//MOUSE
	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse)
	{

		if (io.MouseDown[0])
		{
			//dragging
			currentPitch = std::max(-80.0, std::min(80.0,
				currentPitch + rotateSpeed_ * io.MouseDelta.y));
			currentYaw += rotateSpeed_ * io.MouseDelta.x;
		}
		//zoom
		float mouseWheel = ImGui::GetIO().MouseWheel;
		zoomValue_ += mouseWheel;

		changed = changed || mouseWheel != 0 || (io.MouseDown[0] && (io.MouseDelta.x != 0 || io.MouseDelta.y != 0));
	}
	return changed;
}

bool renderer::CameraOnASphere::drawUI(UIStorage_t& storage)
{
	bool changed = false;

	double& currentPitch = enforceAndGetScalar<double3>(pitchYawDistance_)->x;
	double& currentYaw = enforceAndGetScalar<double3>(pitchYawDistance_)->y;
	double& currentDistance = enforceAndGetScalar<double3>(pitchYawDistance_)->z;
	double3& center = *enforceAndGetScalar<double3>(center_);
		
	//UI
	ImGui::PushID("CameraOnASphere");
	double fovMin = 0.1, fovMax = 90;
	double fovDegree = glm::degrees(fovYradians_);
	if (ImGui::SliderScalar("FoV", 
		ImGuiDataType_Double, &fovDegree, &fovMin, &fovMax, u8"%.5f\u00b0", 2)) 
	{
		fovYradians_ = glm::radians(fovDegree);
		changed = true;
	}
	if (ImGui::InputDouble3("Camera Center", &center.x, "%.3f"))
		changed = true;

	for (int i = 0; i < 6; ++i) {
		if (ImGui::RadioButton(OrientationNames[i], orientation_ == Orientation(i))) {
			orientation_ = Orientation(i);
			changed = true;
		}
		if (i < 5) ImGui::SameLine();
	}

	double minPitch = -80, maxPitch = +80;
	double currentPitchDegree = glm::degrees(currentPitch);
	if (ImGui::SliderDouble("Pitch", &currentPitchDegree, minPitch, maxPitch, u8"%.5f\u00b0"))
	{
		currentPitch = glm::radians(currentPitchDegree);
		changed = true;
	}
	double currentYawDegree = glm::degrees(currentYaw);
	if (ImGui::InputDouble("Yaw", &currentYawDegree, 0, 0, u8"%.5f\u00b0"))
	{
		currentYaw = glm::radians(currentYawDegree);
		changed = true;
	}

	if (ImGui::InputFloat("Zoom", &zoomValue_)) changed = true;
	ImGui::InputDouble("Distance", &currentDistance, 0, 0, ".3f", ImGuiInputTextFlags_ReadOnly);
	ImGui::PopID();

	currentDistance = baseDistance_ * std::pow(zoomSpeed_, zoomValue_);
	
	//TODO: for other applications, also provide ViewProjectionMatrices
	//compute here

	return changed;
}

void renderer::CameraOnASphere::load(const nlohmann::json& json, const ILoadingContext* context)
{
	orientation_ = magic_enum::enum_cast<Orientation>(json.value("orientation", "")).
		value_or(Orientation::Ym);
	*enforceAndGetScalar<double3>(center_) = json.value("center", make_double3(0));
	enforceAndGetScalar<double3>(pitchYawDistance_)->x =
		json.value("pitch", 0.0);
	enforceAndGetScalar<double3>(pitchYawDistance_)->y =
		json.value("yaw", 0.0);
	enforceAndGetScalar<double3>(pitchYawDistance_)->z =
		json.value("distance", 0.0);
	fovYradians_ = json.value("fovY", glm::radians(45.0));
	zoomValue_ = json.value("zoom", 1);
}

void renderer::CameraOnASphere::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["orientation"] = magic_enum::enum_name(orientation_);
	json["center"] = *getScalarOrThrow<double3>(center_);
	json["pitch"] = getScalarOrThrow<double3>(pitchYawDistance_)->x;
	json["yaw"] = getScalarOrThrow<double3>(pitchYawDistance_)->y;
	json["distance"] = getScalarOrThrow<double3>(pitchYawDistance_)->z;
	json["fovY"] = fovYradians_;
	json["zoom"] = zoomValue_;
}

void renderer::CameraOnASphere::registerPybindModule(pybind11::module& m)
{
	ICamera::registerPybindModule(m);

	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<CameraOnASphere, ICamera, std::shared_ptr<CameraOnASphere>> c(m, "CameraOnASphere");
	py::enum_<Orientation> e(c, "Orientation");
	for (int i = 0; i < magic_enum::enum_count<Orientation>(); ++i)
	{
		Orientation v = magic_enum::enum_value<Orientation>(i);
		e.value(magic_enum::enum_name(v).data(), v);
	}
	e.export_values();
	c.def(py::init<>())
		.def_readwrite("orientation", &CameraOnASphere::orientation_)
		.def_readonly("center", &CameraOnASphere::center_)
		.def_readonly("pitchYawDistance", &CameraOnASphere::pitchYawDistance_);
}

std::optional<int> renderer::CameraOnASphere::getBatches(const GlobalSettings& s) const
{
	if (matrixFromExternalSource_)
	{
		int batches = cachedCameraMatrix_.size(0);
		if (batches > 1)
			return { batches };
		return {};
	}

	int batches = 0;
	bool batched = false;
	if (std::holds_alternative<torch::Tensor>(center_.value))
	{
		const torch::Tensor& t = std::get<torch::Tensor>(center_.value);
		CHECK_CUDA(t, true);
		CHECK_DIM(t, 3);
		CHECK_SIZE(t, 1, 3);
		CHECK_SIZE(t, 2, 3);
		int b = t.size(0);
		batched = b>1;
		batches = t.size(0);
	}
	if (std::holds_alternative<torch::Tensor>(pitchYawDistance_.value))
	{
		const torch::Tensor& t = std::get<torch::Tensor>(pitchYawDistance_.value);
		CHECK_CUDA(t, true);
		CHECK_DIM(t, 3);
		CHECK_SIZE(t, 1, 3);
		CHECK_SIZE(t, 2, 3);
		int b = t.size(0);
		if (batched && b>1)
		{
			TORCH_CHECK(batches == b, 
				"batch dimensions of the center tensor and pitchYawDistance must agree");
		}
		batched = batched || (b>1);
		batches = max(batches, t.size(0));
	}
	if (batched)
		return { batches };
	return {};
}

std::vector<std::string> renderer::CameraOnASphere::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_camera.cuh" };
}

std::string renderer::CameraOnASphere::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "cameraReferenceFrameParameters";
}

std::string renderer::CameraOnASphere::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::CameraReferenceFrame";
}

void renderer::CameraOnASphere::updateCameraMatrix(c10::ScalarType scalarType)
{
	if (matrixFromExternalSource_)
	{
		if (cachedCameraMatrix_.scalar_type() != scalarType)
			cachedCameraMatrix_ = cachedCameraMatrix_.to(scalarType);
		TORCH_CHECK(cachedCameraMatrix_.dim() == 3);
		TORCH_CHECK(cachedCameraMatrix_.is_cuda());
		return;
	}

	GlobalSettings s;
	s.scalarType = scalarType;
	int batches = getBatches(s).value_or(1);
	if (!cachedCameraMatrix_.defined() ||
		cachedCameraMatrix_.scalar_type() != s.scalarType ||
		cachedCameraMatrix_.size(0) != batches)
	{
		cachedCameraMatrix_ = torch::empty(
			{ batches, 3, 3 }, at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	}

	if (std::holds_alternative<double3>(center_.value) &&
		std::holds_alternative<double3>(pitchYawDistance_.value))
	{
		TORCH_CHECK(batches == 1);
		//process on the host
		double3 origin, up; double distance;
		computeParameters(origin, up, distance);
		double3 lookAt = *getScalarOrThrow<double3>(center_);
		double3 front = normalize(lookAt - origin);
		double3 right = normalize(cross(front, up));
		double3 up2 = normalize(cross(right, front));
		//UI cache
		cacheOrigin_ = origin;
		cacheFront_ = front;
		//copy to tensor
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "CameraOnASphere-ComputeReferenceFrame", [&]()
		{
			torch::Tensor matrixHost = torch::empty(
				{ 1, 3, 3 }, at::TensorOptions().dtype(s.scalarType).device(c10::kCPU));
			auto acc = matrixHost.accessor<scalar_t, 3>();
			acc[0][0][0] = static_cast<scalar_t>(origin.x);
			acc[0][0][1] = static_cast<scalar_t>(origin.y);
			acc[0][0][2] = static_cast<scalar_t>(origin.z);
			acc[0][1][0] = static_cast<scalar_t>(right.x);
			acc[0][1][1] = static_cast<scalar_t>(right.y);
			acc[0][1][2] = static_cast<scalar_t>(right.z);
			acc[0][2][0] = static_cast<scalar_t>(up2.x);
			acc[0][2][1] = static_cast<scalar_t>(up2.y);
			acc[0][2][2] = static_cast<scalar_t>(up2.z);
			cachedCameraMatrix_.copy_(matrixHost, false);
			//CU_SAFE_CALL(cuMemcpyHtoDAsync(
			//	reinterpret_cast<CUdeviceptr>(cachedCameraMatrix_.data_ptr()),
			//	matrixHost.data_ptr(),
			//	matrixHost.nbytes(),
			//	stream));
		});
	}
	else
	{
		//at least one batched tensor, process in a separate kernel
		throw std::runtime_error("batched camera parameters not implemented yet");
	}
}


void renderer::CameraOnASphere::fillConstantMemory(
	const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
	//convert from camera on a sphere to reference frame matrix.
	//TODO: don't forget to invert that in the adjoint method
	// and forward method when computing gradients!

	int batches = getBatches(s).value_or(1);
	updateCameraMatrix(s.scalarType);

	//std::cout << "fillConstantMemory, ptr=" << ptr << ", stream=" << stream << std::endl;
	
	//write parameter structure
	RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "CameraOnASphere-Parameters", [&]()
	{
		struct Parameters
		{
			scalar_t fovYRadians;
			scalar_t aspect; //width / height
			::kernel::Tensor3Read<scalar_t> matrix;
		} p;
		p.fovYRadians = static_cast<scalar_t>(fovYradians_);
		p.aspect = static_cast<scalar_t>(aspectRatio_);
		p.matrix = accessor<::kernel::Tensor3Read<scalar_t>>(cachedCameraMatrix_);
		CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
	});
}

double3 renderer::CameraOnASphere::eulerToCartesian(double pitch, double yaw, double distance,
	Orientation orientation)
{
	yaw = !OrientationInvertYaw[orientation] ? -yaw : +yaw;
	pitch = !OrientationInvertPitch[orientation] ? -pitch : +pitch;
	double pos[3];
	pos[0] = std::cos(pitch) * std::cos(yaw) * distance;
	pos[1] = std::sin(pitch) * distance;
	pos[2] = std::cos(pitch) * std::sin(yaw) * distance;
	double pos2[3];
	for (int i = 0; i < 3; ++i)
	{
		int p = (&OrientationPermutation[orientation].x)[i];
		pos2[i] = pos[std::abs(p) - 1] * (p > 0 ? 1 : -1);
	}
	return make_double3(pos2[0], pos2[1], pos2[2]);
}

void renderer::CameraOnASphere::computeParameters(double3& origin, double3& up, double& distance) const
{
	up = make_double3(OrientationUp[orientation_]);

	double3 lookAt = *getScalarOrThrow<double3>(center_);
	double currentPitch = getScalarOrThrow<double3>(pitchYawDistance_)->x;
	double currentYaw = getScalarOrThrow<double3>(pitchYawDistance_)->y;
	distance = getScalarOrThrow<double3>(pitchYawDistance_)->z;

	origin = eulerToCartesian(currentPitch, currentYaw, distance, orientation_) + lookAt;
}

double3 renderer::CameraOnASphere::getOrigin(int batch) const
{
	TORCH_CHECK(batch == 0, "getOrigin is only available for batch=0 (for now)");
	return cacheOrigin_;
}

double3 renderer::CameraOnASphere::getFront(int batch) const
{
	TORCH_CHECK(batch == 0, "getFront is only available for batch=0 (for now)");
	return cacheFront_;
}

void renderer::CameraOnASphere::computeOpenGlMatrices(int width, int height, 
	glm::mat4& viewOut, glm::mat4& projectionOut, glm::mat4& normalOut, glm::vec3& originOut) const
{
	double3 up = make_double3(OrientationUp[orientation_]);
	double3 lookAt = *getScalarOrThrow<double3>(center_);
	double currentPitch = getScalarOrThrow<double3>(pitchYawDistance_)->x;
	double currentYaw = getScalarOrThrow<double3>(pitchYawDistance_)->y;
	double distance = getScalarOrThrow<double3>(pitchYawDistance_)->z;
	double3 origin = eulerToCartesian(currentPitch, currentYaw, distance, orientation_) + lookAt;

	const auto toGLM = [](const double3& v) {return glm::vec3(v.x, v.y, v.z); };
	glm::vec3 cameraUp = toGLM(up);
	glm::vec3 cameraOrigin = toGLM(origin);
	glm::vec3 cameraLookAt = toGLM(lookAt);
	originOut = cameraOrigin;

	float fovYRadians = static_cast<float>(fovYradians_);
	float nearClip = 0.01f;
	float farClip = 100.0f;

	viewOut = glm::lookAtRH(cameraOrigin, cameraLookAt, normalize(cameraUp));
	projectionOut = glm::perspectiveFovRH_NO(fovYRadians, float(width), float(height), nearClip, farClip);
	normalOut = glm::inverse(glm::transpose(glm::mat4(glm::mat3(viewOut))));
}

