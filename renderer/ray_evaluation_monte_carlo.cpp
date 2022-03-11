#include "ray_evaluation_monte_carlo.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "json_utils.h"

#include "renderer_commons.cuh"
#include "camera.h"
#include "helper_math.cuh"
#include "module_registry.h"
#include "volume_interpolation_grid.h"
#include "pytorch_utils.h"
#include <cuMat/src/Context.h>


renderer::RayEvaluationMonteCarlo::RayEvaluationMonteCarlo()
	: minDensity_(0)
	, maxDensity_(1)
	//, scatteringFactor_(0.2)
	, colorScaling_(1)
	, numBounces_(3)
	, lightPitchYawDistance_(make_double3(0,0,1))
	, lightRadius_(0.1)
	, lightIntensity_(1)
	, showLight_(false)
#if RENDERER_OPENGL_SUPPORT==1
    , lightMesh_(MeshCpu::createCube())
    , lightShader_("PassThrough.vs", "SimpleDiffuse.fs")
#endif
{
}

std::string renderer::RayEvaluationMonteCarlo::getName() const
{
	return "MonteCarlo";
}

void renderer::RayEvaluationMonteCarlo::prepareRendering(GlobalSettings& s) const
{
	IRayEvaluation::prepareRendering(s);
	//nothing to do otherwise (for now)
}

std::string renderer::RayEvaluationMonteCarlo::getDefines(const GlobalSettings& s) const
{
	auto volume = getSelectedVolume(s);
	if (!volume) throw std::runtime_error("No volume loaded!");

	std::stringstream ss;
	ss << "#define RAY_EVALUATION_MONTE_CARLO__VOLUME_INTERPOLATION_T "
		<< volume->getPerThreadType(s)
		<< "\n";
	ss << "#define RAY_EVALUATION_MONTE_CARLO__TRANSFER_FUNCTION_T "
		<< getSelectedTF()->getPerThreadType(s)
		<< "\n";
	//ss << "#define RAY_EVALUATION_MONTE_CARLO__BRDF_T "
	//	<< getSelectedBRDF()->getPerThreadType(s)
	//	<< "\n";
	ss << "#define RAY_EVALUATION_MONTE_CARLO__PHASE_FUNCTIION_T "
		<< getSelectedPhaseFunction()->getPerThreadType(s)
		<< "\n";
	ss << "#define RAY_EVALUATION_MONTE_CARLO__SAMPLING_T void\n";
	//if (!showLight_)
	ss << "#define RAY_EVALUATION_MONTE_CARLO__HIDE_LIGHT\n";
	return ss.str();
}

std::vector<std::string> renderer::RayEvaluationMonteCarlo::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_ray_evaluation_monte_carlo.cuh" };
}

std::string renderer::RayEvaluationMonteCarlo::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "rayEvaluationMonteCarloParameters";
}

std::string renderer::RayEvaluationMonteCarlo::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::RayEvaluationMonteCarlo";
}

double3 renderer::RayEvaluationMonteCarlo::getLightPosition(IModuleContainer_ptr root) const
{
	//assemble light position
	ICamera_ptr cam = std::dynamic_pointer_cast<ICamera>(root->getSelectedModuleForTag(ICamera::Tag()));
	CameraOnASphere::Orientation o = CameraOnASphere::Yp;
	if (const auto cams = std::dynamic_pointer_cast<CameraOnASphere>(cam))
	{
		o = cams->orientation();
	}
	return CameraOnASphere::eulerToCartesian(
		lightPitchYawDistance_.x, lightPitchYawDistance_.y, lightPitchYawDistance_.z, o);
}

void renderer::RayEvaluationMonteCarlo::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
	auto volume = getSelectedVolume(s);
	if (!volume) throw std::runtime_error("No volume loaded!");
	
	//assemble light position
	double3 lightPos = getLightPosition(s.root);
	double4 light = make_double4(lightPos, lightRadius_);
	
	RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "RayEvaluationMonteCarlo", [&]()
		{
			using scalar4 = kernel::scalar_traits<scalar_t>::real4;
			struct Parameters
			{
				scalar_t maxAbsorption;
				scalar_t densityMin;
				scalar_t densityMax;
				//scalar_t scatteringFactor;
				int numBounces;

				scalar4 lightPositionAndRadius;
				scalar_t lightIntensity;
				scalar_t colorScaling;
			} p;
			p.maxAbsorption = static_cast<scalar_t>(getSelectedTF()->getMaxAbsorption());
			p.densityMin = static_cast<scalar_t>(minDensity_);
			p.densityMax = static_cast<scalar_t>(maxDensity_);
			//p.scatteringFactor = static_cast<scalar_t>(scatteringFactor_);
			p.numBounces = numBounces_;
			p.lightPositionAndRadius = kernel::cast4<scalar_t>(light);
			p.lightIntensity = static_cast<scalar_t>(lightIntensity_);
			p.colorScaling = static_cast<scalar_t>(colorScaling_);
			CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
		});
}

bool renderer::RayEvaluationMonteCarlo::drawUI(UIStorage_t& storage)
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
		if (ImGui::SliderDouble("Color Scaling", &colorScaling_, 1e-5, 1.0, "%.5f", 2))
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

		minDensity_ = fmax(minDensity_, minDensity);
		maxDensity_ = fmin(maxDensity_, maxDensity);
		if (ImGui::SliderDouble("Min Density", &minDensity_, minDensity, maxDensity))
			changed = true;
		if (ImGui::SliderDouble("Max Density", &maxDensity_, minDensity, maxDensity))
			changed = true;
		storage[UI_KEY_SELECTED_MIN_DENSITY] = static_cast<double>(minDensity_);
		storage[UI_KEY_SELECTED_MAX_DENSITY] = static_cast<double>(maxDensity_);

		//if (ImGui::SliderDouble("Scattering##RayEvaluationMonteCarlo",
		//	&scatteringFactor_, 0.0, 1.0))
		//	changed = true;

		if (ImGui::SliderInt("Num Bounces##RayEvaluationMonteCarlo", &numBounces_, 0, 5))
			changed = true;

		double currentPitchDegree = glm::degrees(lightPitchYawDistance_.x);
		if (ImGui::SliderDouble("Light Pitch##RayEvaluationMonteCarlo", 
			&currentPitchDegree, -90, +90, u8"%.5f\u00b0"))
		{
			lightPitchYawDistance_.x = glm::radians(currentPitchDegree);
			changed = true;
		}
		double currentYawDegree = glm::degrees(lightPitchYawDistance_.y);
		if (ImGui::SliderDouble("Light Yaw##RayEvaluationMonteCarlo",
			&currentYawDegree, -180, +180, u8"%.5f\u00b0"))
		{
			lightPitchYawDistance_.y = glm::radians(currentYawDegree);
			changed = true;
		}
		if (ImGui::SliderDouble("Light Distance##RayEvaluationMonteCarlo",
			&lightPitchYawDistance_.z, 0.2, 3.0))
			changed = true;
		if (ImGui::SliderDouble("Light Radius##RayEvaluationMonteCarlo",
			&lightRadius_, 0.001, 1.0))
			changed = true;
		if (ImGui::SliderDouble("Light Intensity##RayEvaluationMonteCarlo",
			&lightIntensity_, 0.01, 10.0, "%.3f", 2))
			changed = true;
		if (ImGui::Checkbox("Show Light", &showLight_))
			changed = true;
	}

	//Phase Function
	const auto& phaseFunctions =
		ModuleRegistry::Instance().getModulesForTag(IPhaseFunction::Tag());
	if (!phaseFunction_)
		phaseFunction_ = std::dynamic_pointer_cast<IPhaseFunction>(phaseFunctions[0].first);
	if (ImGui::CollapsingHeader("PhaseFunction##IRayEvaluation", ImGuiTreeNodeFlags_DefaultOpen))
	{
		for (int i = 0; i < phaseFunctions.size(); ++i) {
			const auto& name = phaseFunctions[i].first->getName();
			if (ImGui::RadioButton(name.c_str(), phaseFunctions[i].first == phaseFunction_)) {
				phaseFunction_ = std::dynamic_pointer_cast<IPhaseFunction>(phaseFunctions[i].first);
				changed = true;
			}
			if (i < phaseFunctions.size() - 1) ImGui::SameLine();
		}
		if (phaseFunction_->drawUI(storage))
			changed = true;
	}

	////BRDF
	//const auto& brdfs =
	//	ModuleRegistry::Instance().getModulesForTag(IBRDF::Tag());
	//if (!brdf_)
	//	brdf_ = std::dynamic_pointer_cast<IBRDF>(brdfs[0].first);
	//if (ImGui::CollapsingHeader("BRDF##IRayEvaluation", ImGuiTreeNodeFlags_DefaultOpen))
	//{
	//	for (int i = 0; i < brdfs.size(); ++i) {
	//		const auto& name = brdfs[i].first->getName();
	//		if (ImGui::RadioButton(name.c_str(), brdfs[i].first == brdf_)) {
	//			brdf_ = std::dynamic_pointer_cast<IBRDF>(brdfs[i].first);
	//			changed = true;
	//		}
	//		if (i < brdfs.size() - 1) ImGui::SameLine();
	//	}
	//	if (brdf_->drawUI(storage))
	//		changed = true;
	//}

	return changed;
}

renderer::IModule_ptr renderer::RayEvaluationMonteCarlo::getSelectedModuleForTag(const std::string& tag) const
{
	if (tag == ITransferFunction::TAG)
		return tf_;
	//else if (tag == IBRDF::TAG)
	//	return brdf_;
	else if (tag == IPhaseFunction::TAG)
		return phaseFunction_;
	else
		return IRayEvaluation::getSelectedModuleForTag(tag);
}

std::vector<std::string> renderer::RayEvaluationMonteCarlo::getSupportedTags() const
{
	std::vector<std::string> tags = IRayEvaluation::getSupportedTags();
	tags.push_back(tf_->getTag());
	if (IModuleContainer_ptr mc = std::dynamic_pointer_cast<IModuleContainer>(tf_))
	{
		const auto& t = mc->getSupportedTags();
		tags.insert(tags.end(), t.begin(), t.end());
	}
	//tags.push_back(brdf_->getTag());
	//if (IModuleContainer_ptr mc = std::dynamic_pointer_cast<IModuleContainer>(brdf_))
	//{
	//	const auto& t = mc->getSupportedTags();
	//	tags.insert(tags.end(), t.begin(), t.end());
	//}
	tags.push_back(phaseFunction_->getTag());
	if (IModuleContainer_ptr mc = std::dynamic_pointer_cast<IModuleContainer>(phaseFunction_))
	{
		const auto& t = mc->getSupportedTags();
		tags.insert(tags.end(), t.begin(), t.end());
	}
	return tags;
}

void renderer::RayEvaluationMonteCarlo::load(const nlohmann::json& json, const ILoadingContext* context)
{
	IRayEvaluation::load(json, context);

	std::string tfName = json.value("selectedTF", "");
	tf_ = std::dynamic_pointer_cast<ITransferFunction>(
		context->getModule(ITransferFunction::Tag(), tfName));

	//std::string brdfName = json.value("selectedBRDF", "");
	//brdf_ = std::dynamic_pointer_cast<IBRDF>(
	//	fetcher->getModule(IBRDF::Tag(), brdfName));
	std::string phaseFunctionName = json.value("selectedPhaseFunction", "");
	phaseFunction_ = std::dynamic_pointer_cast<IPhaseFunction>(
		context->getModule(IPhaseFunction::Tag(), phaseFunctionName));

	minDensity_ = json.value("minDensity", 0.0);
	maxDensity_ = json.value("maxDensity", 1.0);
	//scatteringFactor_ = json.value("scatteringFactor", 0.5);
	numBounces_ = json.value("numBounces", 1);
	lightPitchYawDistance_ = json.value("lightPitchYawDistance", make_double3(0, 0, 1));
	lightRadius_ = json.value("lightRadius", 1.0);
	lightIntensity_ = json.value("lightIntensity", 1.0);
	showLight_ = json.value("showLight", false);
	colorScaling_ = json.value("colorScaling", 1.0);
}

void renderer::RayEvaluationMonteCarlo::save(nlohmann::json& json, const ISavingContext* context) const
{
	IRayEvaluation::save(json, context);
	
	json["selectedTF"] = tf_ ? tf_->getName() : "";
	/*json["selectedBRDF"] = brdf_ ? brdf_->getName() : "";*/
	json["selectedPhaseFunction"] = phaseFunction_ ? phaseFunction_->getName() : "";
	json["minDensity"] = minDensity_;
	json["maxDensity"] = maxDensity_;
	//json["scatteringFactor"] = scatteringFactor_;
	json["numBounces"] = numBounces_;
	json["lightPitchYawDistance"] = lightPitchYawDistance_;
	json["lightRadius"] = lightRadius_;
	json["lightIntensity"] = lightIntensity_;
	json["showLight"] = showLight_;
	json["colorScaling"] = colorScaling_;
}

bool renderer::RayEvaluationMonteCarlo::hasRasterizing() const
{
	return showLight_ || IRayEvaluation::hasRasterizing();
}

void renderer::RayEvaluationMonteCarlo::performRasterization(const RasterizingContext* context)
{
    IRayEvaluation::performRasterization(context); //call children

	//render light bulb
	if (showLight_)
	{
#if RENDERER_OPENGL_SUPPORT==1
		glDisable(GL_CULL_FACE);
		lightShader_.use();
		double3 lightPos = getLightPosition(context->root);
		glm::mat4 modelMatrix = glm::translate(glm::vec3(lightPos.x, lightPos.y, lightPos.z))
	        * glm::scale(glm::vec3(lightRadius_, lightRadius_, lightRadius_));
		lightShader_.setMat4("modelMatrix", modelMatrix);
		lightShader_.setMat4("viewMatrix", context->view);
		lightShader_.setMat4("projectionMatrix", context->projection);
		lightShader_.setVec3("ambientColor", 0.5f, 0.5, 0.5f);
		lightShader_.setVec3("diffuseColor", 0.5f, 0.5f, 0.5f);
		lightShader_.setVec3("lightDirection", 1, 0, 0);
		lightShader_.setVec3("cameraOrigin", context->origin);

		lightMesh_.drawIndexed();
#else
		throw std::runtime_error("OpenGL-Support disabled, can't visualize the light position");
#endif
	}
}

torch::Tensor renderer::RayEvaluationMonteCarlo::SampleLight(IImageEvaluator_ptr context,
	int numSamples, unsigned time, CUstream stream)
{
	int batches = context->computeBatchCount();
	if (batches != 1)
		throw std::runtime_error("batching not supported");
	auto m = context->getSelectedModuleForTag(IRayEvaluation::Tag());
	auto ray = std::dynamic_pointer_cast<RayEvaluationMonteCarlo>(m);
	if (!ray)
		throw std::runtime_error("Context does not point to MonteCarlo Ray Evaluation");

	IKernelModule::GlobalSettings s = context->getGlobalSettings();
	context->modulesPrepareRendering(s);

	const std::string kernelName = "RayEvaluationMCSampleLight";
	std::stringstream extraSource;
	extraSource << "#include \"renderer_ray_evaluation_monte_carlo_kernels.cuh\"\n";
	KernelLoader::KernelFunction fun = context->getKernel(
		s, kernelName, extraSource.str());
	context->fillConstants(fun, s, stream);

	torch::Tensor output = torch::empty({ numSamples, 3 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));

	int blockSize = fun.bestBlockSize();
	dim3 virtual_size{
		static_cast<unsigned int>(numSamples),
		static_cast<unsigned int>(1),
		static_cast<unsigned int>(1) };
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(numSamples, blockSize)), fun.minGridSize());
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "RayEvaluationMonteCarlo::SampleLight", [&]()
		{
			const auto acc = accessor< ::kernel::Tensor2RW<scalar_t>>(output);
			const void* args[] = { &virtual_size, &acc, &time };
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, blockSize, 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return printError(result, kernelName);
			return true;
		});
	if (!success) throw std::runtime_error("Error during rendering!");

	return output;
}

torch::Tensor renderer::RayEvaluationMonteCarlo::EvalBackground(IImageEvaluator_ptr context,
    const torch::Tensor& rayStart, const torch::Tensor& rayDir, unsigned time, CUstream stream)
{
	CHECK_CUDA(rayDir, true);
	CHECK_CUDA(rayStart, true);
	CHECK_DIM(rayStart, 2);
	CHECK_DIM(rayDir, 2);
	CHECK_SIZE(rayStart, 1, 3);
	CHECK_SIZE(rayDir, 1, 3);
	CHECK_SIZE(rayStart, 0, rayDir.size(0));
	int numSamples = rayStart.size(0);

	int batches = context->computeBatchCount();
	if (batches != 1)
		throw std::runtime_error("batching not supported");
	auto m = context->getSelectedModuleForTag(IRayEvaluation::Tag());
	auto ray = std::dynamic_pointer_cast<RayEvaluationMonteCarlo>(m);
	if (!ray)
		throw std::runtime_error("Context does not point to MonteCarlo Ray Evaluation");

	IKernelModule::GlobalSettings s = context->getGlobalSettings();
	context->modulesPrepareRendering(s);

	const std::string kernelName = "RayEvaluationMCEvalBackground";
	std::stringstream extraSource;
	extraSource << "#include \"renderer_ray_evaluation_monte_carlo_kernels.cuh\"\n";
	KernelLoader::KernelFunction fun = context->getKernel(
		s, kernelName, extraSource.str());
	context->fillConstants(fun, s, stream);

	torch::Tensor output = torch::empty({ numSamples, 4 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));

	int blockSize = fun.bestBlockSize();
	dim3 virtual_size{
		static_cast<unsigned int>(numSamples),
		static_cast<unsigned int>(1),
		static_cast<unsigned int>(1) };
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(numSamples, blockSize)), fun.minGridSize());
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "RayEvaluationMonteCarlo::EvalBackground", [&]()
		{
			auto accStart = accessor<::kernel::Tensor2Read<scalar_t>>(rayStart);
			auto accDir = accessor<::kernel::Tensor2Read<scalar_t>>(rayDir);
			const auto accOut = accessor< ::kernel::Tensor2RW<scalar_t>>(output);
			const void* args[] = { &virtual_size, &accStart, &accDir, &accOut, &time };
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, blockSize, 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return printError(result, kernelName);
			return true;
		});
	if (!success) throw std::runtime_error("Error during rendering!");

	return output;
}

std::tuple<torch::Tensor, torch::Tensor> renderer::RayEvaluationMonteCarlo::NextDirection(IImageEvaluator_ptr context,
    const torch::Tensor& rayStart, const torch::Tensor& rayDir, unsigned time, CUstream stream)
{
	CHECK_CUDA(rayDir, true);
	CHECK_CUDA(rayStart, true);
	CHECK_DIM(rayStart, 2);
	CHECK_DIM(rayDir, 2);
	CHECK_SIZE(rayStart, 1, 3);
	CHECK_SIZE(rayDir, 1, 3);
	CHECK_SIZE(rayStart, 0, rayDir.size(0));
	int numSamples = rayStart.size(0);

	int batches = context->computeBatchCount();
	if (batches != 1)
		throw std::runtime_error("batching not supported");
	auto m = context->getSelectedModuleForTag(IRayEvaluation::Tag());
	auto ray = std::dynamic_pointer_cast<RayEvaluationMonteCarlo>(m);
	if (!ray)
		throw std::runtime_error("Context does not point to MonteCarlo Ray Evaluation");

	IKernelModule::GlobalSettings s = context->getGlobalSettings();
	context->modulesPrepareRendering(s);

	const std::string kernelName = "RayEvaluationMCNextDir";
	std::stringstream extraSource;
	extraSource << "#include \"renderer_ray_evaluation_monte_carlo_kernels.cuh\"\n";
	KernelLoader::KernelFunction fun = context->getKernel(
		s, kernelName, extraSource.str());
	context->fillConstants(fun, s, stream);

	torch::Tensor outDir = torch::empty({ numSamples, 3 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	torch::Tensor outBeta = torch::empty({ numSamples, 1 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));

	int blockSize = fun.bestBlockSize();
	dim3 virtual_size{
		static_cast<unsigned int>(numSamples),
		static_cast<unsigned int>(1),
		static_cast<unsigned int>(1) };
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(numSamples, blockSize)), fun.minGridSize());
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "RayEvaluationMonteCarlo::NextDirection", [&]()
		{
			auto accStart = accessor<::kernel::Tensor2Read<scalar_t>>(rayStart);
			auto accDir = accessor<::kernel::Tensor2Read<scalar_t>>(rayDir);
			const auto accOutDir = accessor< ::kernel::Tensor2RW<scalar_t>>(outDir);
			const auto accOutBeta = accessor< ::kernel::Tensor2RW<scalar_t>>(outBeta);
			const void* args[] = { &virtual_size, &accStart, &accDir, &accOutDir, &accOutBeta, &time };
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, blockSize, 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return printError(result, kernelName);
			return true;
		});
	if (!success) throw std::runtime_error("Error during rendering!");

	return std::make_tuple(outDir, outBeta);
}

torch::Tensor renderer::RayEvaluationMonteCarlo::PhaseFunctionProbability(IImageEvaluator_ptr context,
	const torch::Tensor& dirIn, const torch::Tensor& dirOut, const torch::Tensor& position, CUstream stream)
{
	CHECK_CUDA(dirIn, true);
	CHECK_CUDA(dirOut, true);
	CHECK_CUDA(position, true);
	CHECK_DIM(dirIn, 2);
	CHECK_DIM(dirOut, 2);
	CHECK_DIM(position, 2);
	CHECK_SIZE(dirIn, 1, 3);
	CHECK_SIZE(dirOut, 1, 3);
	CHECK_SIZE(position, 1, 3);
	int numSamples = dirIn.size(0);
	CHECK_SIZE(dirOut, 0, numSamples);
	CHECK_SIZE(position, 0, numSamples);

	int batches = context->computeBatchCount();
	if (batches != 1)
		throw std::runtime_error("batching not supported");
	auto m = context->getSelectedModuleForTag(IRayEvaluation::Tag());
	auto ray = std::dynamic_pointer_cast<RayEvaluationMonteCarlo>(m);
	if (!ray)
		throw std::runtime_error("Context does not point to MonteCarlo Ray Evaluation");

	IKernelModule::GlobalSettings s = context->getGlobalSettings();
	context->modulesPrepareRendering(s);

	const std::string kernelName = "RayEvaluationMCPhaseFunctionProbability";
	std::stringstream extraSource;
	extraSource << "#include \"renderer_ray_evaluation_monte_carlo_kernels.cuh\"\n";
	KernelLoader::KernelFunction fun = context->getKernel(
		s, kernelName, extraSource.str());
	context->fillConstants(fun, s, stream);

	torch::Tensor output = torch::empty({ numSamples, 1 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));

	int blockSize = fun.bestBlockSize();
	dim3 virtual_size{
		static_cast<unsigned int>(numSamples),
		static_cast<unsigned int>(1),
		static_cast<unsigned int>(1) };
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(numSamples, blockSize)), fun.minGridSize());
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "RayEvaluationMonteCarlo::PhaseFunctionProbability", [&]()
		{
			auto accDirIn = accessor<::kernel::Tensor2Read<scalar_t>>(dirIn);
			auto accDirOut = accessor<::kernel::Tensor2Read<scalar_t>>(dirOut);
			auto accPos = accessor<::kernel::Tensor2Read<scalar_t>>(position);
			const auto accOut = accessor< ::kernel::Tensor2RW<scalar_t>>(output);
			const void* args[] = { &virtual_size, &accDirIn, &accDirOut, &accPos, &accOut};
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, blockSize, 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return printError(result, kernelName);
			return true;
		});
	if (!success) throw std::runtime_error("Error during rendering!");

	return output;
}

void renderer::RayEvaluationMonteCarlo::registerPybindModule(pybind11::module& m)
{
	IRayEvaluation::registerPybindModule(m);

	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<RayEvaluationMonteCarlo, IRayEvaluation, std::shared_ptr<RayEvaluationMonteCarlo>>(m, "RayEvaluationMonteCarlo")
		.def(py::init<>())
		.def_readwrite("min_density", &RayEvaluationMonteCarlo::minDensity_)
		.def_readwrite("max_density", &RayEvaluationMonteCarlo::maxDensity_)
		//.def_readwrite("scattering_factor", &RayEvaluationMonteCarlo::scatteringFactor_)
		.def_readwrite("num_bounces", &RayEvaluationMonteCarlo::numBounces_)
		.def_readwrite("color_scaling", &RayEvaluationMonteCarlo::colorScaling_)
		.def_readwrite("light_pitch_yaw_distance", &RayEvaluationMonteCarlo::lightPitchYawDistance_)
		.def_readwrite("light_radius", &RayEvaluationMonteCarlo::lightRadius_)
		.def_readwrite("light_intensity", &RayEvaluationMonteCarlo::lightIntensity_)
		.def_readwrite("tf", &RayEvaluationMonteCarlo::tf_)
		//.def_readwrite("brdf", &RayEvaluationMonteCarlo::brdf_);
		.def_readwrite("phase_function", &RayEvaluationMonteCarlo::phaseFunction_)
	    .def_static("SampleLight", [](IImageEvaluator_ptr context, int numSamples, unsigned int time)
	    {
				return RayEvaluationMonteCarlo::SampleLight(context, numSamples, time, IImageEvaluator::getDefaultStream());
	    }, py::doc(R"(
Samples the light position for MC evaluation.
context: the image evaluator for the context (camera, volume, phase functions, ...)
num_samples: the number of samples to generate
time: the time for the random number generator
Return: the light position of shape (N,3)
        )"), py::arg("context"), py::arg("num_samples"), py::arg("time"))
		.def_static("EvalBackground", [](IImageEvaluator_ptr context, torch::Tensor rayStart, torch::Tensor rayDir, unsigned int time)
			{
				return RayEvaluationMonteCarlo::EvalBackground(context, rayStart, rayDir, time, IImageEvaluator::getDefaultStream());
			}, py::doc(R"(
Evaluates the background illumination.
context: the image evaluator for the context (camera, volume, phase functions, ...)
ray_start: the start position of the ray of shape (N,3)
ray_dir: the ray direction of shape (N,3)
time: the time for the random number generator
Return: the rgb-alpha color of shape (N,4)
        )"), py::arg("context"), py::arg("ray_start"), py::arg("ray_dir"), py::arg("time"))
		.def_static("NextDirection", [](IImageEvaluator_ptr context, torch::Tensor rayStart, torch::Tensor rayDir, unsigned int time)
			{
				return RayEvaluationMonteCarlo::NextDirection(context, rayStart, rayDir, time, IImageEvaluator::getDefaultStream());
			}, py::doc(R"(
Evaluates the next direction after a scattering event.
context: the image evaluator for the context (camera, volume, phase functions, ...)
ray_start: the start position of the ray of shape (N,3)
ray_dir: the ray direction of shape (N,3)
time: the time for the random number generator
Return: a tuple [ next direction of shape (N,3),  beta scaling of shape (N,1) ]
        )"), py::arg("context"), py::arg("ray_start"), py::arg("ray_dir"), py::arg("time"))
        .def_static("PhaseFunctionProbability", [](IImageEvaluator_ptr context, torch::Tensor dirIn, torch::Tensor dirOut, torch::Tensor pos)
			{
				return RayEvaluationMonteCarlo::PhaseFunctionProbability(context, dirIn, dirOut, pos, IImageEvaluator::getDefaultStream());
			}, py::doc(R"(
Evaluates the phase function probability
context: the image evaluator for the context (camera, volume, phase functions, ...)
dir_in: input direction of shape (N,3)
dir_out: output direction of shape (N,3)
pos: current position of shape (N,3)
Return: probability of the scattering event of shape (N,1)
        )"), py::arg("context"), py::arg("dir_in"), py::arg("dir_out"), py::arg("pos"))
    ;
}
