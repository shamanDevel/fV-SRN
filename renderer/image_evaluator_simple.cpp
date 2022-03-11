#include "image_evaluator_simple.h"

#include <sstream>
#include <magic_enum.hpp>
#include <torch/csrc/cuda/Stream.h>
#include <cuMat/src/Context.h>

#include "tinyformat.h"
#include "pytorch_utils.h"
#include "module_registry.h"


const std::string renderer::ImageEvaluatorSimple::UI_KEY_SELECTED_VOLUME = "ImageEvaluatorSimple::SelectedVolume";
const std::string renderer::ImageEvaluatorSimple::NAME = "Simple";

renderer::ImageEvaluatorSimple::ImageEvaluatorSimple()
	: isUImode_(false)
	, samplesPerIterationLog2_(3)
	, currentTime_(0)
	, refiningCounter_(0)
	, useTonemapping_(false)
	, lastMaxExposure_(0)
	, tonemappingShoulder_(1.0f)
	, fixMaxExposure_(false)
	, fixedMaxExposure_(1.0f)
    , rasterizationContainer_(
		std::dynamic_pointer_cast<RasterizationContainer>(ModuleRegistry::Instance().getModule(
			RasterizationContainer::Tag(), RasterizationContainer::Name())))
{
}

const std::string& renderer::ImageEvaluatorSimple::Name()
{
	return NAME;
}

std::string renderer::ImageEvaluatorSimple::getName() const
{
	return Name();
}

bool renderer::ImageEvaluatorSimple::drawUI(UIStorage_t& storage)
{
	isUImode_ = true;
	bool changed = false;
	if (drawUIGlobalSettings(storage))
		changed = true;

	//camera
	ICamera_ptr camera = selectedCamera_;
	if (const auto& it = storage.find(UI_KEY_SELECTED_CAMERA);
		it != storage.end())
	{
		camera = std::any_cast<ICamera_ptr>(it->second);
	}
	const auto& cameras =
		ModuleRegistry::Instance().getModulesForTag(ICamera::Tag());
	if (camera == nullptr)
		camera = std::dynamic_pointer_cast<ICamera>(cameras[0].first);
	if (ImGui::CollapsingHeader("Camera##IImageEvaluator", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (cameras.size() > 1) {
			for (int i = 0; i < cameras.size(); ++i) {
				const auto& name = cameras[i].first->getName();
				if (ImGui::RadioButton(name.c_str(), cameras[i].first == camera)) {
					camera = std::dynamic_pointer_cast<ICamera>(cameras[i].first);
					changed = true;
				}
				if (i < cameras.size() - 1) ImGui::SameLine();
			}
		}
		if (camera->drawUI(storage))
			changed = true;
	}
	selectedCamera_ = camera;
	storage[UI_KEY_SELECTED_CAMERA] = camera;

	//volume selection
	IVolumeInterpolation_ptr selection = selectedVolume_;
	ImGui::PushID("IRayEvaluation");
	const auto& volumes =
		ModuleRegistry::Instance().getModulesForTag(IVolumeInterpolation::Tag());
	if (selection == nullptr)
		selection = std::dynamic_pointer_cast<IVolumeInterpolation>(volumes[0].first);
	if (ImGui::CollapsingHeader("Volume", ImGuiTreeNodeFlags_DefaultOpen))
	{
		for (int i = 0; i < volumes.size(); ++i) {
			const auto& name = volumes[i].first->getName();
			if (ImGui::RadioButton(name.c_str(), volumes[i].first == selection)) {
				selection = std::dynamic_pointer_cast<IVolumeInterpolation>(volumes[i].first);
				changed = true;
			}
			if (i < volumes.size() - 1) ImGui::SameLine();
		}
		if (selection->drawUI(storage))
			changed = true;
	}
	ImGui::PopID();

	storage[UI_KEY_SELECTED_VOLUME] = static_cast<IVolumeInterpolation_ptr>(selection);
	selectedVolume_ = selection;

	//ray evaluator (not shared)
	ImGui::Separator();
	const auto& rayEvaluators =
		ModuleRegistry::Instance().getModulesForTag(IRayEvaluation::Tag());
	if (selectedRayEvaluator_ == nullptr)
		selectedRayEvaluator_ = std::dynamic_pointer_cast<IRayEvaluation>(rayEvaluators[0].first);
	if (rayEvaluators.size() > 1)
	{
		for (int i = 0; i < rayEvaluators.size(); ++i) {
			const auto& name = rayEvaluators[i].first->getName();
			if (ImGui::RadioButton(name.c_str(), rayEvaluators[i].first == selectedRayEvaluator_)) {
				selectedRayEvaluator_ = std::dynamic_pointer_cast<IRayEvaluation>(rayEvaluators[i].first);
				changed = true;
			}
			if (i < rayEvaluators.size() - 1) ImGui::SameLine();
		}
	}
	if (selectedRayEvaluator_->shouldSupersample()) {
		std::string currentSamples = std::to_string(1 << samplesPerIterationLog2_);
		if (ImGui::SliderInt("SPP##ImageEvaluatorSimple", &samplesPerIterationLog2_, 0, 6, currentSamples.c_str()))
			changed = true;
		ImGui::Text("Refinements: %d", refiningCounter_);
	}
	if (selectedRayEvaluator_->drawUI(storage))
		changed = true;

	//rasterization
	ImGui::Separator();
	if (rasterizationContainer_->drawUI(storage))
		changed = true;

	//Tonemapping + output
	ImGui::Separator();
	bool tonemappingChanged = false;
	if (ImGui::Checkbox("Tonemapping##ImageEvaluatorSimple", &useTonemapping_))
		tonemappingChanged = true;
	if (useTonemapping_)
	{
		if (ImGui::Checkbox("fix max exposure", &fixMaxExposure_))
		{
			if (fixMaxExposure_)
				fixedMaxExposure_ = lastMaxExposure_;
			tonemappingChanged = true;
		}
		ImGui::Text("max exposure: %.3e", fixMaxExposure_ ? fixedMaxExposure_ : lastMaxExposure_);
		if (ImGui::SliderFloat("Shoulder##ImageEvaluatorSimple",
				&tonemappingShoulder_, 1e-4, 1, "%.5f", 10))
			tonemappingChanged = true;
	}
	if (!isIterativeRefining())
		changed = changed || tonemappingChanged;
	
	if (drawUIOutputChannel(storage))
		changed = true;
	return changed;
}

renderer::IModule_ptr renderer::ImageEvaluatorSimple::getSelectedModuleForTag(const std::string& tag) const
{
	//TODO: add IRasterization

	if (tag == ICamera::Tag())
		return selectedCamera_;
	if (tag == IRayEvaluation::Tag())
		return selectedRayEvaluator_;
	if (tag == IVolumeInterpolation::Tag())
		return getSelectedVolume();
	if (tag == RasterizationContainer::Tag())
		return rasterizationContainer_;
	if (const auto m = selectedRayEvaluator_->getSelectedModuleForTag(tag); m != nullptr) 
		return m;
	return nullptr;
}

std::vector<std::string> renderer::ImageEvaluatorSimple::getSupportedTags() const
{
	//TODO: add IRasterization

	std::vector<std::string> tags;
	tags.push_back(ICamera::Tag());
	tags.push_back(IVolumeInterpolation::Tag());
	tags.push_back(RasterizationContainer::Tag());
	if (selectedRayEvaluator_) {
		const auto& t = selectedRayEvaluator_->getSupportedTags();
		tags.insert(tags.end(), t.begin(), t.end());
	}
	tags.push_back(IRayEvaluation::Tag());
	return tags;
}

bool renderer::ImageEvaluatorSimple::isIterativeRefining() const
{
	return selectedRayEvaluator_ ? selectedRayEvaluator_->isIterativeRefining() : false;
}

torch::Tensor renderer::ImageEvaluatorSimple::render(
	int width, int height, CUstream stream, bool refine, const torch::Tensor& out)
{
	//settings - part 1
	int batches = computeBatchCount();
	//settings
	selectedCamera_->setAspectRatio(double(width) / height);
	IKernelModule::GlobalSettings s = getGlobalSettings();

	torch::Tensor backgroundImage;
#if RENDERER_OPENGL_SUPPORT==1
	bool hasRasterization = this->hasRasterizing();
	if (hasRasterization)
	{
		if (batches>1)
		{
			throw std::runtime_error("Rasterization requires batch count = 1 (no batched rasterization possible)");
		}

	    //Check if the framebuffer is of the correct size
		if (!framebuffer_ || framebuffer_->width()!=width || framebuffer_->height()!=height)
		{
			std::cout << "(Re)Create the Framebuffer for Rasterization" << std::endl;
			framebuffer_ = std::make_unique<Framebuffer>(width, height);
		}

		//create context (camera)
		RasterizingContext context;
		selectedCamera_->computeOpenGlMatrices(width, height,
			context.view, context.projection, context.normal, context.origin);
		context.root = shared_from_this();

		//bind - render - unbind
		framebuffer_->bind();
		glClearColor(0, 0, 0, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		this->performRasterization(&context);

	    framebuffer_->unbind();

		//copy to CUDA
		backgroundImage = torch::empty({ 1, 5, height, width }, 
			at::TensorOptions().device(c10::kCUDA).dtype(s.scalarType));
		framebuffer_->copyToCuda(backgroundImage);
	}
#else
	const bool hasRasterization = false;
#endif

	//settings - part 2
	if (selectedChannel_ == ChannelMode::ChannelNormal) {
		s.volumeShouldProvideNormals = true;
		//std::cout << "ChannelNormal selected, request that the volume provides normals" << std::endl;
	}
	modulesPrepareRendering(s);
	if (s.synchronizedThreads)
	{
	    if (width*height*batches % 32 != 0)
	    {
			//I'm too lazy to handle half-filled warps
			throw std::runtime_error(R"(
width*height*batches is not a multiple of 32 -> can't launch kernel with synchronous tracing.
Synchronous tracing, i.e. the threads per warp don't terminate early or diverge, is needed for scene networks.
)");
	    }
	}
	const std::string kernelName = "ImageEvaluatorSimpleKernel";
	std::stringstream extraSource;
	extraSource << "#define IMAGE_EVALUATOR__CAMERA_T " <<
		selectedCamera_->getPerThreadType(s) << "\n";
	extraSource << "#define IMAGE_EVALUATOR__RAY_EVALUATOR_T " <<
		selectedRayEvaluator_->getPerThreadType(s) << "\n";
	int numSamples = 1;
	if (selectedRayEvaluator_->requiresSampler())
		extraSource << "#define IMAGE_EVALUATOR__REQUIRES_SAMPLER\n";
	if (selectedRayEvaluator_->shouldSupersample())
	{
		numSamples = 1 << samplesPerIterationLog2_;
		extraSource << "#define IMAGE_EVALUATOR__SUPERSAMPLING\n";
	}
	if (hasRasterization)
	{
		extraSource << "#define IMAGE_EVALUATOR__HAS_BACKGROUND_IMAGE\n";
	}
	extraSource << "#include \"renderer_image_evaluator_simple.cuh\"\n";
	KernelLoader::KernelFunction fun = getKernel(
		s, kernelName, extraSource.str());
	fillConstants(fun, s, stream);

	if (!refine)
		refiningCounter_ = 0;
	
	//allocate tensor
	torch::Tensor t = out;
	if (refine ||
		t.scalar_type() != s.scalarType ||
		!t.is_cuda() ||
		t.ndimension() != 4 ||
		t.size(0) != batches ||
		t.size(1) != NUM_CHANNELS ||
		t.size(2) != height ||
		t.size(3) != width)
	{
		//re-allocate
		t = torch::empty({ batches, NUM_CHANNELS, height, width },
			at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	}

	if (refine)
	{
		TORCH_CHECK(t.sizes() == out.sizes());
	}
	
	//launch kernel
	int blockSize;
	if (s.fixedBlockSize > 0)
	{
		if (s.fixedBlockSize > fun.bestBlockSize())
			throw std::runtime_error(tinyformat::format(
				"larger block size requested that can be fulfilled. Requested: %d, possible: %d",
				s.fixedBlockSize, fun.bestBlockSize()));
		blockSize = s.fixedBlockSize;
	}
	else
	{
		blockSize = fun.bestBlockSize();
	}
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(width * height * batches, blockSize)),
		fun.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(width),
		static_cast<unsigned int>(height),
		static_cast<unsigned int>(batches) };
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "ImageEvaluatorSimple::render", [&]()
	{
		const auto acc = accessor< ::kernel::Tensor4Read<scalar_t>>(t);
		::kernel::Tensor4Read<scalar_t> backgroundAcc;
		if (hasRasterization)
			backgroundAcc = accessor<::kernel::Tensor4Read<scalar_t>>(backgroundImage);
		const void* args[] = { &virtual_size, &acc, &numSamples, &currentTime_, &backgroundAcc };
		auto result = cuLaunchKernel(
			fun.fun(), minGridSize, 1, 1, blockSize, 1, 1,
			0, stream, const_cast<void**>(args), NULL);
		if (result != CUDA_SUCCESS)
			return printError(result, kernelName);
		return true;
	});
	currentTime_++;
	if (!success) throw std::runtime_error("Error during rendering!");

	
	if (refine)
	{
		//blend with previous
		refiningCounter_++;
		t = out + (t - out) * (1.0 / refiningCounter_);
	}

	lastMaxExposure_ = torch::max(t.slice(1, 0, 3)).item().toFloat();
	
	return t;
}

float renderer::ImageEvaluatorSimple::getExposureForTonemapping() const
{
	if (fixMaxExposure_)
		return fmaxf(0.001f, fixedMaxExposure_ * tonemappingShoulder_);
	else
		return fmaxf(0.001f, lastMaxExposure_ * tonemappingShoulder_);
}

void renderer::ImageEvaluatorSimple::extractColor(const torch::Tensor& input_tensor, tensor_or_texture_t output,
    ChannelMode channel, CUstream stream)
{
    IImageEvaluator::ExtractColor(input_tensor, output, 
		useTonemapping_, getExposureForTonemapping(), channel, stream);
}


torch::Tensor renderer::ImageEvaluatorSimple::extractColorTorch(const torch::Tensor& rawInputTensor,
	bool useTonemapping, float maxExposure, ChannelMode channel)
{
	auto B = rawInputTensor.size(0);
	auto H = rawInputTensor.size(2);
	auto W = rawInputTensor.size(3);
	torch::Tensor output = torch::empty({ B, 4, H, W }, rawInputTensor.options());
	CUstream stream = getDefaultStream();
	ExtractColor(rawInputTensor, output, useTonemapping, maxExposure, channel, stream);
	return output;
}

void renderer::ImageEvaluatorSimple::load(const nlohmann::json& json, const ILoadingContext* context)
{
	IImageEvaluator::load(json, context);
	
	std::string cameraName = json.value("selectedCamera", "");
	selectedCamera_ = std::dynamic_pointer_cast<ICamera>(
		context->getModule(ICamera::Tag(), cameraName));

	std::string selectionName = json.value("selectedVolume", "");
	selectedVolume_ = std::dynamic_pointer_cast<IVolumeInterpolation>(
		context->getModule(IVolumeInterpolation::Tag(), selectionName));

	std::string rayEvaluatorName = json.value("selectedRayEvaluator", "");
	selectedRayEvaluator_ = std::dynamic_pointer_cast<IRayEvaluation>(
		context->getModule(IRayEvaluation::Tag(), rayEvaluatorName));

	samplesPerIterationLog2_ = json.value("samplesPerIterationLog2", 2);
	useTonemapping_ = json.value("useTonemapping", false);
	tonemappingShoulder_ = json.value("tonemappingShoulder", 1.0f);
	fixMaxExposure_ = json.value("fixMaxExposure", false);
	fixedMaxExposure_ = json.value("fixedMaxExposure", 1.0f);
}

void renderer::ImageEvaluatorSimple::save(nlohmann::json& json, const ISavingContext* context) const
{
	IImageEvaluator::save(json, context);
	json["selectedCamera"] = selectedCamera_->getName();
	json["selectedVolume"] = selectedVolume_ ? selectedVolume_->getName() : "";
	json["selectedRayEvaluator"] = selectedRayEvaluator_->getName();
	json["samplesPerIterationLog2"] = samplesPerIterationLog2_;
	json["useTonemapping"] = useTonemapping_;
	json["tonemappingShoulder"] = tonemappingShoulder_;
	json["fixMaxExposure"] = fixMaxExposure_;
	json["fixedMaxExposure"] = fixedMaxExposure_;
}

void renderer::ImageEvaluatorSimple::registerPybindModule(pybind11::module& m)
{
	IImageEvaluator::registerPybindModule(m);

	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<ImageEvaluatorSimple, IImageEvaluator, ImageEvaluatorSimple_ptr>(m, "ImageEvaluatorSimple")
		.def_readwrite("camera", &ImageEvaluatorSimple::selectedCamera_,
			py::doc("The selected ICamera method"))
		.def_readwrite("ray_evaluator", &ImageEvaluatorSimple::selectedRayEvaluator_,
			py::doc("The selected IRayEvaluation method"))
		.def_readwrite("volume", &ImageEvaluatorSimple::selectedVolume_,
			py::doc("The selected IVolumeINterpolation method"))
	    .def_readonly("rasterization_container", &ImageEvaluatorSimple::rasterizationContainer_,
			py::doc("The container for rasterized objects"))
		.def_readwrite("spp_log2", &ImageEvaluatorSimple::samplesPerIterationLog2_)
		.def_readwrite("use_tonemapping", &ImageEvaluatorSimple::useTonemapping_)
		.def_readwrite("tonemapping_shoulder", &ImageEvaluatorSimple::tonemappingShoulder_)
		.def_readonly("last_max_exposure", &ImageEvaluatorSimple::lastMaxExposure_)
		.def_readwrite("fix_max_exposure", &ImageEvaluatorSimple::fixMaxExposure_)
		.def_readwrite("fixed_max_exposure", &ImageEvaluatorSimple::fixedMaxExposure_)
		.def_static("Extract_color", &ImageEvaluatorSimple::extractColorTorch,
			py::doc(R"(
				Extracts the color and performs tonemapping.
				:param raw_input: the raw input tensor from :render
				:param use_tonemapping: true to enable tonemapping
				:param max_exposure: the maximal exposure to consider in tonemapping
                :param channel: the channel to use
				:return: a RGB-alpha tensor of shape (B, 4, H, W)
				)"),
			py::arg("raw_input"), py::arg("use_tonemapping"), py::arg("max_exposure"),
			py::arg("channel") = ChannelMode::ChannelColor)
		.def("extract_color", [](const ImageEvaluatorSimple& self, const torch::Tensor& rawInput)
			{
				return ImageEvaluatorSimple::extractColorTorch(rawInput, self.useTonemapping_,
					self.getExposureForTonemapping(), self.selectedChannel());
			}, py::doc(R"(
				Extracts the color and performs tonemapping.
				This variant uses the default setting from tonemapping and last_max_exposure,
                as well as the selected channel.
				:param raw_input: the raw input tensor from :render
				:return: a RGB-alpha tensor of shape (B, 4, H, W)
				)"), py::arg("raw_input"));
	//TODO: rasterization
}
