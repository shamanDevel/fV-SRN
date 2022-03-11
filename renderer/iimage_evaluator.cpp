#include "iimage_evaluator.h"

#include <sstream>
#include <magic_enum.hpp>
#include <torch/csrc/cuda/Stream.h>
#include <cuMat/src/Context.h>

#include "tinyformat.h"
#include "pytorch_utils.h"
#include "module_registry.h"

const char* renderer::IImageEvaluator::ChannelModeNames[] = {
	"Mask", "Normal", "Depth", "Color"
};

const std::string renderer::IImageEvaluator::UI_KEY_SELECTED_CAMERA = "camera::IImageEvaluator";
const std::string renderer::IImageEvaluator::UI_KEY_SELECTED_OUTPUT_CHANNEL = "outputChannel::IImageEvaluator";
const std::string renderer::IImageEvaluator::UI_KEY_USE_DOUBLE_PRECISION = "doublePrecision::IImageEvaluator";

renderer::IImageEvaluator::IImageEvaluator()
	: selectedChannel_(ChannelMode::ChannelColor)
	, isDoublePrecision_(false)
{
}

void renderer::IImageEvaluator::ExtractColor(const torch::Tensor& inputTensor, tensor_or_texture_t output,
    bool useTonemapping, float maxExposure, ChannelMode channel, CUstream stream)
{
	CHECK_CUDA(inputTensor, true);
	CHECK_DIM(inputTensor, 4);
	CHECK_SIZE(inputTensor, 1, 8);
	CHECK_SIZE(inputTensor, 0, 1); //only one batch
	int B = inputTensor.size(0);
	int height = inputTensor.size(2);
	int width = inputTensor.size(3);

	if (std::holds_alternative<torch::Tensor>(output))
	{
		auto& t = std::get<torch::Tensor>(output);
		CHECK_CUDA(t, true);
		CHECK_DIM(t, 4);
		CHECK_SIZE(t, 0, B);
		CHECK_SIZE(t, 1, 4);
		CHECK_SIZE(t, 2, height);
		CHECK_SIZE(t, 3, width);
	}
	else
	{
		TORCH_CHECK(B == 1, "If the output is an OpenGL texture, batches are not supported");
	}

	int r, g, b, a;
	float scaleRGB, offsetRGB, scaleA, offsetA;
	switch (channel)
	{
	case ChannelMode::ChannelColor:
	{
		if (useTonemapping)
		{
			RENDERER_DISPATCH_FLOATING_TYPES(inputTensor.scalar_type(), "IImageEvaluator::copyOutputToTexture", [&]()
				{
					const auto accIn = accessor<::kernel::Tensor4Read<scalar_t>>(inputTensor);
					if (std::holds_alternative<torch::Tensor>(output))
					{
						auto& t = std::get<torch::Tensor>(output);
						auto accOut = accessor<::kernel::Tensor4RW<scalar_t>>(t);
						::kernel::Tonemapping(
							width, height, B, accIn, accOut,
							maxExposure,
							stream);
					}
					else
					{
						GLubyte* t = std::get<GLubyte*>(output);
						::kernel::Tonemapping(
							width, height, accIn, t,
							maxExposure,
							stream);
					}
				});
			return;
		}
		else if (std::holds_alternative<torch::Tensor>(output))
        {
			auto& t = std::get<torch::Tensor>(output);
			t.copy_(inputTensor.slice(1, 0, 4)); //fast-track
        }
		r = 0; g = 1; b = 2; a = 3;
		scaleRGB = 1; offsetRGB = 0;
		scaleA = 1; offsetA = 0;
	} break;
	case ChannelMode::ChannelDepth:
	{
		//TODO: only select pixels that contain data
		float minDepth = inputTensor.select(1, 7).min().item().toFloat();
		float maxDepth = inputTensor.select(1, 7).max().item().toFloat();
		r = g = b = 7; a = 3;
		scaleRGB = 1 / (maxDepth - minDepth);
		offsetRGB = -minDepth / (maxDepth - minDepth);
		scaleA = 0; offsetA = 1;
	} break;
	case ChannelMode::ChannelMask:
	{
		r = g = b = a = 3;
		scaleRGB = 1; offsetRGB = 0;
		scaleA = 0; offsetA = 1;
	} break;
	case ChannelMode::ChannelNormal:
		r = 4; g = 5; b = 6; a = 3;
		scaleRGB = 0.5; offsetRGB = 0.5;
		scaleA = 1; offsetA = 0;
	}

	RENDERER_DISPATCH_FLOATING_TYPES(inputTensor.scalar_type(), "IImageEvaluator::copyOutputToTexture", [&]()
		{
			const auto accIn = accessor<::kernel::Tensor4Read<scalar_t>>(inputTensor);
			if (std::holds_alternative<torch::Tensor>(output))
			{
				auto& t = std::get<torch::Tensor>(output);
				auto accOut = accessor<::kernel::Tensor4RW<scalar_t>>(t);
				::kernel::CopyOutputToTexture(
					width, height, B, accIn, accOut,
					r, g, b, a, scaleRGB, offsetRGB, scaleA, offsetA,
					stream);
			}
			else
			{
				GLubyte* t = std::get<GLubyte*>(output);
				::kernel::CopyOutputToTexture(
					width, height, accIn, t,
					r, g, b, a, scaleRGB, offsetRGB, scaleA, offsetA,
					stream);
			}
		});
}


int renderer::IImageEvaluator::computeBatchCount()
{
	const auto s = getGlobalSettings();
	int batch = 1;
	std::string lastBatchModule;
	for (const auto& tag : getSupportedTags())
	{
		const auto m = getSelectedModuleForTag(tag);
		if (IKernelModule_ptr km = std::dynamic_pointer_cast<IKernelModule>(m))
		{
			auto b = km->getBatches(s).value_or(1);
			if (b > 1)
			{
				std::string currentBatchModule = km->getTag() + ":" + km->getName();
				if (batch > 1 && batch != b)
				{
					throw std::runtime_error(tinyformat::format(
						"Batch counts don't match. Module %s has %d batches, but the current module %s has %d batches",
						lastBatchModule, batch, currentBatchModule, b
					));
				}
				lastBatchModule = currentBatchModule;
				batch = b;
			}
		}
	}
	return batch;
}

CUstream renderer::IImageEvaluator::getDefaultStream()
{
	return c10::cuda::getCurrentCUDAStream();
}

renderer::IKernelModule::GlobalSettings renderer::IImageEvaluator::getGlobalSettings(
	const std::optional<bool>& useDoublePrecision)
{
	renderer::IKernelModule::GlobalSettings s;
	s.root = shared_from_this();
	s.scalarType = useDoublePrecision.value_or(isDoublePrecision_)
		? c10::kDouble : c10::kFloat;
	s.volumeShouldProvideNormals = false;
	return s;
}

void renderer::IImageEvaluator::modulesPrepareRendering(IKernelModule::GlobalSettings& s)
{
	const auto oldScalarType = s.scalarType;
	for (const auto& tag : getSupportedTags())
	{
		const auto m = getSelectedModuleForTag(tag);
		if (IKernelModule_ptr km = std::dynamic_pointer_cast<IKernelModule>(m))
		{
			km->prepareRendering(s);
		}
	}
	TORCH_CHECK(oldScalarType == s.scalarType, "the scalar type must not be changed!");
}

renderer::KernelLoader::KernelFunction renderer::IImageEvaluator::getKernel(
	const IKernelModule::GlobalSettings& s, const std::string& kernelName, const std::string& extraSource)
{
	std::stringstream defines;
	std::stringstream includes;
	std::stringstream extraModuleSource;
	std::vector<std::string> constantNames;

	defines << "#define KERNEL_DOUBLE_PRECISION "
		<< (s.scalarType == IKernelModule::GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
	defines << "#define KERNEL_SYNCHRONIZED_TRACING "
		<< (s.synchronizedThreads ? 1 : 0)
		<< "\n";

	for (const auto& tag : getSupportedTags())
	{
		const auto m = getSelectedModuleForTag(tag);
		if (IKernelModule_ptr km = std::dynamic_pointer_cast<IKernelModule>(m))
		{
			defines <<
				"// " << km->getTag() << " : " << km->getName() << "\n"
				<< km->getDefines(s) << "\n";
			includes <<
				"// " << km->getTag() << " : " << km->getName() << "\n";
			for (const auto& i : km->getIncludeFileNames(s))
				includes << "#include \"" << i << "\"\n";
			km->fillExtraSourceCode(s, extraModuleSource);
			const auto c = km->getConstantDeclarationName(s);
			if (!c.empty())
				constantNames.push_back(c);
		}
	}

	std::stringstream sourceFile;
	sourceFile << "// DEFINES:\n" << defines.str()
		<< "\n// EXTRA SOURCES:\n" << extraModuleSource.str()
		<< "\n// INCLUDES:\n" << includes.str()
		<< "\n// MAIN SOURCE:\n" << extraSource;

	//create kernel
	const auto fun = KernelLoader::Instance().getKernelFunction(
		kernelName, sourceFile.str(), constantNames, false, false);
	return fun.value();
}

void renderer::IImageEvaluator::fillConstants(
	KernelLoader::KernelFunction& f, const IKernelModule::GlobalSettings& s, CUstream stream)
{
	for (const auto& tag : getSupportedTags())
	{
		const auto m = getSelectedModuleForTag(tag);
		if (IKernelModule_ptr km = std::dynamic_pointer_cast<IKernelModule>(m))
		{
			const auto c = km->getConstantDeclarationName(s);
			if (!c.empty())
			{
				CUdeviceptr ptr = f.constant(c);
				km->fillConstantMemory(s, ptr, stream);
			}
		}
	}
}

bool renderer::IImageEvaluator::drawUIGlobalSettings(UIStorage_t& storage)
{
	//double precision is shared
	bool isDoublePrecision = isDoublePrecision_;
	if (const auto& it = storage.find(UI_KEY_USE_DOUBLE_PRECISION);
		it != storage.end())
	{
		isDoublePrecision = std::any_cast<bool>(it->second);
	}

	bool changed = false;
	if (ImGui::Checkbox("double precision##IImageEvaluator", &isDoublePrecision))
		changed = true;

	isDoublePrecision_ = isDoublePrecision;
	storage[UI_KEY_USE_DOUBLE_PRECISION] = isDoublePrecision;

	return changed;
}

bool renderer::IImageEvaluator::drawUIOutputChannel(UIStorage_t& storage, bool readOnly)
{
	//selected channel is shared
	ChannelMode selectedChannel = selectedChannel_;
	if (const auto& it = storage.find(UI_KEY_SELECTED_OUTPUT_CHANNEL);
		it != storage.end())
	{
		selectedChannel = std::any_cast<ChannelMode>(it->second);
	}

	bool changed = false;
	const char* currentChannelName = ChannelModeNames[int(selectedChannel)];
	if (readOnly)
	{
		int tmp = static_cast<int>(selectedChannel);
		ImGui::SliderInt("Channel", &tmp,
			0, int(ChannelMode::_ChannelCount_) - 1, currentChannelName);
	}
	else {
		if (ImGui::SliderInt("Channel", reinterpret_cast<int*>(&selectedChannel),
			0, int(ChannelMode::_ChannelCount_) - 1, currentChannelName))
			changed = true;
	}

	selectedChannel_ = selectedChannel;
	storage[UI_KEY_SELECTED_OUTPUT_CHANNEL] = selectedChannel;

	return changed;
}


void renderer::IImageEvaluator::load(const nlohmann::json& json, const ILoadingContext* context)
{
	selectedChannel_ = magic_enum::enum_cast<ChannelMode>(json.value("outputChannel", "")).
		value_or(ChannelMode::ChannelColor);
	isDoublePrecision_ = json.value("doublePrecision", false);
}

void renderer::IImageEvaluator::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["outputChannel"] = magic_enum::enum_name(selectedChannel_);
	json["doublePrecision"] = isDoublePrecision_;
}

void renderer::IImageEvaluator::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;

	namespace py = pybind11;
	py::class_<IImageEvaluator, IImageEvaluator_ptr> c(m, "IImageEvaluator");
	py::enum_<ChannelMode>(c, "ChannelMode")
		.value("Mask", ChannelMode::ChannelMask)
		.value("Normal", ChannelMode::ChannelNormal)
		.value("Depth", ChannelMode::ChannelDepth)
		.value("Color", ChannelMode::ChannelColor)
		.export_values();
	c.def_property("selected_channel", &IImageEvaluator::selectedChannel, &IImageEvaluator::setSelectedChannel)
		.def_readwrite("double_precision", &IImageEvaluator::isDoublePrecision_)
		.def("render", [](IImageEvaluator* self, int width, int height)
			{
				return self->render(width, height, getDefaultStream(), false);
			},
			py::doc(R"(
    Renders the image at the given resolution of width*height.
    For normals to be available, set the output mode to 'Normal'.)"))
		.def("refine", [](IImageEvaluator* self, int width, int height, torch::Tensor previous)
			{
				return self->render(width, height, getDefaultStream(), true, previous);
			},
			py::doc("Refines the image at the given resolution of width*height"))
				.def("compute_batch_count", &IImageEvaluator::computeBatchCount)
				.def("is_iterative_refining", &IImageEvaluator::isIterativeRefining)
				.def("get_supported_tags", &IImageEvaluator::getSupportedTags)
				.def("get_module_for_tag", &IImageEvaluator::getSelectedModuleForTag);
}
