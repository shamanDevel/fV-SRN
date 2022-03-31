#include "transfer_function.h"

#include <filesystem>
#include <c10/cuda/CUDAStream.h>
#include <cuMat/src/Macros.h>

#include "IconsFontAwesome5.h"
#include "portable-file-dialogs.h"
#include "renderer_utils.cuh"
#include "helper_math.cuh"
#include "pytorch_utils.h"
#include "kernel_loader.h"
#include "ray_evaluation_stepping.h"
#include "renderer_tensor.cuh"
#include "volume_interpolation.h"
#include "volume_interpolation_grid.h"

const std::string renderer::ITransferFunction::UI_KEY_ABSORPTION_SCALING
	= "TF::absorptionScaling";
const std::string renderer::ITransferFunction::UI_KEY_COPIED_TF
= "TF::copiedTF";

bool renderer::ITransferFunction::drawUILoadSaveCopyPaste(UIStorage_t& storage)
{
	bool changed = false;
	
	std::string tfDirectory_ = ""; //TODO: load from settings
	if (ImGui::Button(ICON_FA_FOLDER_OPEN "##Load TF"))
	{
		// open file dialog
		auto results = pfd::open_file(
			"Load transfer function",
			tfDirectory_,
			{ "Transfer Function", "*.tf" },
			false
		).result();
		if (results.empty())
			return false;;
		std::string fileNameStr = results[0];

		auto fileNamePath = std::filesystem::path(fileNameStr);
		std::cout << "TF is loaded from " << fileNamePath << std::endl;
		tfDirectory_ = fileNamePath.string();

		//TODO
		throw std::runtime_error("not implemented yet");
		changed = true;
	}
	ImGui::SameLine();
	if (ImGui::Button(ICON_FA_SAVE "##Save TF"))
	{
		// save file dialog
		auto fileNameStr = pfd::save_file(
			"Save transfer function",
			tfDirectory_,
			{ "Transfer Function", "*.tf" },
			true
		).result();
		if (fileNameStr.empty())
			return false;

		auto fileNamePath = std::filesystem::path(fileNameStr);
		fileNamePath = fileNamePath.replace_extension(".tf");
		std::cout << "TF is saved under " << fileNamePath << std::endl;
		tfDirectory_ = fileNamePath.string();

		//TODO
		throw std::runtime_error("not implemented yet");
	}
	ImGui::SameLine();
	if (ImGui::Button(ICON_FA_COPY "##Copy TF"))
	{
		storage[UI_KEY_COPIED_TF] = shared_from_this();
		std::cout << "TF copied (as reference)" << std::endl;
	}
	ImGui::SameLine();
	ITransferFunction_ptr copiedTF = get_or(storage, UI_KEY_COPIED_TF,
		ITransferFunction_ptr());
	bool hasCopiedTF = static_cast<bool>(copiedTF);
	bool canPasteIntoCurrent = hasCopiedTF && canPaste(copiedTF);
	if (ImGui::ButtonEx(ICON_FA_PASTE " (?)##Paste TF", ImVec2(0,0), 
		!canPasteIntoCurrent ? ImGuiButtonFlags_Disabled : 0))
	{
		doPaste(copiedTF);
		storage.erase(UI_KEY_COPIED_TF);
		std::cout << "TF pasted" << std::endl;
		changed = true;
	}
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		if (canPasteIntoCurrent)
			ImGui::TextUnformatted("Pastes the copied TF into the current TF");
		else if (hasCopiedTF)
			ImGui::Text("TF copied, but cannot be pasted into the current TF\nThe TFs '%s' and '%s' are not compatible",
				copiedTF->getName(), this->getName());
		else
			ImGui::TextUnformatted("No TF was copied");
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
	return changed;
}

bool renderer::ITransferFunction::drawUIAbsorptionScaling(UIStorage_t& storage, double& scaling)
{
	if (const auto& it = storage.find(UI_KEY_ABSORPTION_SCALING);
		it != storage.end())
	{
		scaling = std::any_cast<double>(it->second);
	}

	bool changed = false;
	if (ImGui::SliderDouble("Opacity Scaling", &scaling, 1.0f, 500.0f, "%.3f", 2))
	{
		changed = true;
	}
	storage[UI_KEY_ABSORPTION_SCALING] = static_cast<double>(scaling);
	return changed;
}

void renderer::ITransferFunction::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<ITransferFunction, std::shared_ptr<ITransferFunction>>(m, "ITransferFunction")
		.def("evaluate", [](ITransferFunction* self, torch::Tensor densities, double densityMin, double densityMax, std::optional<torch::Tensor> gradient)
			{
				return self->evaluate(densities, densityMin, densityMax, {}, {}, gradient, c10::cuda::getCurrentCUDAStream());
			},
			py::doc("Evaluates the TF on the given density array of shape (B,1) and returns the colors of shape (B,4)"),
				py::arg("densities"), py::arg("min_density"), py::arg("max_density"), py::arg("gradients") = std::optional<torch::Tensor>())
		.def("evaluate_with_previous", [](ITransferFunction* self, torch::Tensor densities, double densityMin, double densityMax,
			torch::Tensor previousDensity, double stepsize, std::optional<torch::Tensor> gradient)
			{
				return self->evaluate(densities, densityMin, densityMax, previousDensity, stepsize, gradient, c10::cuda::getCurrentCUDAStream());
			},
			py::doc("Evaluates the TF on the given density array of shape (B,1) and returns the colors of shape (B,4) with support for preintegratio."),
				py::arg("densities"), py::arg("min_density"), py::arg("max_density"), 
				py::arg("previous_density"), py::arg("stepsize"), py::arg("gradients") = std::optional<torch::Tensor>())
		.def("get_max_absorption", &ITransferFunction::getMaxAbsorption,
			py::doc("Returns the maximal possible absorption per ray differential (i.e. unit step size)."
				"This is used for delta-tracking and importance sampling"))
	    .def("requires_gradients", &ITransferFunction::requiresGradients,
			py::doc("Checks, if the TF requires gradients for evaluation."))
    ;
}

void renderer::ITransferFunction::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
	double stepsize = 1;
	auto rayEval = s.root ? std::dynamic_pointer_cast<IRayEvaluationStepping>(
		s.root->getSelectedModuleForTag(IRayEvaluation::Tag())) : nullptr;
	if (rayEval) {
		stepsize = rayEval->getStepsizeWorld();
	}
	this->fillConstantMemoryTF(s, ptr, stepsize, stream);
}

struct HistogramData
{
	renderer::VolumeInterpolationGrid::HistogramValue* histo;
	int offset;
};
static float histogramGetter(void* data, int idx)
{
	const auto* histo = reinterpret_cast<
		HistogramData*>(data);
	idx += histo->offset;
	if (idx < 0 || idx >= histo->histo->NUM_BINS) return 0.f;
	return static_cast<float>(histo->histo->bins[idx]) / histo->histo->maxBinValue;
}
void renderer::ITransferFunction::drawUIHistogram(UIStorage_t& storage, const ImRect& histogramRect)
{
	//histogram
	VolumeInterpolationGrid::HistogramValue_ptr histogram;
	if (const auto& it = storage.find(VolumeInterpolationGrid::UI_KEY_HISTOGRAM);
		it != storage.end())
	{
		histogram = it->second.has_value()
			? std::any_cast<VolumeInterpolationGrid::HistogramValue_ptr>(it->second)
			: nullptr;
	}
	if (histogram) {
		double minDensity = get_or(storage, IRayEvaluation::UI_KEY_SELECTED_MIN_DENSITY, 0.0);
		double maxDensity = get_or(storage, IRayEvaluation::UI_KEY_SELECTED_MAX_DENSITY, 1.0);
		auto histogramRes = (histogram->maxDensity - histogram->minDensity) / histogram->NUM_BINS;
		int histogramBeginOffset = (minDensity - histogram->minDensity) / histogramRes;
		int histogramEndOffset = (histogram->maxDensity - maxDensity) / histogramRes;
		auto maxElement = std::max_element(std::begin(histogram->bins) + histogramBeginOffset, std::end(histogram->bins) - histogramEndOffset);
	    auto maxFractionVal =
			maxElement
			? static_cast<float>(*maxElement) / histogram->maxBinValue
			: 1.0f;
		HistogramData data{ histogram.get(), histogramBeginOffset };
		ImGui::PlotHistogram("##Histogram", histogramGetter, &data,
			histogram->NUM_BINS - histogramEndOffset - histogramBeginOffset,
			0, nullptr, 0.0f, maxFractionVal, 
			histogramRect.GetSize());
	}
	else
	{
		ImGuiWindow* window = ImGui::GetCurrentWindow();
		ImGuiContext& g = *GImGui;
		const ImGuiStyle& style = g.Style;
		ImGui::ItemSize(histogramRect, style.FramePadding.y);
		ImGui::ItemAdd(histogramRect, window->GetID("TF Editor Histogram Dummy"));
	}
}

bool renderer::ITransferFunction::requiresGradients() const
{
	GlobalSettings s{};
	s.scalarType = GlobalSettings::kFloat;
	s.volumeShouldProvideNormals = false;
	this->prepareRendering(s);
	return s.volumeShouldProvideNormals;
}

torch::Tensor renderer::ITransferFunction::evaluate(
	const torch::Tensor& density, double densityMin, double densityMax,
	const std::optional<torch::Tensor>& previousDensity,
	const std::optional<double>& stepsize, 
	const std::optional<torch::Tensor>& gradient,
	CUstream stream)
{
	CHECK_CUDA(density, true);
	CHECK_DIM(density, 2);
	CHECK_SIZE(density, 1, 1);

	bool hasPrevious;
	if (previousDensity.has_value() && stepsize.has_value())
	{
		hasPrevious = true;
		TORCH_CHECK(stepsize.value() > 0, "Stepsize must be >0");
		CHECK_CUDA(previousDensity.value(), true);
		CHECK_DIM(previousDensity.value(), 2);
		CHECK_SIZE(previousDensity.value(), 1, 1);
		CHECK_SIZE(previousDensity.value(), 0, density.size(0));
	}
	else if (!previousDensity.has_value() && !stepsize.has_value())
	{
		hasPrevious = false;
	}
	else
	{
		throw std::runtime_error("Either specify no previousDensity and no stepsize, or specify both. But not mixed.");
	}

	bool hasGradient = false;
	if (gradient.has_value())
	{
		hasGradient = true;
		CHECK_CUDA(gradient.value(), true);
		CHECK_DIM(gradient.value(), 2);
		CHECK_SIZE(gradient.value(), 1, 3);
		CHECK_SIZE(gradient.value(), 0, density.size(0));
	}

	GlobalSettings s{};
	s.scalarType = density.scalar_type();
	s.volumeShouldProvideNormals = false;

	//kernel
	this->prepareRendering(s);
	if (s.volumeShouldProvideNormals && !hasGradient)
	{
		throw std::runtime_error("The TF requested gradients, but no gradients were passed to the evaluation function");
	}
	std::string kernelName = "EvaluateTF";
	if (hasPrevious) kernelName += "WithPrevious";
	if (hasGradient) kernelName += "WithGradient";
	std::vector<std::string> constantNames;
	if (const auto c = getConstantDeclarationName(s); !c.empty())
		constantNames.push_back(c);
	std::stringstream extraSource;
	extraSource << "#define KERNEL_DOUBLE_PRECISION "
		<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
	extraSource << getDefines(s) << "\n";
	for (const auto& i : getIncludeFileNames(s))
		extraSource << "#include \"" << i << "\"\n";
	extraSource << "#define TRANSFER_FUNCTION_T " <<
		getPerThreadType(s) << "\n";
	extraSource << "#include \"renderer_tf_kernels.cuh\"\n";
	const auto fun = KernelLoader::Instance().getKernelFunction(
		kernelName, extraSource.str(), constantNames, false, false).value();
	if (auto c = getConstantDeclarationName(s); !c.empty())
	{
		CUdeviceptr ptr = fun.constant(c);
		fillConstantMemoryTF(s, ptr, stepsize.value_or(1.0), stream);
	}

	//output tensors
	int batches = density.size(0);
	auto colors = torch::empty({ batches, 4 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));

	//launch kernel
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(batches, fun.bestBlockSize())),
		fun.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(batches), 1, 1 };
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "ITransferFunction::evaluate", [&]()
		{
			const auto accDensity = accessor< ::kernel::Tensor2Read<scalar_t>>(density);
			const auto accColor = accessor< ::kernel::Tensor2RW<scalar_t>>(colors);
			const auto accPrevDensity = hasPrevious
				? accessor<::kernel::Tensor2Read<scalar_t>>(previousDensity.value())
				: kernel::Tensor2Read<scalar_t>{};
			const auto accGradients = hasGradient
				? accessor<::kernel::Tensor2Read<scalar_t>>(gradient.value())
				: kernel::Tensor2Read<scalar_t>{};
			const scalar_t densityMin_s = static_cast<scalar_t>(densityMin);
			const scalar_t densityMax_s = static_cast<scalar_t>(densityMax);
			const scalar_t stepsize_s = hasPrevious ? static_cast<scalar_t>(stepsize.value()) : scalar_t(1);

	        const void* args[] = { &virtual_size,
	            &accDensity, &accPrevDensity, &accGradients, &accColor, 
	            &densityMin_s, &densityMax_s, &stepsize_s };
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, fun.bestBlockSize(), 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return printError(result, kernelName);
			return true;
		});
	if (!success) throw std::runtime_error("Error during rendering!");

	return colors;
}





const std::string renderer::detail::TFPartPiecewiseColor::UI_KEY_SHOW_COLOR_CONTROL_POINTS = "TFPartPiecewiseColor-ShowControlPoints";

renderer::detail::TFPartPiecewiseColor::TFPartPiecewiseColor()
{
	pointsUI_.push_back({ 0, 1,0,0 });
	pointsUI_.push_back({ 1,{1,1,1} });

#if RENDERER_OPENGL_SUPPORT==1
	glBindTexture(GL_TEXTURE_2D, colorMapImage_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ColorMapWidth, 1, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);
#endif

	sortPointsAndUpdateTexture();
}

renderer::detail::TFPartPiecewiseColor::~TFPartPiecewiseColor()
{
#if RENDERER_OPENGL_SUPPORT==1
	glDeleteTextures(1, &colorMapImage_);
#endif
}

bool renderer::detail::TFPartPiecewiseColor::drawUI(const ImRect& rect, bool showControlPoints)
{
	ImGuiWindow* window = ImGui::GetCurrentWindow();

	bool changed = false;
	if (handleIO(rect))
		changed = true;

	if (changed)
		sortPointsAndUpdateTexture();

	auto colorMapWidth = rect.Max.x - rect.Min.x;
	auto colorMapHeight = rect.Max.y - rect.Min.y;
	//draw background
#if RENDERER_OPENGL_SUPPORT==1
	window->DrawList->AddImage((void*)colorMapImage_, rect.Min, rect.Max);
#else
	ImGui::TextColored(ImVec4(1, 0, 0, 1), "OpenGL-support disabled, can't display TF");
#endif

	if (showControlPoints)
	{
		//Draw the control points
		int cpIndex = 0;
		for (const auto& cp : pointsUI_)
		{
			//If this is the selected control point, use different color.
			auto rect2 = createControlPointRect(editorToScreen(cp.position, rect), rect);
			if (selectedControlPointForColor_ == cpIndex++)
			{
				window->DrawList->AddRect(rect2.Min, rect2.Max, ImColor(ImVec4(1.0f, 0.8f, 0.1f, 1.0f)), 16.0f, ImDrawCornerFlags_All, 3.0f);
			}
			else
			{
				window->DrawList->AddRect(rect2.Min, rect2.Max, ImColor(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), 16.0f, ImDrawCornerFlags_All, 2.0f);
			}
		}
	}

	return changed;
}

bool renderer::detail::TFPartPiecewiseColor::drawColorUI()
{
	if (selectedControlPointForColor_<0 || selectedControlPointForColor_>=pointsUI_.size())
	{
		ImGui::TextUnformatted("No point selected");
		return false;
	}
	ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputRGB;
	ImGui::ColorEdit3("##TFPartPiecewiseColor::drawColorUI", &pickedColor_.x, colorFlags);
	return false;
}

bool renderer::detail::TFPartPiecewiseColor::handleIO(const ImRect& rect)
{
	bool isChanged_ = false;

	auto mousePosition = ImGui::GetMousePos();

	if (selectedControlPointForColor_ >= 0)
	{
		auto& cp = pointsUI_[selectedControlPointForColor_];

		double3 pickedColor = make_double3(pickedColor_.x, pickedColor_.y, pickedColor_.z);
		if (any(cp.colorRGB != pickedColor))
		{
			cp.colorRGB = pickedColor;
			isChanged_ = true;
		}
	}

	//Early leave if mouse is not on color editor.
	if (!rect.Contains(mousePosition) && selectedControlPointForMove_ == -1)
	{
		return isChanged_;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
		isChanged_ = true;
		double3 pickedColor = make_double3(pickedColor_.x, pickedColor_.y, pickedColor_.z);
		pointsUI_.push_back({ screenToEditor(mousePosition.x, rect), pickedColor });
	}
	else if (isLeftClicked)
	{
		//Move selected point.
		if (selectedControlPointForMove_ >= 0)
		{
			isChanged_ = true;

			float center = std::min(std::max(mousePosition.x, rect.Min.x), rect.Max.x);

			pointsUI_[selectedControlPointForMove_].position = screenToEditor(center, rect);
		}
		//Check whether new point is selected.
		else
		{
			int size = pointsUI_.size();
			int idx;
			for (idx = 0; idx < size; ++idx)
			{
				auto cp = createControlPointRect(editorToScreen(pointsUI_[idx].position, rect), rect);
				if (cp.Contains(mousePosition))
				{
					selectedControlPointForColor_ = selectedControlPointForMove_ = idx;
					const auto& c = pointsUI_[idx].colorRGB;
					pickedColor_ = ImVec4(c.x, c.y, c.z, 1.0);
					break;
				}
			}

			//In case of no hit on any control point, unselect for color pick as well.
			if (idx == size)
			{
				selectedControlPointForColor_ = -1;
			}
		}
	}
	else if (isRightClicked)
	{
		int size = pointsUI_.size();
		int idx;
		for (idx = 0; idx < size; ++idx)
		{
			auto cp = createControlPointRect(editorToScreen(pointsUI_[idx].position, rect), rect);
			if (cp.Contains(mousePosition) && pointsUI_.size() > 1)
			{
				isChanged_ = true;

				pointsUI_.erase(pointsUI_.begin() + idx);
				selectedControlPointForColor_ = selectedControlPointForMove_ = -1;
				break;
			}
		}
	}
	else if (isLeftReleased)
	{
		selectedControlPointForMove_ = -1;
	}
	return isChanged_;
}

void renderer::detail::TFPartPiecewiseColor::updateControlPoints(const std::vector<Point>& points)
{
	selectedControlPointForMove_ = -1;
	selectedControlPointForColor_ = -1;
	pointsUI_.clear();
	pointsUI_.insert(pointsUI_.end(), points.begin(), points.end());
	sortPointsAndUpdateTexture();
}

std::vector<double3> renderer::detail::TFPartPiecewiseColor::getAsTexture(
	int resolution, ColorSpace colorSpace) const
{
	std::vector<double3> pixelData(resolution);
	int numPoints = static_cast<int>(pointsSorted_.size());

	for (int i = 0; i < resolution; ++i)
	{
		float density = (i + 0.5f) / resolution;
		//find control point
		int idx;
		for (idx = 0; idx < numPoints - 2; ++idx)
			if (pointsSorted_[idx + 1].position > density) break;
		//interpolate
		const float pLow = pointsSorted_[idx].position;
		const float pHigh = pointsSorted_[idx + 1].position;
		const auto vLow = pointsSorted_[idx].colorRGB;
		const auto vHigh = pointsSorted_[idx + 1].colorRGB;

		const float frac = clamp(
			(density - pLow) / (pHigh - pLow),
			0.0f, 1.0f);
		const auto v = (1 - frac) * vLow + frac * vHigh;
		pixelData[i] = v;
	}
	return pixelData;
}

ImRect renderer::detail::TFPartPiecewiseColor::createControlPointRect(float x, const ImRect& rect)
{
	return ImRect(ImVec2(x - 0.5f * cpWidth_, rect.Min.y),
		ImVec2(x + 0.5f * cpWidth_, rect.Max.y));
}

float renderer::detail::TFPartPiecewiseColor::screenToEditor(float screenPositionX, const ImRect& rect)
{
	float editorPositionX;
	editorPositionX = (screenPositionX - rect.Min.x) / (rect.Max.x - rect.Min.x);

	return editorPositionX;
}

float renderer::detail::TFPartPiecewiseColor::editorToScreen(float editorPositionX, const ImRect& rect)
{
	float screenPositionX;
	screenPositionX = editorPositionX * (rect.Max.x - rect.Min.x) + rect.Min.x;

	return screenPositionX;
}

void renderer::detail::TFPartPiecewiseColor::sortPointsAndUpdateTexture()
{
	pointsSorted_.clear();
	pointsSorted_.insert(pointsSorted_.end(), pointsUI_.begin(), pointsUI_.end());
	std::sort(pointsSorted_.begin(), pointsSorted_.end(),
		[](const Point& p1, const Point& p2)
		{
			return p1.position < p2.position;
		});


	std::vector<double3> tex = getAsTexture(ColorMapWidth, ColorSpace::RGB);
	std::vector<unsigned int> pixelData(ColorMapWidth);
	for (int i = 0; i < ColorMapWidth; ++i)
	{
		pixelData[i] = kernel::rgbaToInt(tex[i].x, tex[i].y, tex[i].z, 1.f);
	}

#if RENDERER_OPENGL_SUPPORT==1
	glBindTexture(GL_TEXTURE_2D, colorMapImage_);
	glTexSubImage2D(
		GL_TEXTURE_2D, 0, 0, 0, ColorMapWidth, 1, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV,
		pixelData.data());
	glBindTexture(GL_TEXTURE_2D, 0);
#endif
}

renderer::detail::TfPartPiecewiseOpacity::TfPartPiecewiseOpacity()
	: pointsUI_({{0,0}, {1,1}})
{
	sortPoints();
}

bool renderer::detail::TfPartPiecewiseOpacity::drawUI(const ImRect& rect)
{
	bool changed = handleIO(rect);
	render(rect);
	return changed;
}

void renderer::detail::TfPartPiecewiseOpacity::updateControlPoints(const std::vector<Point>& points)
{
	pointsUI_.clear();
	pointsUI_.insert(pointsUI_.end(), points.begin(), points.end());
	sortPoints();
}

bool renderer::detail::TfPartPiecewiseOpacity::handleIO(const ImRect& rect)
{
	bool isChanged_ = false;

	auto mousePosition = ImGui::GetMousePos();

	//Early leave if mouse is not on opacity editor and no control point is selected.
	if (!rect.Contains(mousePosition) && selectedControlPoint_ == -1)
	{
		return isChanged_;
	}

	//0=left, 1=right, 2=middle
	bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isRightClicked = ImGui::IsMouseClicked(1);
	bool isLeftReleased = ImGui::IsMouseReleased(0);

	if (isLeftDoubleClicked)
	{
		isChanged_ = true;

		pointsUI_.push_back(screenToEditor(mousePosition, rect));
	}
	else if (isLeftClicked)
	{
		//Move selected point.
		if (selectedControlPoint_ >= 0)
		{
			isChanged_ = true;

			ImVec2 center(std::min(std::max(mousePosition.x, rect.Min.x), rect.Max.x),
				std::min(std::max(mousePosition.y, rect.Min.y), rect.Max.y));

			pointsUI_[selectedControlPoint_] = screenToEditor(center, rect);
		}
		//Check whether new point is selected.
		else
		{
			int size = pointsUI_.size();
			for (int idx = 0; idx < size; ++idx)
			{
				auto cp = createControlPointRect(editorToScreen(pointsUI_[idx], rect));
				if (cp.Contains(mousePosition))
				{
					selectedControlPoint_ = idx;
					break;
				}
			}
		}
	}
	else if (isRightClicked)
	{
		int size = pointsUI_.size();
		for (int idx = 0; idx < size; ++idx)
		{
			auto cp = createControlPointRect(editorToScreen(pointsUI_[idx], rect));
			if (cp.Contains(mousePosition) && pointsUI_.size() > 1)
			{
				isChanged_ = true;

				pointsUI_.erase(pointsUI_.begin() + idx);
				selectedControlPoint_ = -1;
				break;
			}
		}
	}
	else if (isLeftReleased)
	{
		selectedControlPoint_ = -1;
	}

	if (isChanged_)
		sortPoints();
	return isChanged_;
}

void renderer::detail::TfPartPiecewiseOpacity::render(const ImRect& rect)
{
	//Draw the bounding rectangle.
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	window->DrawList->AddRect(rect.Min, rect.Max, ImColor(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)), 0.0f, ImDrawCornerFlags_All, 1.0f);

	std::vector<ImVec2> controlPointsRender;
	std::transform(pointsSorted_.begin(), pointsSorted_.end(),
		std::back_inserter(controlPointsRender),
		[this, rect](const Point& p)
		{
			return editorToScreen(p, rect);
		});

	//Draw lines between the control points.
	int size = controlPointsRender.size();
	for (int i = 0; i < size + 1; ++i)
	{
		auto left = (i == 0) ? ImVec2(rect.Min.x, controlPointsRender.front().y) : controlPointsRender[i - 1];
		auto right = (i == size) ? ImVec2(rect.Max.x, controlPointsRender.back().y) : controlPointsRender[i];

		window->DrawList->AddLine(left, right, ImColor(ImVec4(1.0f, 1.0f, 1.0f, 1.0f)), 1.0f);
	}

	//Draw the control points
	for (const auto& cp : controlPointsRender)
	{
		window->DrawList->AddCircleFilled(cp, circleRadius_, ImColor(ImVec4(0.0f, 1.0f, 0.0f, 1.0f)), 16);
	}
}

ImRect renderer::detail::TfPartPiecewiseOpacity::createControlPointRect(const ImVec2& controlPoint)
{
	return ImRect(ImVec2(controlPoint.x - circleRadius_, controlPoint.y - circleRadius_),
		ImVec2(controlPoint.x + circleRadius_, controlPoint.y + circleRadius_));
}

renderer::detail::TfPartPiecewiseOpacity::Point renderer::detail::TfPartPiecewiseOpacity::screenToEditor(const ImVec2& screenPosition, const ImRect& rect)
{
	float editorPositionX = (screenPosition.x - rect.Min.x) / (rect.Max.x - rect.Min.x);
	float editorPositionY = 1 - (screenPosition.y - rect.Min.y) / (rect.Max.y - rect.Min.y);
	return { editorPositionX, editorPositionY };
}

ImVec2 renderer::detail::TfPartPiecewiseOpacity::editorToScreen(const Point& editorPosition, const ImRect& rect)
{
	float screenPositionX = editorPosition.position * (rect.Max.x - rect.Min.x) + rect.Min.x;
	float screenPositionY = (1-editorPosition.absorption) * (rect.Max.y - rect.Min.y) + rect.Min.y;
	return { screenPositionX, screenPositionY };
}

void renderer::detail::TfPartPiecewiseOpacity::sortPoints()
{
	pointsSorted_.clear();
	pointsSorted_.insert(pointsSorted_.end(), pointsUI_.begin(), pointsUI_.end());
	std::sort(pointsSorted_.begin(), pointsSorted_.end(),
		[](const Point& p1, const Point& p2)
		{
			return p1.position < p2.position;
		});
}
