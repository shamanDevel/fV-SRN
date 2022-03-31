#include "transfer_function.h"

#include "ray_evaluation.h"
#include "volume.h"
#include "volume_interpolation.h"

#include "helper_math.cuh"
#include "pytorch_utils.h"
#include "renderer_tensor.cuh"

renderer::TransferFunctionGaussian::TransferFunctionGaussian()
	: points_{
		Point{ImVec4(1,0,0,alpha1_), 0.6f, 0.7f, 0.05f},
		Point{ImVec4(0,1,0,alpha1_), 0.3f, 0.3f, 0.03f}
		}
	, absorptionScaling_(10)
	, scaleWithGradient_(false)
{
	computeTensor();
}

renderer::TransferFunctionGaussian::~TransferFunctionGaussian()
{
}

std::string renderer::TransferFunctionGaussian::getName() const
{
	return "Gaussian";
}

bool renderer::TransferFunctionGaussian::drawUI(UIStorage_t& storage)
{
	bool changed = false;

	if (drawUILoadSaveCopyPaste(storage))
		changed = true;

	ImGuiWindow* window = ImGui::GetCurrentWindow();
	ImGuiContext& g = *GImGui;
	const ImGuiStyle& style = g.Style;

	//Opacity
	const ImGuiID tfEditorOpacityId = window->GetID("TF Editor Opacity");
	auto pos = window->DC.CursorPos;
	auto tfEditorOpacityWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
	auto tfEditorOpacityHeight = 100.0f;
	const ImRect tfEditorOpacityRect(pos, ImVec2(pos.x + tfEditorOpacityWidth, pos.y + tfEditorOpacityHeight));

	//histogram
	drawUIHistogram(storage, tfEditorOpacityRect);

	if (renderAndIO(tfEditorOpacityRect))
		changed = true;

	if (drawUIAbsorptionScaling(storage, absorptionScaling_))
		changed = true;

	if (ImGui::Checkbox("Scale with gradient##TransferFunctionGaussian", &scaleWithGradient_))
		changed = true;
	if (ImGui::Checkbox("Analytic Integration##TransferFunctionGaussian", &usePiecewiseAnalyticIntegration_))
		changed = true;

	if (changed) {
		computeTensor();
	}
	return changed;
}


bool renderer::TransferFunctionGaussian::renderAndIO(const ImRect& rect)
{
	//Draw the bounding rectangle.
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	window->DrawList->AddRect(rect.Min, rect.Max, ImColor(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)), 0.0f, ImDrawCornerFlags_All, 1.0f);

	//draw lines
	static const int Samples = 100;
	const int numPoints = points_.size();
	if (numPoints == 0) selectedPoint_ = -1;
	//mixture
	const auto eval = [this, numPoints](float x)
	{
		float opacity = 0;
		float3 color = make_float3(0, 0, 0);
		for (int i = 0; i < numPoints; ++i) {
			float w = points_[i].opacity * gaussian(x, points_[i].mean, points_[i].variance);
			opacity += w;
			color += make_float3(points_[i].color.x, points_[i].color.y, points_[i].color.z) * w;
		}
		color /= opacity;
		return std::make_pair(opacity, ImGui::GetColorU32(ImVec4(color.x, color.y, color.z, alpha2_)));
	};
	auto [o0, c0] = eval(0);
	float x0 = 0;
	for (int j = 1; j <= Samples; ++j)
	{
		float x1 = j / float(Samples);
		const auto [o1, c1] = eval(x1);

		ImVec2 a = ImLerp(rect.Min, rect.Max,
			ImVec2(x0, 1 - o0));
		ImVec2 b = ImLerp(rect.Min, rect.Max,
			ImVec2(x1, 1 - o1));
		window->DrawList->AddLine(a, b, c0, thickness2_);

		x0 = x1;
		o0 = o1;
		c0 = c1;
	}
	//single gaussians
	for (int i = 0; i < numPoints; ++i)
	{
		ImU32 col = ImGui::GetColorU32(points_[i].color);
		float opacity = points_[i].opacity;
		float mean = points_[i].mean;
		float variance = points_[i].variance;
		float denom = 1.0f / (variance * variance);
		float y0 = opacity * expf(-mean * mean * denom);
		float x0 = 0;
		for (int j = 1; j <= Samples; ++j)
		{
			float x1 = j / float(Samples);
			float y1 = opacity * expf(-(x1 - mean) * (x1 - mean) * denom);
			if (y0 > 1e-3 || y1 > 1e-3) {
				ImVec2 a = ImLerp(rect.Min, rect.Max,
					ImVec2(x0, 1 - y0));
				ImVec2 b = ImLerp(rect.Min, rect.Max,
					ImVec2(x1, 1 - y1));
				window->DrawList->AddLine(a, b, col, thickness1_);
			}
			y0 = y1; x0 = x1;
		}
	}

	//draw control points
	float minDistance = FLT_MAX;
	int bestSelection = -1;
	auto mousePosition = ImGui::GetMousePos();
	for (int i = 0; i < numPoints; ++i)
	{
		ImU32 col = ImGui::GetColorU32(points_[i].color);
		float opacity = points_[i].opacity;
		float mean = points_[i].mean;
		auto cp = editorToScreen(ImVec2(mean, opacity), rect);
		window->DrawList->AddCircleFilled(cp, circleRadius_, col, 16);
		window->DrawList->AddCircle(cp, circleRadius_ + 1, ImGui::GetColorU32(ImVec4(0, 0, 0, 1)));
		if (i == selectedPoint_)
			window->DrawList->AddCircle(cp, circleRadius_ + 2, ImGui::GetColorU32(ImGuiCol_Text));
		float dist = ImLengthSqr(ImVec2(mousePosition.x - cp.x, mousePosition.y - cp.y));
		if (dist < minDistance)
		{
			minDistance = dist;
			bestSelection = i;
		}
	}

	//handle io
	bool changed = false;
	if (rect.Contains(mousePosition)
		|| draggedPoint_ != -1)
	{
		bool isLeftDoubleClicked = ImGui::IsMouseDoubleClicked(0);
		bool isLeftClicked = ImGui::IsMouseDown(0);
		bool isRightClicked = ImGui::IsMouseClicked(1);
		bool isLeftReleased = ImGui::IsMouseReleased(0);

		auto mouseEditor = screenToEditor(mousePosition, rect);

		if (isLeftDoubleClicked)
		{
			changed = true;
			points_.push_back({
				ImVec4(1,1,1,alpha1_),
				mouseEditor.y,
				mouseEditor.x,
				0.01f
				}
			);
			selectedPoint_ = points_.size() - 1;
			draggedPoint_ = -1;
		}
		else if (isLeftClicked)
		{
			if (draggedPoint_ >= 0)
			{
				//move selected point
				changed = true;
				ImVec2 center(std::min(std::max(mousePosition.x, rect.Min.x), rect.Max.x),
					std::min(std::max(mousePosition.y, rect.Min.y), rect.Max.y));
				auto p = screenToEditor(center, rect);
				points_[draggedPoint_].mean = p.x;
				points_[draggedPoint_].opacity = p.y;
			}
			else
			{
				//select new point for hovering
				if (minDistance < (circleRadius_ + 2) * (circleRadius_ + 2))
				{
					draggedPoint_ = bestSelection;
					selectedPoint_ = bestSelection;
				}
			}
		}
		else if (isRightClicked)
		{
			if (minDistance < (circleRadius_ + 2) * (circleRadius_ + 2))
			{
				changed = true;
				points_.erase(points_.begin() + bestSelection);
				selectedPoint_ = points_.empty() ? -1 : 0;
				draggedPoint_ = -1;
			}
		}
		else if (isLeftReleased)
		{
			draggedPoint_ = -1;
		}
	}

	//controls
	if (selectedPoint_ >= 0) {
		ImGuiColorEditFlags colorFlags = ImGuiColorEditFlags_Float | ImGuiColorEditFlags_InputRGB;
		if (ImGui::ColorEdit3("", &points_[selectedPoint_].color.x, colorFlags))
			changed = true;
		if (ImGui::SliderFloat("Variance", &points_[selectedPoint_].variance,
			0.001f, 0.5f, "%.3f", 2))
			changed = true;
	}

	return changed;
}


void renderer::TransferFunctionGaussian::load(const nlohmann::json& json, const ILoadingContext* context)
{
	points_ = json["points"].get<std::vector<Point>>();
	absorptionScaling_ = json["absorptionScaling"];
	scaleWithGradient_ = json.value("scaleWithGradient", false);
	usePiecewiseAnalyticIntegration_ = json.value("usePiecewiseAnalyticIntegration", false);

	computeTensor();
}

void renderer::TransferFunctionGaussian::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["points"] = points_;
	json["absorptionScaling"] = absorptionScaling_;
	json["scaleWithGradient"] = scaleWithGradient_;
	json["usePiecewiseAnalyticIntegration"] = usePiecewiseAnalyticIntegration_;
}

void renderer::TransferFunctionGaussian::registerPybindModule(pybind11::module& m)
{
	ITransferFunction::registerPybindModule(m);
	
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<TransferFunctionGaussian, ITransferFunction, std::shared_ptr<TransferFunctionGaussian>>(m, "TransferFunctionGaussian")
		.def_readonly("tensor", &TransferFunctionGaussian::textureTensor_)
	    .def_readwrite("absorption_scaling", &TransferFunctionGaussian::absorptionScaling_)
	    .def_readwrite("piecewise_analytic_integraton", &TransferFunctionGaussian::usePiecewiseAnalyticIntegration_)
    ;
}

void renderer::TransferFunctionGaussian::prepareRendering(GlobalSettings& s) const
{
	if (scaleWithGradient_) {
		s.volumeShouldProvideNormals = true;
		//std::cout << "Gaussian TF: scale with gradients enabled, request that the volume provides normals" << std::endl;
	}
}

std::optional<int> renderer::TransferFunctionGaussian::getBatches(const GlobalSettings& s) const
{
	if (!textureTensor_.value.defined())
		throw std::runtime_error("Empty Texture TF!");
	int b = textureTensor_.value.size(0);
	if (b > 1)
		return b;
	else
		return {};
}

std::vector<std::string> renderer::TransferFunctionGaussian::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_tf_gaussian.cuh" };
}

std::string renderer::TransferFunctionGaussian::getDefines(const GlobalSettings& s) const
{
	std::stringstream ss;
	if (scaleWithGradient_ && usePiecewiseAnalyticIntegration_)
		throw std::runtime_error("Gaussian TF: gradient scaling and piecewise analytic integration are incompatible");
	if (scaleWithGradient_)
		ss << "#define TRANSFER_FUNCTION_GAUSSIAN__SCALE_WITH_GRADIENT\n";
	ss << "#define TRANSFER_FUNCTION_GAUSSIAN__ANALYTIC "
		<< (usePiecewiseAnalyticIntegration_ ? "1" : "0") << "\n";
	return ss.str();
}

std::string renderer::TransferFunctionGaussian::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "transferFunctionGaussianParameters";
}

std::string renderer::TransferFunctionGaussian::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::TransferFunctionGaussian";
}

void renderer::TransferFunctionGaussian::fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr, double stepsize, CUstream stream)
{
	if (!textureTensor_.value.defined())
		throw std::runtime_error("texture tensor is not defined");
	CHECK_CUDA(textureTensor_.value, true);
	CHECK_DIM(textureTensor_.value, 3);
	CHECK_SIZE(textureTensor_.value, 2, 6);
	if (textureTensor_.value.scalar_type() != s.scalarType)
		textureTensor_.value = textureTensor_.value.to(s.scalarType);
	RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "TransferFunctionGaussian", [&]()
		{
			struct Parameters
			{
				::kernel::Tensor3Read<scalar_t> tex;
			} p;
			p.tex = accessor<::kernel::Tensor3Read<scalar_t>>(textureTensor_.value);
			CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
		});
}

float renderer::TransferFunctionGaussian::gaussian(float x, float mean, float variance)
{
	return expf(-(x - mean) * (x - mean) / (variance * variance));
}

void renderer::TransferFunctionGaussian::computeTensor()
{
	int numPoints = static_cast<int>(points_.size());
	torch::Tensor t = torch::empty(
		{ 1, numPoints, 6 },
		at::TensorOptions().dtype(c10::kFloat));
	auto acc = t.packed_accessor32<float, 3>();
	for (int i = 0; i < numPoints; ++i)
	{
		acc[0][i][0] = points_[i].color.x;
		acc[0][i][1] = points_[i].color.y;
		acc[0][i][2] = points_[i].color.z;
		acc[0][i][3] = points_[i].opacity * absorptionScaling_;
		acc[0][i][4] = points_[i].mean;
		acc[0][i][5] = points_[i].variance;
	}

	textureTensor_.value = t.to(c10::kCUDA);
	textureTensor_.forwardIndex = {};
	textureTensor_.grad = {};
}

ImVec2 renderer::TransferFunctionGaussian::screenToEditor(const ImVec2& screenPosition, const ImRect& rect)
{
	float editorPositionX = (screenPosition.x - rect.Min.x) / (rect.Max.x - rect.Min.x);
	float editorPositionY = 1 - (screenPosition.y - rect.Min.y) / (rect.Max.y - rect.Min.y);
	return { editorPositionX, editorPositionY };
}

ImVec2 renderer::TransferFunctionGaussian::editorToScreen(const ImVec2& editorPosition, const ImRect& rect)
{
	float screenPositionX = editorPosition.x * (rect.Max.x - rect.Min.x) + rect.Min.x;
	float screenPositionY = (1 - editorPosition.y) * (rect.Max.y - rect.Min.y) + rect.Min.y;
	return { screenPositionX, screenPositionY };
}

double4 renderer::TransferFunctionGaussian::evaluate(double density) const
{
	double4 c = make_double4(0);
	for (const auto& p : points_)
	{
		double ni = gaussian(density, p.mean, p.variance);
		c += make_double4(p.color.x, p.color.y, p.color.z, p.opacity * absorptionScaling_) * ni;
	}
	return c;
}

double renderer::TransferFunctionGaussian::getMaxAbsorption() const
{
	if (textureTensor_.value.defined() && textureTensor_.value.size(0) > 1)
		std::cerr << "TF is batched, absorption scaling is most likely not accurate anymore" << std::endl;
	return absorptionScaling_;
}
