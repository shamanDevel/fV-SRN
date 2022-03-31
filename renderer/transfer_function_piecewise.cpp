#include "ray_evaluation.h"
#include "transfer_function.h"
#include "volume.h"
#include "volume_interpolation.h"

#include "helper_math.cuh"
#include "pytorch_utils.h"
#include "renderer_tensor.cuh"
#include "renderer_tf_piecewise.cuh"

renderer::TransferFunctionPiecewiseLinear::TransferFunctionPiecewiseLinear()
	: absorptionScaling_(10)
{
	computeTensor();
}

renderer::TransferFunctionPiecewiseLinear::~TransferFunctionPiecewiseLinear()
{
}

std::string renderer::TransferFunctionPiecewiseLinear::getName() const
{
	return "Piecewise";
}

bool renderer::TransferFunctionPiecewiseLinear::drawUI(UIStorage_t& storage)
{
	bool changed = false;

	if (drawUILoadSaveCopyPaste(storage))
		changed = true;
	ImGui::SameLine();
	bool showColorControlPoints = get_or(storage, detail::TFPartPiecewiseColor::UI_KEY_SHOW_COLOR_CONTROL_POINTS, false);
	ImGui::Checkbox("Show CPs", &showColorControlPoints);
	storage[detail::TFPartPiecewiseColor::UI_KEY_SHOW_COLOR_CONTROL_POINTS] = showColorControlPoints;

	ImGuiWindow* window = ImGui::GetCurrentWindow();
	ImGuiContext& g = *GImGui;
	const ImGuiStyle& style = g.Style;

	//Color
	const ImGuiID tfEditorColorId = window->GetID("TF Editor Color");
	auto pos = window->DC.CursorPos;
	auto tfEditorColorWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
	auto tfEditorColorHeight = 50.0f;
	const ImRect tfEditorColorRect(pos, ImVec2(pos.x + tfEditorColorWidth, pos.y + tfEditorColorHeight));
	ImGui::ItemSize(tfEditorColorRect, style.FramePadding.y);
	ImGui::ItemAdd(tfEditorColorRect, tfEditorColorId);

	//Opacity
	const ImGuiID tfEditorOpacityId = window->GetID("TF Editor Opacity");
	pos = window->DC.CursorPos;
	auto tfEditorOpacityWidth = window->WorkRect.Max.x - window->WorkRect.Min.x;
	auto tfEditorOpacityHeight = 100.0f;
	const ImRect tfEditorOpacityRect(pos, ImVec2(pos.x + tfEditorOpacityWidth, pos.y + tfEditorOpacityHeight));

	//histogram
	drawUIHistogram(storage, tfEditorOpacityRect);

	//color
	if (colorEditor_.drawUI(tfEditorColorRect, showColorControlPoints))
		changed = true;

	//absorption
	if (opacityEditor_.drawUI(tfEditorOpacityRect))
		changed = true;

	if (colorEditor_.drawColorUI())
		changed = true;
	
	if (drawUIAbsorptionScaling(storage, absorptionScaling_))
		changed = true;

	if (changed) {
		computeTensor();
	}
	return changed;
}

void renderer::TransferFunctionPiecewiseLinear::load(const nlohmann::json& json, const ILoadingContext* context)
{
	std::vector<renderer::detail::TFPartPiecewiseColor::Point> cpoints =
		json["colorPoints"];
	colorEditor_.updateControlPoints(cpoints);
	std::vector<renderer::detail::TfPartPiecewiseOpacity::Point> opoints =
		json["opacityPoints"];
	opacityEditor_.updateControlPoints(opoints);
	absorptionScaling_ = json["absorptionScaling"];

	computeTensor();
}

void renderer::TransferFunctionPiecewiseLinear::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["colorPoints"] = colorEditor_.getPoints();
	json["opacityPoints"] = opacityEditor_.getPoints();
	json["absorptionScaling"] = absorptionScaling_;
}

void renderer::TransferFunctionPiecewiseLinear::registerPybindModule(pybind11::module& m)
{
	ITransferFunction::registerPybindModule(m);

	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<TransferFunctionPiecewiseLinear, ITransferFunction, std::shared_ptr<TransferFunctionPiecewiseLinear>>(m, "TransferFunctionPiecewiseLinear")
		.def_readonly("tensor", &TransferFunctionPiecewiseLinear::textureTensor_);
}

std::optional<int> renderer::TransferFunctionPiecewiseLinear::getBatches(const GlobalSettings& s) const
{
	if (!textureTensor_.value.defined())
		throw std::runtime_error("Empty Texture TF!");
	int b = textureTensor_.value.size(0);
	if (b > 1)
		return b;
	else
		return {};
}

std::vector<std::string> renderer::TransferFunctionPiecewiseLinear::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_tf_piecewise.cuh" };
}

std::string renderer::TransferFunctionPiecewiseLinear::getDefines(const GlobalSettings& s) const
{
	return "";
}

std::string renderer::TransferFunctionPiecewiseLinear::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "transferFunctionPiecewiseParameters";
}

std::string renderer::TransferFunctionPiecewiseLinear::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::TransferFunctionPiecewise";
}

void renderer::TransferFunctionPiecewiseLinear::fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr,
	double stepsize, CUstream stream)
{
	if (!textureTensor_.value.defined())
		throw std::runtime_error("texture tensor is not defined");
	CHECK_CUDA(textureTensor_.value, true);
	CHECK_DIM(textureTensor_.value, 3);
	CHECK_SIZE(textureTensor_.value, 2, 5);
	if (textureTensor_.value.scalar_type() != s.scalarType)
		textureTensor_.value = textureTensor_.value.to(s.scalarType);
	RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "TransferFunctionPiecewiseLinear", [&]()
		{
			struct Parameters
			{
				::kernel::Tensor3Read<scalar_t> tex;
			} p;
			p.tex = accessor<::kernel::Tensor3Read<scalar_t>>(textureTensor_.value);
			CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
		});
}

void renderer::TransferFunctionPiecewiseLinear::computeTensor()
{
	//assemble merged points
	struct TFPoint
	{
		double pos;
		double3 rgb;
		double absorption;
	};
	std::vector<TFPoint> points;

	auto colorPoints = colorEditor_.getPoints();
	auto opacityPoints = opacityEditor_.getPoints();

	//add control points at t=0 if not existing
	//but not directly 0 and 1, better -1 and 2 as this is better for the iso-intersections
	if (colorPoints.front().position > 0)
	{
		colorPoints.insert(colorPoints.begin(), { -1.0, colorPoints.front().colorRGB });
	}
	if (opacityPoints.front().position > 0)
	{
		opacityPoints.insert(opacityPoints.begin(), { -1.0, opacityPoints.front().absorption });
	}
	//same with t=1
	if (colorPoints.back().position < 1)
	{
		colorPoints.insert(colorPoints.end(), { 2.0, colorPoints.back().colorRGB });
	}
	if (opacityPoints.back().position < 1)
	{
		opacityPoints.insert(opacityPoints.end(), { 2.0, opacityPoints.back().absorption });
	}

	//first point at pos=0
	if (colorPoints[0].position <= opacityPoints[0].position)
		points.push_back({ colorPoints[0].position, colorPoints[0].colorRGB, opacityPoints[0].absorption });
	else
		points.push_back({ opacityPoints[0].position, colorPoints[0].colorRGB, opacityPoints[0].absorption });

	//assemble the points
	int64_t iOpacity = 0; //next indices
	int64_t iColor = 0;
	while (iOpacity < opacityPoints.size() - 1 && iColor < colorPoints.size() - 1)
	{
		if (opacityPoints[iOpacity + 1].position < colorPoints[iColor + 1].position)
		{
			//next point is an opacity point
			double f = (opacityPoints[iOpacity + 1].position - colorPoints[iColor].position) /
				(colorPoints[iColor + 1].position - colorPoints[iColor].position);
			points.push_back({
				opacityPoints[iOpacity + 1].position ,
				lerp(colorPoints[iColor].colorRGB,colorPoints[iColor + 1].colorRGB, f),
				opacityPoints[iOpacity + 1].absorption
				});
			iOpacity++;
		}
		else
		{
			double f = (colorPoints[iColor + 1].position - opacityPoints[iOpacity].position) /
				(opacityPoints[iOpacity + 1].position - opacityPoints[iOpacity].position);
			points.push_back({
				colorPoints[iColor + 1].position ,
				colorPoints[iColor + 1].colorRGB,
				lerp(opacityPoints[iOpacity].absorption,opacityPoints[iOpacity + 1].absorption, f)
				});
			iColor++;
		}

	}

	//filter the points
	//removes all color control points in areas of zero density
	//and also remove points that are close together
	constexpr bool purgeZeroOpacityRegions = true;
	if (purgeZeroOpacityRegions) {
		int numPointsRemoved = 0;
		constexpr float EPS = 1e-7;
		for (int64_t i = 0; i < static_cast<int64_t>(points.size()) - 2; )
		{
			if ((points[i].absorption < EPS && points[i + 1].absorption < EPS && points[i + 2].absorption < EPS) ||
				(points[i + 1].pos - points[i].pos < EPS)) {
				points.erase(points.begin() + (i + 1));
				numPointsRemoved++;
			}
			else
				i++;
		}
		//std::cout << numPointsRemoved << " points removed with zero density" << std::endl;
	}

	//clamp color and scale opacity
	for (TFPoint& p : points)
	{
		p.rgb = clamp(p.rgb, 0.0f, 1.0f - FLT_EPSILON);
		p.absorption = clamp(p.absorption, 0.0, 1.0)*absorptionScaling_;
	}

	//to PyTorch tensor
	int numPoints = static_cast<int>(points.size());
	textureTensorCpu_ = torch::empty(
		{ 1, numPoints, 5 },
		at::TensorOptions().dtype(c10::kFloat));
	auto acc = textureTensorCpu_.packed_accessor32<float, 3>();
	for (int i = 0; i < numPoints; ++i)
	{
		acc[0][i][0] = points[i].rgb.x;
		acc[0][i][1] = points[i].rgb.y;
		acc[0][i][2] = points[i].rgb.z;
		acc[0][i][3] = points[i].absorption;
		acc[0][i][4] = points[i].pos;
	}
	textureTensor_.value = textureTensorCpu_.to(c10::kCUDA);
	textureTensor_.forwardIndex = {};
	textureTensor_.grad = {};
}

double4 renderer::TransferFunctionPiecewiseLinear::evaluate(double density) const
{
	auto tf = accessor<::kernel::Tensor3Read<float>>(textureTensorCpu_);
	return make_double4(kernel::TransferFunctionPiecewise::sampleTF<float>(density, tf, 0));
}

double renderer::TransferFunctionPiecewiseLinear::getMaxAbsorption() const
{
	return absorptionScaling_;
}
