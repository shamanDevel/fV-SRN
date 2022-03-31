#include "transfer_function.h"

#include <algorithm>
#include <magic_enum.hpp>
#include <cuMat/src/Errors.h>

#include "helper_math.cuh"
#include "volume_interpolation.h"
#include "ray_evaluation.h"
#include "renderer_tensor.cuh"
#include "pytorch_utils.h"
#include "ray_evaluation_stepping.h"

renderer::TransferFunctionTexture::TransferFunctionTexture()
	: useTensor_(false)
	, plot_(RESOLUTION)
	, textureArray_(0), textureObject_(0)
	, absorptionScaling_(10)
{
	for (int i = 0; i < RESOLUTION; ++i)
		plot_[i] = i / (RESOLUTION - 1.0);

	//regular texture
	auto desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
	CUMAT_SAFE_CALL(cudaMallocArray(&textureArray_, &desc, RESOLUTION));
	
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = textureArray_;
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&textureObject_, &resDesc, &texDesc, NULL));

	//preintegration table
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat); //XYZA

	CUMAT_SAFE_CALL(cudaMallocArray(&preintegrationCudaArray1D_, &channelDesc, preintegrationResolution_, 0, cudaArraySurfaceLoadStore));
	CUMAT_SAFE_CALL(cudaMallocArray(&preintegrationCudaArray2D_, &channelDesc, preintegrationResolution_, preintegrationResolution_, cudaArraySurfaceLoadStore));

	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaTextureAddressMode(cudaAddressModeClamp);
	texDesc.addressMode[1] = cudaTextureAddressMode(cudaAddressModeClamp);
	texDesc.filterMode = cudaTextureFilterMode(cudaFilterModeLinear);
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 1;

	//Create the surface and texture object.
	resDesc.res.array.array = preintegrationCudaArray1D_;
	CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&preintegrationCudaSurface1D_, &resDesc));
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&preintegrationCudaTexture1D_, &resDesc, &texDesc, nullptr));
	resDesc.res.array.array = preintegrationCudaArray2D_;
	CUMAT_SAFE_CALL(cudaCreateSurfaceObject(&preintegrationCudaSurface2D_, &resDesc));
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&preintegrationCudaTexture2D_, &resDesc, &texDesc, nullptr));

	computeTexture();
}

renderer::TransferFunctionTexture::~TransferFunctionTexture()
{
    CUMAT_SAFE_CALL_NO_THROW(cudaDestroyTextureObject(textureObject_));
	CUMAT_SAFE_CALL_NO_THROW(cudaFreeArray(textureArray_));

	CUMAT_SAFE_CALL_NO_THROW(cudaDestroyTextureObject(preintegrationCudaTexture1D_));
	CUMAT_SAFE_CALL_NO_THROW(cudaDestroySurfaceObject(preintegrationCudaSurface1D_));
	CUMAT_SAFE_CALL_NO_THROW(cudaFreeArray(preintegrationCudaArray1D_));

	CUMAT_SAFE_CALL_NO_THROW(cudaDestroyTextureObject(preintegrationCudaTexture2D_));
	CUMAT_SAFE_CALL_NO_THROW(cudaDestroySurfaceObject(preintegrationCudaSurface2D_));
	CUMAT_SAFE_CALL_NO_THROW(cudaFreeArray(preintegrationCudaArray2D_));
}

std::string renderer::TransferFunctionTexture::Name()
{
	return "Texture";
}

std::string renderer::TransferFunctionTexture::getName() const
{
	return Name();
}

bool renderer::TransferFunctionTexture::drawUI(UIStorage_t& storage)
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

	//opacity
	//ui
	auto mousePosition = ImGui::GetMousePos();
	bool isLeftClicked = ImGui::IsMouseDown(0);
	bool isLeftReleased = ImGui::IsMouseReleased(0);
	if (isLeftClicked && tfEditorOpacityRect.Contains(mousePosition))
	{
		ImVec2 p;
		p.x = (mousePosition.x - tfEditorOpacityRect.Min.x) / (tfEditorOpacityRect.Max.x - tfEditorOpacityRect.Min.x);
		p.y = 1.0f - (mousePosition.y - tfEditorOpacityRect.Min.y) / (tfEditorOpacityRect.Max.y - tfEditorOpacityRect.Min.y);
		p.y = 1 / 18.0f * (20 * p.y - 1);
		p.x = clamp(p.x, 0.0f, 1.0f);
		p.y = clamp(p.y, 0.0f, 1.0f);
		//Draw line from lastPos_to p
		const auto drawLine = [](std::vector<float>& d, const ImVec2& start, const ImVec2& end)
		{
			int N = d.size();
			int xa = clamp(int(std::round(start.x * (N - 1))), 0, N - 1);
			int xb = clamp(int(std::round(end.x * (N - 1))), 0, N - 1);
			float ya = start.y;
			float yb = end.y;
			if (xa <= xb)
			{
				for (int x = xa; x <= xb; ++x)
				{
					float f = (x - xa) / fmax(1e-10f, float(xb - xa));
					d[x] = lerp(ya, yb, f);
				}
			}
			else
			{
				for (int x = xb; x <= xa; ++x)
				{
					float f = (x - xb) / fmax(1e-10f, float(xa - xb));
					d[x] = lerp(yb, ya, f);
				}
			}
		};
		if (wasClicked_) {
			//std::cout << "Line from " << lastPos_.x << "," << lastPos_.y
			//	<< " to " << p.x << "," << p.y << std::endl;
			drawLine(plot_, lastPos_, p);
			changed = true;
		}
		lastPos_ = p;
		wasClicked_ = true;
	}
	if (isLeftReleased)
	{
		wasClicked_ = false;
	}
	//draw
	window->DrawList->AddRect(tfEditorOpacityRect.Min, tfEditorOpacityRect.Max, ImColor(ImVec4(0.3f, 0.3f, 0.3f, 1.0f)), 0.0f, ImDrawCornerFlags_All, 1.0f);
	int N = plot_.size();
	ImU32 col = ImGui::GetColorU32(ImGuiCol_PlotLines);
	for (int i = 1; i < N; ++i)
	{
		ImVec2 a = ImLerp(tfEditorOpacityRect.Min, tfEditorOpacityRect.Max,
			ImVec2((i - 1) / (N - 1.0f), lerp(0.05f, 0.95f, 1 - plot_[i - 1])));
		ImVec2 b = ImLerp(tfEditorOpacityRect.Min, tfEditorOpacityRect.Max,
			ImVec2(i / (N - 1.0f), lerp(0.05f, 0.95f, 1 - plot_[i])));
		window->DrawList->AddLine(a, b, col, thickness_);
	}

	if (colorEditor_.drawColorUI())
		changed = true;
	
	if (drawUIAbsorptionScaling(storage, absorptionScaling_))
		changed = true;
		
	if (changed) {
		computeTexture();
	}

	static const char* PreintegrationModeNames[] = {
		"None", "1D", "2D"
	};
	if (ImGui::Combo("Preintegration##TFTexture",
		reinterpret_cast<int*>(&preintegrationMode_), 
		PreintegrationModeNames, 3))
	{
		changed = true;
	}

	return changed;
}

	
void renderer::TransferFunctionTexture::load(const nlohmann::json& json, const ILoadingContext* context)
{
	std::vector<renderer::detail::TFPartPiecewiseColor::Point> points =
		json["colorPoints"];
	colorEditor_.updateControlPoints(points);
	plot_ = json["opacityPoints"].get<std::vector<float>>();
	absorptionScaling_ = json["absorptionScaling"];
	preintegrationMode_ = magic_enum::enum_cast<PreintegrationMode>(
		json.value("preintegrationMode", "")).value_or(PreintegrationMode::None);
	
	computeTexture();
}

void renderer::TransferFunctionTexture::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["colorPoints"] = colorEditor_.getPoints();
	json["opacityPoints"] = plot_;
	json["absorptionScaling"] = absorptionScaling_;
	json["preintegrationMode"] = magic_enum::enum_name(preintegrationMode_);
}

void renderer::TransferFunctionTexture::registerPybindModule(pybind11::module& m)
{
	ITransferFunction::registerPybindModule(m);

	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<TransferFunctionTexture, ITransferFunction, std::shared_ptr<TransferFunctionTexture>> c(m, "TransferFunctionTexture");
	py::enum_<TransferFunctionTexture::PreintegrationMode>(c, "PreintegrationMode")
		.value("Off", PreintegrationMode::None)
		.value("Preintegrate1D", PreintegrationMode::Preintegrate1D)
		.value("Preintegrate2D", PreintegrationMode::Preintegrate2D)
		.export_values();

	c.def_readonly("tensor", &TransferFunctionTexture::textureTensor_)
		.def("switch_to_tensor", [this]()
			{
				useTensor_ = true;
				copyTextureToTensor();
			}, py::doc(R"(
By default after loading, the TF uses native 1D textures for interpolation.
With this method, the texture is copied to the pytorch tensor, obtainable
and modifyable by \ref texture, and is used for rendering ever since.
	)"))
	    .def_readwrite("preintegration_mode", &TransferFunctionTexture::preintegrationMode_)
    ;
}

std::optional<int> renderer::TransferFunctionTexture::getBatches(const GlobalSettings& s) const
{
	if (!useTensor_) return {};
	if (!textureTensor_.value.defined())
		throw std::runtime_error("Empty Texture TF!");
	int b = textureTensor_.value.size(0);
	if (b > 1)
		return b;
	else
		return {};
}

std::vector<std::string> renderer::TransferFunctionTexture::getIncludeFileNames(const GlobalSettings& s) const
{
	return { "renderer_tf_texture.cuh" };
}

std::string renderer::TransferFunctionTexture::getDefines(const GlobalSettings& s) const
{
	std::stringstream ss;
	if (useTensor_)
		ss << "#define TRANSFER_FUNCTION_TEXTURE__USE_TENSOR\n";
	ss << "#define TRANSFER_FUNCTION_TEXTURE__PREINTEGRATION_MODE " << int(preintegrationMode_) << "\n";
	return ss.str();
}

std::string renderer::TransferFunctionTexture::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "transferFunctionTextureParameters";
}

std::string renderer::TransferFunctionTexture::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::TransferFunctionTexture";
}

void renderer::TransferFunctionTexture::fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr,
	double stepsize, CUstream stream)
{
	if (useTensor_)
	{
		CHECK_CUDA(textureTensor_.value, true);
		CHECK_DIM(textureTensor_.value, 3);
		CHECK_SIZE(textureTensor_.value, 2, 4);
		CHECK_DTYPE(textureTensor_.value, s.scalarType);
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "VolumeInterpolationGrid", [&]()
			{
				struct Parameters
				{
					::kernel::Tensor3Read<scalar_t> tex;
				} p;
				p.tex = accessor<::kernel::Tensor3Read<scalar_t>>(textureTensor_.value);
				CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
			});
	} else
	{
		if (preintegrationMode_ != None) {
			updatePreintegrationTable(stepsize, stream);
		}
		struct Parameters
		{
			cudaTextureObject_t tex;
			cudaTextureObject_t preintegrated;
		} p;
		p.tex = textureObject_;
		switch (preintegrationMode_)
		{
		case None:
			p.preintegrated = 0; break;
		case Preintegrate1D:
			p.preintegrated = preintegrationCudaTexture1D_; break;
		case Preintegrate2D:
			p.preintegrated = preintegrationCudaTexture2D_; break;
		default:
			throw std::runtime_error("Unknown preintegration mode");
		}
		CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
	}
}

void renderer::TransferFunctionTexture::computeTexture()
{
	static_assert(sizeof(float4) == 4 * sizeof(float),
		"float4 is not equal to 4*float");
	auto colors = colorEditor_.getAsTexture(RESOLUTION, detail::TFPartPiecewiseColor::ColorSpace::RGB);
	textureCpu_.resize(RESOLUTION);
	for (int i=0; i<RESOLUTION; ++i)
	{
		textureCpu_[i].x = colors[i].x;
		textureCpu_[i].y = colors[i].y;
		textureCpu_[i].z = colors[i].z;
		textureCpu_[i].w = absorptionScaling_ * plot_[i];
	}
	CUMAT_SAFE_CALL(cudaMemcpyToArray(textureArray_, 0, 0, textureCpu_.data(), RESOLUTION * 4 * sizeof(float), cudaMemcpyHostToDevice));
	preintegrationValid_ = false;
}

void renderer::TransferFunctionTexture::updatePreintegrationTable(double newStepsize, CUstream stream)
{
	if (preintegrationMode_ == Preintegrate2D && newStepsize != preintegrationLastStepsize_)
		preintegrationValid_ = false;
	if (preintegrationValid_) return;

	//std::cout << "Update Preintegration tables ... " << std::flush;
	detail::Compute1DPreintegrationTable(textureObject_, 
		preintegrationCudaSurface1D_, preintegrationResolution_, stream);
	detail::Compute2DPreintegrationTable(textureObject_,
		preintegrationCudaSurface2D_, preintegrationResolution_,
		static_cast<float>(newStepsize), preintegrationSteps_, stream);
	//std::cout << "Done" << std::endl;
	preintegrationLastStepsize_ = newStepsize;
	preintegrationValid_ = true;
}

void renderer::TransferFunctionTexture::copyTextureToTensor()
{
	//TODO: copy textureArray_ to textureTensor_;
	throw std::runtime_error("Not implemented yet");
}

double4 renderer::TransferFunctionTexture::evaluate(double density) const
{
	const int R = textureCpu_.size();
	const float d = density * R - 0.5f;
	const int di = int(floorf(d));
	const float df = d - di;
	const float4 val0 = textureCpu_[clamp(di, 0, R - 1)];
	const float4 val1 = textureCpu_[clamp(di + 1, 0, R - 1)];
	return make_double4(lerp(val0, val1, df));
}

double renderer::TransferFunctionTexture::getMaxAbsorption() const
{
	if (useTensor_)
		throw std::runtime_error("getMaxAbsorption not implemented yet for tensors");
	return absorptionScaling_;
}

bool renderer::TransferFunctionTexture::canPaste(std::shared_ptr<ITransferFunction> other)
{
	//because every TF implements \ref evaluate,
	//pasting into Texture TF always works
	return true;
}

void renderer::TransferFunctionTexture::doPaste(std::shared_ptr<ITransferFunction> other)
{
	std::vector<detail::TFPartPiecewiseColor::Point> colorPoints;
	double maxW = 0;
	for (int i=0; i<RESOLUTION; ++i)
	{
		double d = (i + 0.5) / RESOLUTION;
		double4 rgba = other->evaluate(d);
		colorPoints.push_back({
			d,
			make_double3(rgba)
			});
		plot_[i] = rgba.w / absorptionScaling_;
		maxW = fmax(maxW, rgba.w);
	}
	if (maxW > absorptionScaling_)
	{
		//update absorption scaling
		double f = absorptionScaling_ / maxW;
		absorptionScaling_ = maxW;
		for (int i = 0; i < RESOLUTION; ++i)
			plot_[i] *= f;
	}
	colorEditor_.updateControlPoints(colorPoints);
	computeTexture();
}

