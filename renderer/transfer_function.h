#pragma once

#include <GL/glew.h>
#include <cuda_runtime.h>

#include "imgui.h"
#include "imgui_internal.h"
#include "imodule.h"
#include "parameter.h"

BEGIN_RENDERER_NAMESPACE

/**
 * Defines the transfer function.
 * The transfer function is called in the following way:
 * <pre>
 *  real4_t color = tf.eval(real_t density, real3 normal, real_t previousDensity, real_t stepsize, int batch)
 * </pre>
 * If the \c normal vector is required or can be left undefined is
 * given by \ref requiresNormal().
 * The TF is also responsible for scaling the absorption with the given \c stepsize.
 * This way, together with \c previousDensity, allows for preintegrated TFs,
 * that approximate the whole interval at once.
 */
class ITransferFunction : public IKernelModule, public std::enable_shared_from_this<ITransferFunction>
{

	static const std::string UI_KEY_ABSORPTION_SCALING;
	static const std::string UI_KEY_COPIED_TF;

public:
	static constexpr std::string_view TAG = "tf";
	std::string getTag() const override { return std::string(TAG); }
	static std::string Tag() { return std::string(TAG); }

	bool drawUILoadSaveCopyPaste(UIStorage_t& storage);
	bool drawUIAbsorptionScaling(UIStorage_t& storage, double& scaling);
protected:
	void registerPybindModule(pybind11::module& m) override;
public:

	virtual bool canPaste(std::shared_ptr<ITransferFunction> other) { return false; }
	virtual void doPaste(std::shared_ptr<ITransferFunction> other) {}
	/**
	 * UI only: Evaluates this TF at the given density in [0,1].
	 * This is used for copy-paste
	 */
	virtual double4 evaluate(double density) const = 0;

	/**
	 * Returns the maximal possible absorption per ray differential.
	 * This is used for delta tracking.
	 */
	virtual double getMaxAbsorption() const = 0;

	/**
	 * Checks, if the TF requires gradients for evaluation.
	 * If this returns true and and no gradients are passed
	 * to \ref evaluate, the latter method will throw an error.
	 */
	virtual bool requiresGradients() const;

	/**
	 * Evaluates the transfer function for the specified density (1D).
	 * First, the input densities are mapped from [densityMin, densityMax] to [0,1],
	 * then sent to the TF. For densities < densityMin, the color is (0,0,0,0),
	 * densities>densityMax are clamped.
	 *
	 * For preintegration, \c previousDensity and \c stepsize have to be passed.
	 *
	 * \param density the density values of shape (B,1)
	 * \param densityMin minimal density
	 * \param densityMax the maximal density
	 * \param previousDensity the previous density for pre-integration
	 * \param stepsize the current stepsize
	 * \param gradient the density gradient
	 * \return the resulting color values of shape (B,4)
	 */
	virtual torch::Tensor evaluate(const torch::Tensor& density,
		double densityMin, double densityMax,
		const std::optional<torch::Tensor>& previousDensity,
		const std::optional<double>& stepsize,
		const std::optional<torch::Tensor>& gradient,
		CUstream stream);

	void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;
protected:
	virtual void fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr, double stepsize, CUstream stream) = 0;

	virtual void drawUIHistogram(UIStorage_t& storage, const ImRect& histogramRect);
};
typedef std::shared_ptr<ITransferFunction> ITransferFunction_ptr;

//build-in TFs

/**
 * Simple identity transfer function.
 * Just contains a scaling from density to color+absorption
 */
class TransferFunctionIdentity : public ITransferFunction
{
private:
	Parameter<double2> scaleAbsorptionEmission_;

public:

	TransferFunctionIdentity();
	~TransferFunctionIdentity() override;
	
	std::string getName() const override;
	bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* context) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
public:
	std::optional<int> getBatches(const GlobalSettings& s) const override;
	std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	std::string getDefines(const GlobalSettings& s) const override;
	std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	std::string getPerThreadType(const GlobalSettings& s) const override;
protected:
	void fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr, double stepsize, CUstream stream) override;
public:
	double4 evaluate(double density) const override;
	double getMaxAbsorption() const override;
};

namespace detail
{
	class TFPartPiecewiseColor : NonAssignable
	{
	public:
		struct Point
		{
			double position;
			double3 colorRGB;
		};
		static const std::string UI_KEY_SHOW_COLOR_CONTROL_POINTS;

		TFPartPiecewiseColor();
		~TFPartPiecewiseColor();
		bool drawUI(const ImRect& rect, bool showControlPoints);
		bool drawColorUI(); //displays the UI to change the colors
		void updateControlPoints(const std::vector<Point>& points);
		const std::vector<Point>& getPoints() const { return pointsSorted_; }

		enum class ColorSpace
		{
			RGB
		};
		std::vector<double3> getAsTexture(int resolution, ColorSpace colorSpace) const;

	private:
		const float cpWidth_{ 8.0f };
		ImVec4 pickedColor_{ 0.0f, 0.0f, 1.0f, 1.0f };
		int selectedControlPointForMove_{ -1 };
		int selectedControlPointForColor_{ -1 };
		std::deque<Point> pointsUI_;
		std::vector<Point> pointsSorted_;

		static constexpr int ColorMapWidth = 256;
		GLuint colorMapImage_{ 0 };

		bool handleIO(const ImRect& rect);
		
		ImRect createControlPointRect(float x, const ImRect& rect);
		float screenToEditor(float screenPositionX, const ImRect& rect);
		float editorToScreen(float editorPositionX, const ImRect& rect);
		void sortPointsAndUpdateTexture();
	};

	class TfPartPiecewiseOpacity
	{
	public:
		struct Point
		{
			double position;
			double absorption;
		};

		TfPartPiecewiseOpacity();
		bool drawUI(const ImRect& rect);
		void updateControlPoints(const std::vector<Point>& points);
		const std::vector<Point>& getPoints() const { return pointsSorted_; }

	private:
		const float circleRadius_{ 4.0f };

		int selectedControlPoint_{ -1 };
		std::deque<Point> pointsUI_;
		std::vector<Point> pointsSorted_;

	private:
		bool handleIO(const ImRect& rect);
		void render(const ImRect& rect);
		ImRect createControlPointRect(const ImVec2& controlPoint);
		Point screenToEditor(const ImVec2& screenPosition, const ImRect& rect);
		ImVec2 editorToScreen(const Point& editorPosition, const ImRect& rect);
		void sortPoints();
	};
}

/**
 * texture-based transfer function.
 * Just contains a scaling from density to color+absorption
 */
class TransferFunctionTexture : public ITransferFunction
{
public:
	enum PreintegrationMode
	{
	    None, //no preintegration
		Preintegrate1D, //1D-preintegration table, stepsize-independent
		Preintegrate2D  //2D-preintegration table, stepsize dependent but more precise
	};

private:
	static constexpr int RESOLUTION = 256;
	bool useTensor_;
	Parameter<torch::Tensor> textureTensor_; //BxRx4
	cudaArray_t textureArray_ {0};
	cudaTextureObject_t textureObject_ {0};
	std::vector<float4> textureCpu_;

	//UI
	detail::TFPartPiecewiseColor colorEditor_;
	const float thickness_ = 2.0f;
	bool resized_ = false;
	std::vector<float> plot_;
	double absorptionScaling_;

	PreintegrationMode preintegrationMode_;
	int preintegrationResolution_ = 256; //table resolution
	int preintegrationSteps_ = 256; //integration steps
	bool preintegrationValid_ = false;
	double preintegrationLastStepsize_ = 0;

	cudaArray_t preintegrationCudaArray1D_{ 0 };
	cudaTextureObject_t preintegrationCudaTexture1D_{ 0 };
	cudaSurfaceObject_t preintegrationCudaSurface1D_{ 0 };
	cudaArray_t preintegrationCudaArray2D_{ 0 };
	cudaTextureObject_t preintegrationCudaTexture2D_{ 0 };
	cudaSurfaceObject_t preintegrationCudaSurface2D_{ 0 };

	bool wasClicked_ = false;
	ImVec2 lastPos_;

public:
	TransferFunctionTexture();
	~TransferFunctionTexture();

	static std::string Name();
	std::string getName() const override;
	bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* context) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
public:
	std::optional<int> getBatches(const GlobalSettings& s) const override;
	std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	std::string getDefines(const GlobalSettings& s) const override;
	std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	std::string getPerThreadType(const GlobalSettings& s) const override;
protected:
	void fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr, double stepsize, CUstream stream) override;

private:
	void computeTexture();
	void updatePreintegrationTable(double newStepsize, CUstream stream);
	void copyTextureToTensor();

public:
	double4 evaluate(double density) const override;
	double getMaxAbsorption() const override;

	bool canPaste(std::shared_ptr<ITransferFunction> other) override;
	void doPaste(std::shared_ptr<ITransferFunction> other) override;
};
namespace detail
{
	void Compute1DPreintegrationTable(cudaTextureObject_t srcTFTexture, cudaSurfaceObject_t dstSurface,
		int dstResolution, CUstream stream);
	void Compute2DPreintegrationTable(cudaTextureObject_t srcTFTexture, cudaSurfaceObject_t dstSurface,
		int dstResolution, float stepsize, int quadratureSteps, CUstream stream);
}

/**
 * piecewise linear transfer function.
 */
class TransferFunctionPiecewiseLinear : public ITransferFunction
{
private:
	//BxRx5
	torch::Tensor textureTensorCpu_;
	Parameter<torch::Tensor> textureTensor_;

	//UI
	detail::TFPartPiecewiseColor colorEditor_;
	detail::TfPartPiecewiseOpacity opacityEditor_;
	double absorptionScaling_;

public:
	TransferFunctionPiecewiseLinear();
	~TransferFunctionPiecewiseLinear() override;

	[[nodiscard]] std::string getName() const override;
	bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* context) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
public:
	[[nodiscard]] std::optional<int> getBatches(const GlobalSettings& s) const override;
	[[nodiscard]] std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getDefines(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getPerThreadType(const GlobalSettings& s) const override;
protected:
	void fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr, double stepsize, CUstream stream) override;

private:
	void computeTensor();

public:
	double4 evaluate(double density) const override;
	double getMaxAbsorption() const override;
};


/**
 * piecewise linear transfer function.
 */
class TransferFunctionGaussian : public ITransferFunction
{
public:
	struct Point
	{
		ImVec4 color;
		float opacity;
		float mean;
		float variance;
	};

private:
	static constexpr float circleRadius_ = 4.0f;
	static constexpr float alpha1_ = 0.4f;
	static constexpr float alpha2_ = 0.4f;
	static constexpr float thickness1_ = 3;
	static constexpr float thickness2_ = 2;
	
	std::vector<Point> points_;
	int selectedPoint_ = 0;
	int draggedPoint_ = 0;
	double absorptionScaling_;
	bool usePiecewiseAnalyticIntegration_ = false;

	//BxRx6
	Parameter<torch::Tensor> textureTensor_;
	bool scaleWithGradient_;

public:
	TransferFunctionGaussian();
	~TransferFunctionGaussian() override;
	const std::vector<Point>& getPoints() const { return points_; }

	[[nodiscard]] std::string getName() const override;
	[[nodiscard]] bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
public:

	void prepareRendering(GlobalSettings& s) const override;
	[[nodiscard]] std::optional<int> getBatches(const GlobalSettings& s) const override;
	[[nodiscard]] std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getDefines(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getPerThreadType(const GlobalSettings& s) const override;
protected:
	void fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr, double stepsize, CUstream stream) override;
private:
	bool renderAndIO(const ImRect& rect);
	static float gaussian(float x, float mean, float variance);
	void computeTensor();

	ImVec2 screenToEditor(const ImVec2& screenPosition, const ImRect& rect);
	ImVec2 editorToScreen(const ImVec2& editorPosition, const ImRect& rect);

public:
	double4 evaluate(double density) const override;
	double getMaxAbsorption() const override;
};


END_RENDERER_NAMESPACE

namespace nlohmann {
	template <>
	struct adl_serializer<renderer::detail::TFPartPiecewiseColor::Point> {
		static void to_json(json& j, const renderer::detail::TFPartPiecewiseColor::Point& v) {
			j = json::array({ v.position, v.colorRGB.x, v.colorRGB.y, v.colorRGB.z });
		}

		static void from_json(const json& j, renderer::detail::TFPartPiecewiseColor::Point& v) {
			if (j.is_array() && j.size() == 4)
			{
				v.position = j.at(0).get<double>();
				v.colorRGB.x = j.at(1).get<double>();
				v.colorRGB.y = j.at(2).get<double>();
				v.colorRGB.z = j.at(3).get<double>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a TFPartPiecewiseColor::Point" << std::endl;
		}
	};
	template <>
	struct adl_serializer<renderer::detail::TfPartPiecewiseOpacity::Point> {
		static void to_json(json& j, const renderer::detail::TfPartPiecewiseOpacity::Point& v) {
			j = json::array({ v.position, v.absorption });
		}

		static void from_json(const json& j, renderer::detail::TfPartPiecewiseOpacity::Point& v) {
			if (j.is_array() && j.size() == 2)
			{
				v.position = j.at(0).get<double>();
				v.absorption = j.at(1).get<double>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a TfPartPiecewiseOpacity::Point" << std::endl;
		}
	};
	template <>
	struct adl_serializer<renderer::TransferFunctionGaussian::Point> {
		static void to_json(json& j, const renderer::TransferFunctionGaussian::Point& v) {
			j = json::array({ v.color.x, v.color.y, v.color.z, v.opacity, v.mean, v.variance });
		}

		static void from_json(const json& j, renderer::TransferFunctionGaussian::Point& v) {
			if (j.is_array() && j.size() == 6)
			{
				v.color.x = j.at(0).get<float>();
				v.color.y = j.at(1).get<float>();
				v.color.z = j.at(2).get<float>();
				v.opacity = j.at(3).get<float>();
				v.mean = j.at(4).get<float>();
				v.variance = j.at(5).get<double>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a TransferFunctionGaussian::Point" << std::endl;
		}
	};
}
