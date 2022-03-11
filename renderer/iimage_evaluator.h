#pragma once

#include <GL/glew.h>
#include <torch/types.h>
#include <cuda.h>

#include "imodule.h"
#include "kernel_loader.h"
#include "renderer_tensor.cuh"

BEGIN_RENDERER_NAMESPACE

/**
 * The main entry point
 */
class IImageEvaluator : public IModuleContainer, public std::enable_shared_from_this<IImageEvaluator>
{
public:
	enum class ChannelMode
	{
		ChannelMask,
		ChannelNormal,
		ChannelDepth,
		ChannelColor,
		_ChannelCount_
	};
	static const char* ChannelModeNames[int(ChannelMode::_ChannelCount_)];

protected:
	//ui
	static const std::string UI_KEY_SELECTED_CAMERA;
	static const std::string UI_KEY_SELECTED_OUTPUT_CHANNEL;
	ChannelMode selectedChannel_;
	static const std::string UI_KEY_USE_DOUBLE_PRECISION;
	bool isDoublePrecision_;

	IImageEvaluator();

public:
	static constexpr int NUM_CHANNELS = 8;

	/**
	 * Returns the iff this image evaluator iteratively refines the image.
	 * In that case, \render is called with \c refine=true
	 * if there was no UI interaction to allow for an iterative refinement.
	 */
	[[nodiscard]] virtual bool isIterativeRefining() const { return false; }

	/**
	 * Renders the image with the given screen size.
	 * This has to be overwritten by the implementations.
	 *
	 * Typical behaviour:
	 * 1. Render the opaque rasterized objects
	 * 2. Raytrace the object while respecting the depth buffer from rasterization
	 *
	 * \param width the width of the screen
	 * \param height the height of the screen
	 * \param refine true if there was no UI interaction, but \ref isInterativeRefining()
	 *   is true. In that case, \c out contains the previous result.
	 *   False if a change was triggered and the image should be created from scratch
	 * \param out output tensor to reuse
	 * \return a BCHW tensor with the channels being
	 *  0,1,2: rgb
	 *  3: alpha
	 *  4,5,6: normal
	 *  7: depth
	 */
	virtual torch::Tensor render(
		int width, int height, CUstream stream,
		bool refine, const torch::Tensor& out = {}) = 0;

	using tensor_or_texture_t = std::variant<torch::Tensor, GLubyte*>;
	/**
	 * Extracts the channel from the output from \ref render().
	 * What channel is extracted, is determined by \c channel.
	 * For channel 'Color', tonemapping is additionally used.
	 *
	 * The output can either be a RGBA8-texture or a torch tensor of shape (B, 4, H, W).
	 */
	static void ExtractColor(
		const torch::Tensor& inputTensor,
		tensor_or_texture_t output,
		bool useTonemapping, float maxExposure,
		ChannelMode channel, CUstream stream);

	virtual void extractColor(
		const torch::Tensor& inputTensor,
		tensor_or_texture_t output,
		ChannelMode channel, CUstream stream)
	{
		ExtractColor(inputTensor, output, false, 1.0f, channel, stream);
	}

	[[nodiscard]] virtual ChannelMode selectedChannel() const
	{
		return selectedChannel_;
	}

	virtual void setSelectedChannel(const ChannelMode selectedChannel)
	{
		selectedChannel_ = selectedChannel;
	}

	[[nodiscard]] virtual bool isDoublePrecision() const
	{
		return isDoublePrecision_;
	}

	virtual void setIsDoublePrecision(const bool isDoublePrecision)
	{
		isDoublePrecision_ = isDoublePrecision;
	}

	/**
	 * The batch count is specified by the contained submodules.
	 * This functions computes the batch count and verifies that they
	 * are compatible. This means, if one module has a batch size > 1,
	 * all modules with a batch size > 1 must agree to that batch size.
	 *
	 * This calls \ref IKernelModule::getBatches()
	 */
	int computeBatchCount();

	/**
	 * Returns the default stream that is
	 * shared with PyTorch.
	 */
	static CUstream getDefaultStream();

	/**
	 * Creates the global settings for the kernel assembly.
	 * In detail, sets the scalar type and the root module to this.
	 * \param useDoublePrecision true iff double precision should be used.
	 *  If empty, the setting from the UI is used.
	 */
	IKernelModule::GlobalSettings getGlobalSettings(
		const std::optional<bool>& useDoublePrecision = {});

	/**
	 * Calls \ref IKernelModule::prepareRendering of the kernel modules.
	 * This also allows them to modify the settings.
	 */
	void modulesPrepareRendering(IKernelModule::GlobalSettings& s);

	/**
	 * Queries the submodules and compiles the main rendering kernel.
	 * This calls \ref IKernelModule::getDefines(),
	 * \ref IKernelModule::getIncludeFileNames,
	 * \ref IKernelModule::getConstantDeclarationName()
	 */
	KernelLoader::KernelFunction getKernel(
		const IKernelModule::GlobalSettings& s,
		const std::string& kernelName,
		const std::string& extraSource);

	/**
	 * Queries the submodules and fills the constants
	 */
	void fillConstants(
		KernelLoader::KernelFunction& f, const IKernelModule::GlobalSettings& s,
		CUstream stream);

protected:
	/**
	 * Draws UI for the global settings (i.e. the scalar type).
	 * This should be on top of the UI.
	 */
	bool drawUIGlobalSettings(UIStorage_t& s);

	/**
	 * Draws the UI for the output channel.
	 * This should be at the bottom of the UI
	 */
	bool drawUIOutputChannel(UIStorage_t& s, bool readOnly = false);

public:
	static constexpr std::string_view TAG = "ImageEvaluator";
	static std::string Tag() { return std::string(TAG); }
	std::string getTag() const override { return Tag(); }

	void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
};
typedef std::shared_ptr<IImageEvaluator> IImageEvaluator_ptr;

END_RENDERER_NAMESPACE

namespace kernel
{
	void CopyOutputToTexture(
		int width, int height,
		const Tensor4Read<float>& input,
		GLubyte* output,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream);
	void CopyOutputToTexture(
		int width, int height,
		const Tensor4Read<double>& input,
		GLubyte* output,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream);
	void CopyOutputToTexture(
		int width, int height, int batches,
		const Tensor4Read<float>& input,
		Tensor4RW<float>& output,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream);
	void CopyOutputToTexture(
		int width, int height, int batches,
		const Tensor4Read<double>& input,
		Tensor4RW<double>& output,
		int r, int g, int b, int a,
		float scaleRGB, float offsetRGB, float scaleA, float offsetA,
		CUstream stream);

	void Tonemapping(
		int width, int height,
		const Tensor4Read<float>& input,
		GLubyte* texture,
		float maxExposure,
		CUstream stream);
	void Tonemapping(
		int width, int height,
		const Tensor4Read<double>& input,
		GLubyte* texture,
		float maxExposure,
		CUstream stream);
	void Tonemapping(
		int width, int height, int batches,
		const Tensor4Read<float>& input,
		Tensor4RW<float>& output,
		float maxExposure,
		CUstream stream);
	void Tonemapping(
		int width, int height, int batches,
		const Tensor4Read<double>& input,
		Tensor4RW<double>& output,
		float maxExposure,
		CUstream stream);
}
