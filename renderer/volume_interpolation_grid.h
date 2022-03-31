#pragma once

#include "volume_interpolation.h"

#include "parameter.h"
#include "volume.h"
#include "background_worker.h"
#include "lru_cache.h"
#include "renderer_histogram.cuh"
#include <torch/types.h>
#include <filesystem>


BEGIN_RENDERER_NAMESPACE
    /**
 * \brief volume interpolation using a regular grid of voxels.
 * The input can either be a 3D texture or a torch::Tensor.
 * Interpolation modes: nearest neighbor, trilinear, tricubic.
 */
class VolumeInterpolationGrid : public IVolumeInterpolation
{
public:
	enum class VolumeSource
	{
		VOLUME,
		TORCH_TENSOR,
		EMPTY
	};
	enum class VolumeInterpolation
	{
		NEAREST_NEIGHBOR,
		TRILINEAR,
		TRICUBIC,
		_COUNT_
	};
	static const std::string VolumeInterpolationNames[int(VolumeInterpolation::_COUNT_)];

	//Key for the UIStorage containing the histogram (Volume::Histogram_ptr)
	static const std::string UI_KEY_HISTOGRAM;
	//Key for the UIstorage containing the min density (float)
	static const std::string UI_KEY_MIN_DENSITY;
	//Key for the UIstorage containing the max density (float)
	static const std::string UI_KEY_MAX_DENSITY;

	/**
	 * For vector-valued features of the volume, this
	 * enum specifies how to map the channels to density.
	 */
	enum class Feature2Density
	{
		//Do nothing (for scalar features)
	    Identity,
		VelocityX,
		VelocityY,
		VelocityZ,
		VelocityMagnitude,
		Density,
		DensityCurvature,
		__Count__
	};
private:
	static const std::string Feature2DensityNames[int(Feature2Density::__Count__)];
	static const std::string Feature2DensityTextureType[int(Feature2Density::__Count__)];
	static const std::string Feature2DensityTextureChannel[int(Feature2Density::__Count__)];

public:
	/**
	 * Feature descriptor for density outputs.
	 */
    struct FeatureDescriptorDensity
    {
		//the index of the feature in the volume to use
		int featureIndex;
		//how to map the contents to the scalar density
		Feature2Density mapping;
    };

	struct FeatureDescriptorVelocity
	{
		//the index of the feature in the volume to use
		int featureIndex;
	};

	struct FeatureDescriptorColor
	{
		//the index of the feature in the volume to use
		int featureIndex;
	};

	/**
	 * Key for the histogram cache
	 */
	struct HistogramKey
	{
		Volume* volumePtr;
		int featureIndex;
		int mipmapLevel;
		Feature2Density mapping;

        friend std::size_t hash_value(const HistogramKey& obj) noexcept
        {
            std::size_t seed = 0x7FAC4D7E;
            seed ^= (seed << 6) + (seed >> 2) + 0x3036D8A5 + reinterpret_cast<std::size_t>(obj.volumePtr);
            seed ^= (seed << 6) + (seed >> 2) + 0x39C1F81A + static_cast<std::size_t>(obj.featureIndex);
            seed ^= (seed << 6) + (seed >> 2) + 0x579D9BDC + static_cast<std::size_t>(obj.mipmapLevel);
            seed ^= (seed << 6) + (seed >> 2) + 0x6C276826 + static_cast<std::size_t>(obj.mapping);
            return seed;
        }

        friend bool operator==(const HistogramKey& lhs, const HistogramKey& rhs)
        {
            return lhs.volumePtr == rhs.volumePtr
                && lhs.featureIndex == rhs.featureIndex
                && lhs.mipmapLevel == rhs.mipmapLevel
                && lhs.mapping == rhs.mapping;
        }

        friend bool operator!=(const HistogramKey& lhs, const HistogramKey& rhs)
        {
            return !(lhs == rhs);
        }
    };
	/**
	 * The actual histogram
	 */
	typedef kernel::VolumeHistogram HistogramValue;
	typedef kernel::VolumeHistogram_ptr HistogramValue_ptr;

private:
	VolumeSource source_;
	VolumeInterpolation interpolation_;

	float minDensity_;
	float maxDensity_;

	//source = Texture
	Volume_ptr volume_;
	int mipmapLevel_;
	std::vector<FeatureDescriptorDensity> availableDensityFeatures_;
	int selectedDensityFeatureIndex_;
	std::vector<FeatureDescriptorVelocity> availableVelocityFeatures_;
	int selectedVelocityFeatureIndex_;
	std::vector<FeatureDescriptorColor> availableColorFeatures_;
	int selectedColorFeatureIndex_;

	//source = tensor
	Parameter<torch::Tensor> tensor_;

	//UI (source=Texture)
	std::filesystem::path volumeFullFilename_;
	BackgroundWorker worker_;
	std::function<void()> backgroundGui_;
	bool newVolumeLoaded_ = false;
	//ensemble
	VolumeEnsembleFactory_ptr ensembleFactory_;
	int currentEnsemble_;
	int currentTimestep_;

	//histogram
	static constexpr auto histogramHash = [](const HistogramKey& k) {return hash_value(k); };
	LRUCache<HistogramKey, HistogramValue_ptr, decltype(histogramHash)> histogramCache_;
	cudaStream_t histogramStream_;
	cudaEvent_t histogramCompletionEvent_;
	HistogramKey histogramCurrentKey_;
	bool histogramExtractionRunning_;
	kernel::VolumeHistogram* histogramDevice_;
	kernel::VolumeHistogram histogramHost_;

	/*
	 * Switches between old and now behavior for the conversion from normalized world coords to object coords.
	 * Old: multiply position with (resolution-1), introduces slight offset when using mipmaps
	 * New: multiply position with (resolution), no directed offset, global shrinking
	 */
	bool gridResolutionNewBehavior_;

public:
	VolumeInterpolationGrid();
	~VolumeInterpolationGrid() override;

	/**
	 * Sets the source to the volume and mipmap level
	 */
	void setSource(Volume_ptr v, int mipmap);
	/**
	 * Sets the source to the given tensor of shape
	 * (Batch, X, Y, Z)
	 */
	void setSource(const torch::Tensor& t);
	/**
	 * Sets the source to the specified ensemble factory
	 */
	void setSource(VolumeEnsembleFactory_ptr factory);
	/**
	 * Sets the source to the volume provided by the ensemble factory.
	 */
	void setEnsembleAndTime(int ensemble, int time, int mipmap=0);

	[[nodiscard]] VolumeSource source() const
	{
		return source_;
	}

	[[nodiscard]] VolumeInterpolation interpolation() const
	{
		return interpolation_;
	}

	void setInterpolation(const VolumeInterpolation interpolation)
	{
		interpolation_ = interpolation;
	}

	[[nodiscard]] float minDensity() const
	{
		return minDensity_;
	}

	[[nodiscard]] float maxDensity() const
	{
		return maxDensity_;
	}

	[[nodiscard]] Volume_ptr volume() const;

	[[nodiscard]] int mipmapLevel() const;

	/**
	 * Requests the histogram for the current volume, feature, mipmap level
	 * and feature mapping.
	 * If the histogram is not yet computed, a nullptr is returned
	 */
	[[nodiscard]] HistogramValue_ptr requestHistogram();

	[[nodiscard]] Parameter<torch::Tensor> tensor() const;

	[[nodiscard]] VolumeEnsembleFactory_ptr ensembleFactory() const;
	[[nodiscard]] int currentEnsemble() const;
	[[nodiscard]] int currentTimestep() const;

	/*
	 * Switches between old and now behavior for the conversion from normalized world coords to object coords.
	 * Old: multiply position with (resolution-1), introduces slight offset when using mipmaps
	 * New: multiply position with (resolution), no directed offset, global shrinking
	 *
	 * Default:
	 * - UI -> new behavior (and fixed to new behavior)
	 * - Python -> old behavior (value is false)
	 */
	void setGridResolutionNewBehavior(bool enableNewBehavior) { gridResolutionNewBehavior_ = enableNewBehavior; }
	[[nodiscard]] bool isGridResolutionNewBehavior() const { return gridResolutionNewBehavior_; }

	std::string getName() const override;
	bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* context) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
public:
	[[nodiscard]] bool supportsCurvatureEstimation() const override { return true; }
	GlobalSettings::VolumeOutput outputType() const override;
	std::optional<int> getBatches(const GlobalSettings& s) const override;
	std::string getDefines(const GlobalSettings& s) const override;
	std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	std::string getPerThreadType(const GlobalSettings& s) const override;
	void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;

private:
	//UI
	void loadVolumeDialog();
	void loadVolumeFromPath(const std::string& filename);
	void loadEnsemble(const std::string& filename, float* progress);
	void loadVolume(const std::string& filename, float* progress);
	Volume_ptr loadVolumeImpl(const std::string& filename, float* progress);
	bool extractDensityFeaturesFromVolume();
};

END_RENDERER_NAMESPACE
