#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <functional>
#include <cassert>
#include <vector>
#include <optional>
#include <cfloat>
#include <torch/extension.h>
#include <torch/types.h>
#include <pybind11/pybind11.h>

#include "commons.h"
#include "lru_cache.h"

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4251) // dll export of STL types
#endif

class LZ4Compressor;
class LZ4Decompressor;

BEGIN_RENDERER_NAMESPACE
typedef std::function<void(const std::string&)> VolumeLoggingCallback_t;
typedef std::function<void(float)> VolumeProgressCallback_t;
typedef std::function<void(const std::string&, int)> VolumeErrorCallback_t;

/**
 * \brief The main storage class for volumetric datasets.
 *
 * The volume stores multiple feature channels,
 * where each feature describes, e.g., density, velocity, color.
 * Each feature is specified by the number of channel and the data type
 * (unsigned char, unsigned short, float)
 * The feature channels can have different resolutions, but are all
 * mapped to the same bounding volume.
 *
 * Format of the .cvol file (binary file):
 * <pre>
 *  [64 Bytes Header]
 *   4 bytes: magic number "CVOL"
 *   4 bytes: version (int)
 *   3*4 bytes: worldX, worldY, worldZ of type 'float'
 *     the size of the bounding box in world space
 *   4 bytes: number of features (int)
 *   4 bytes: flags (int), OR-combination of \ref Volume::Flags
 *	 4 bytes: unused
 *  [Content, repeated for the number of features]
 *   4 bytes: length of the name (string)
 *	 n bytes: the contents of the name string (std::string)
 *	 3*8 bytes: sizeX, sizeY, sizeZ of type 'uint64_t'
 *	   the voxel resolution of this feature
 *	 4 bytes: number of channels (int)
 *	 4 bytes: datatype (\ref DataType)
 *   Ray memory dump of the volume, sizeX*sizeY*sizeZ*channels entries of type 'datatype'.
 *     channels is fastest, followed by X and Y, Z is slowest.
 * </pre>
 *
 * Legacy format, before multi-channel support was added:
 * Format of the .cvol file (binary file):
 * <pre>
 *  [64 Bytes Header]
 *   4 bytes: magic number "cvol"
 *   3*8 bytes: sizeX, sizeY, sizeZ of type 'uint64_t'
 *   3*8 bytes: voxel size X, Y, Z in world space of type' double'
 *   4 bytes: datatype, uint value of the enum \ref Volume::DataType
 *	 1 byte: bool if the volume contents are LZ4-compressed
 *   7 bytes: padding, unused
 *  [Content]
 *   
 *   Ray memory dump of the volume, sizeX*sizeY*sizeZ entries of type 'datatype'.
 *   X is fastest, Z is slowest.
 * </pre>
 */
class MY_API Volume
{
public:
	template<int numOfBins>
	struct VolumeHistogram
	{
		float bins[numOfBins]{ 0.0f };
		float minDensity{ FLT_MAX };
		float maxDensity{ -FLT_MAX };
		float maxFractionVal{ 1.0f };
		unsigned int numOfNonzeroVoxels{ 0 };

		constexpr int getNumOfBins() const { return numOfBins; }
	};
	using Histogram = VolumeHistogram<512>;
	using Histogram_ptr = std::shared_ptr<Histogram>;

	enum DataType
	{
		TypeUChar,
		TypeUShort,
		TypeFloat,
		_TypeCount_
	};
	static const int BytesPerType[_TypeCount_];

	class Feature;

	class MY_API MipmapLevel
	{
	private:
		uint64_t channels_;
	    uint64_t sizeX_, sizeY_, sizeZ_;
		char* dataCpu_;
		cudaArray_t dataGpu_;
		cudaTextureObject_t dataTexLinear_;
		cudaTextureObject_t dataTexNearest_;
		cudaSurfaceObject_t dataSurface_;
		int cpuDataCounter_;
		int gpuDataCounter_;

		Feature* parent_;
		friend class Feature;

	public:
		MipmapLevel(Feature* parent, uint64_t sizeX, uint64_t sizeY, uint64_t sizeZ);
		~MipmapLevel();

		MipmapLevel(const MipmapLevel& other) = delete;
		MipmapLevel(MipmapLevel&& other) noexcept = delete;
		MipmapLevel& operator=(const MipmapLevel& other) = delete;
		MipmapLevel& operator=(MipmapLevel&& other) noexcept = delete;

		[[nodiscard]] int64_t channels() const { return channels_; }
		[[nodiscard]] int64_t sizeX() const { return sizeX_; }
		[[nodiscard]] int64_t sizeY() const { return sizeY_; }
		[[nodiscard]] int64_t sizeZ() const { return sizeZ_; }
		[[nodiscard]] int3 size() const { return make_int3(sizeX_, sizeY_, sizeZ_); }
		[[nodiscard]] DataType type() const { return parent_->type(); }

		[[nodiscard]] size_t idx(int x, int y, int z, int channel = 0) const
		{
			assert(x >= 0 && x < sizeX_);
			assert(y >= 0 && y < sizeY_);
			assert(z >= 0 && z < sizeZ_);
			return channel + channels_ * (x + sizeX_ * (y + sizeY_ * z));
		}

		template<typename T>
        [[nodiscard]] const T* dataCpu() const { return reinterpret_cast<T*>(dataCpu_); }
		template<typename T>
		[[nodiscard]] T* dataCpu() { cpuDataCounter_++; return reinterpret_cast<T*>(dataCpu_); }
		[[nodiscard]] cudaArray_const_t dataGpu() const;
        /**
		 * Returns the data as a 3d texture with un-normalized coordinates
		 * and linear interpolation.
		 */
		[[nodiscard]] cudaTextureObject_t dataTexGpuLinear() const;
        [[nodiscard]] cudaTextureObject_t dataTexGpuNearest() const;
		[[nodiscard]] cudaSurfaceObject_t gpuSurface() const;

        /**
		 * Copies the CPU data to the GPU.
		 * It is then available as CUDA array and texture object.
		 */
		void copyCpuToGpu();

		/**
	     * \brief clears all GPU resources of this mipmap level.
	     * You must call \ref copyCpuToGpu() again before rendering.
	     */
		void clearGpuResources();

		/**
		 * Returns a copy oft his mipmap's data as a CPU float tensor.
		 * Shape: C*X*Y*Z
		 */
		[[nodicard]] torch::Tensor toTensor() const;
		/**
		 * Sets this mipmap's data from the given float tensor.
		 * The shape of C*X*Y*Z must match.
		 */
		void fromTensor(const torch::Tensor& t);

	private:
		bool checkHasGpu() const;
	};
	typedef std::shared_ptr<MipmapLevel> MipmapLevel_ptr;

	enum class MipmapFilterMode
	{
		/**
		 * Average filtering
		 */
		AVERAGE,
		/**
		 * A random sample is taken
		 */
		 HALTON
	};

	class Feature
	{
		const std::string name_;
		const DataType type_;
		const int numChannels_;
		std::vector<MipmapLevel_ptr> levels_;

		const Volume* parent_;
		friend class Volume;

	public:
		Feature(Volume* parent, const std::string& name, DataType type, int numChannels,
			uint64_t sizeX, uint64_t sizeY, uint64_t sizeZ);

	private:
		//load the feature from the stream
		static std::shared_ptr<Feature> load(
			Volume* parent,
			std::ifstream& s, LZ4Decompressor* compressor,
			const VolumeProgressCallback_t& progress,
			const VolumeLoggingCallback_t& logging,
			const VolumeErrorCallback_t& error);
		//saves the feature to the stream
		void save(std::ofstream& s, LZ4Compressor* compressor,
			const VolumeProgressCallback_t& progress,
			const VolumeLoggingCallback_t& logging,
			const VolumeErrorCallback_t& error);

	public:
        [[nodiscard]] const std::string& name() const
        {
            return name_;
        }

        [[nodiscard]] DataType type() const
        {
            return type_;
        }

        [[nodiscard]] int numChannels() const
        {
            return numChannels_;
        }

        /**
		 * \brief Returns the voxel resolution of this feature channel.
		 */
		int3 baseResolution() const {
			return make_int3(
				levels_[0]->sizeX(), levels_[0]->sizeY(), levels_[0]->sizeZ());
		}

		/**
	     * \brief Creates the mipmap level specified by the given index.
	     * The level zero is always the original data.
	     * Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling.
	     * This function does nothing if that level is already created.
	     * \param level the mipmap level
	     * \param filter the filter mode
	     */
	    void createMipmapLevel(int level, MipmapFilterMode filter);
	private:
		bool mipmapCheckOrCreate(int level);
		void createMipmapLevelAverage(int level);
		void createMipmapLevelHalton(int level);

	public:
		/**
		 * \brief Deletes all mipmap levels.
		 */
		void deleteAllMipmapLevels();

		/**
		 * \brief clears all GPU resources of the mipmap levels.
		 * Calls \ref MipmapLevel::clearGpuResources() for all levels.
		 */
		void clearGpuResources();

		/**
		 * \brief Returns the mipmap level specified by the given index.
		 * The level zero is always the original data.
		 * If the level is not created yet, <code>nullptr</code> is returned.
		 * Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling
		 */
		std::shared_ptr<const MipmapLevel> getLevel(int level) const;
		/**
		 * \brief Returns the mipmap level specified by the given index.
		 * The level zero is always the original data.
		 * If the level is not created yet, <code>nullptr</code> is returned.
		 * Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling
		 */
		MipmapLevel_ptr getLevel(int level);

		/**
		 * Estimates the memory requirements to store this feature channel.
		 * This is used for the LRU-cache in VolumeEnsembleFactory
		 */
		size_t estimateMemory() const;

		/**
	     * Creates the histogram of the volume.
	     */
		[[nodiscard]] Volume::Histogram_ptr extractHistogram() const;

    };
	typedef std::shared_ptr<Feature> Feature_ptr;

	enum Flags
	{
	    Flag_Compressed = 1,
		//more flags to be added in the future
	};

private:
	float worldSizeX_, worldSizeY_, worldSizeZ_;
	std::vector<Feature_ptr> features_;

public:

	/**
	 * Creates a new, empty volume.
	 * The volume has to be populated by adding features.
	 */
	Volume();

	~Volume() = default;

	Volume(const Volume& other) = delete;
	Volume(Volume&& other) noexcept = delete;
	Volume& operator=(const Volume& other) = delete;
	Volume& operator=(Volume&& other) noexcept = delete;

	static constexpr int NO_COMPRESSION = 0;
	static constexpr int MAX_COMPRESSION = 9;
	
	/**
	 * Saves the volume to the file.
	 * Compression must be within [0, MAX_COMPRESSION],
	 * where 0 means no compression (equal to NO_COMPRESSION)
	 */
	void save(const std::string& filename,
		const VolumeProgressCallback_t& progress,
		const VolumeLoggingCallback_t& logging,
		const VolumeErrorCallback_t& error,
		int compression = NO_COMPRESSION) const;

	/**
	 * Saves the volume to the file.
	 * Compression must be within [0, MAX_COMPRESSION],
	 * where 0 means no compression (equal to NO_COMPRESSION)
	 */
	void save(const std::string& filename, int compression = NO_COMPRESSION) const;

	/**
	 * Loads and construct the volume from the .cvol file
	 */
	Volume(const std::string& filename,
		const VolumeProgressCallback_t& progress,
		const VolumeLoggingCallback_t& logging,
		const VolumeErrorCallback_t& error);

	/**
	 * Loads the volume fro the .cvol file.
	 * This uses default no-op callbacks
	 */
	explicit Volume(const std::string& filename);

	static std::shared_ptr<Volume> loadVolumeFromRaw(
		const std::string& file,
		const VolumeProgressCallback_t& progress,
		const VolumeLoggingCallback_t& logging,
		const VolumeErrorCallback_t& error,
		const std::optional<int>& ensemble = {});
	static std::shared_ptr<Volume> loadVolumeFromRaw(
		const std::string& file,
		const std::optional<int>& ensemble = {});

	static std::shared_ptr<Volume> loadVolumeFromXYZ(
		const std::string& file,
		const VolumeProgressCallback_t& progress,
		const VolumeLoggingCallback_t& logging,
		const VolumeErrorCallback_t& error);
	static std::shared_ptr<Volume> loadVolumeFromXYZ(
		const std::string& file);

	float worldSizeX() const { return worldSizeX_; }
	float worldSizeY() const { return worldSizeY_; }
	float worldSizeZ() const { return worldSizeZ_; }
	void setWorldSizeX(float s) { worldSizeX_ = s; }
	void setWorldSizeY(float s) { worldSizeY_ = s; }
	void setWorldSizeZ(float s) { worldSizeZ_ = s; }

	/**
	 * Returns the number of feature channels
	 */
	int numFeatures() const { return static_cast<int>(features_.size()); }

	/**
	 * Returns the feature channel at the given index.
	 * Throws an exception if the index is out of bounds, see \ref numFeatures()
	 */
	Feature_ptr getFeature(int index) const;

	/**
	 * Searches and returns the feature channel with the given name.
	 * If the feature was not found, a nullptr is returned
	 */
	Feature_ptr getFeature(const std::string& name) const;

	/**
	 * Adds a new feature channel.
	 * \param name the name of the feature
	 * \param type the data type of the feature
	 * \param numChannels the number of channels of that feature
	 * \param sizeX the voxel resolution along X
	 * \param sizeY the voxel resolution along Y
	 * \param sizeZ the voxel resolution along Z
	 */
	Feature_ptr addFeature(const std::string& name, DataType type, int numChannels,
		uint64_t sizeX, uint64_t sizeY, uint64_t sizeZ);

	/**
	 * Creates and adds feature channel from a raw buffer (e.g. from numpy).
	 *
	 * Indexing:
	 *     for (int channel=0; channel<sizes[0]; ++channel)
	 *         for(int x=0; x<sizes[1]; ++x) for (y...) for (z...)
	 *             float value = buffer[channel*strides[0] + x*strides[1] + y*strides[2] + z*...];
	 *
	 * \param buffer the float buffer with the contents
	 * \param sizes the size of the volume in C, X, Y, Z
	 * \param strides the strides of the buffer in C, X, Y, Z
	 * \return the added feature channel
	 */
	Feature_ptr addFeatureFromBuffer(
		const std::string& name,
		const float* buffer, long long sizes[4], long long strides[4]);

	typedef std::function<float(float3)> ImplicitFunction_t;

	/**
	 * \brief Creates a synthetic dataset using the
	 * implicit function 'f'.
	 * The function is called with positions in the range [boxMin, boxMax]
	 * (inclusive bounds), equal-spaced with a resolution of 'resolution'
	 * \param resolution the volume resolution
	 * \param boxMin the minimal coordinate
	 * \param boxMax the maximal coordinate
	 * \param f the generative function
	 * \return the volume
	 */
	static std::unique_ptr<Volume> createSyntheticDataset(
		int resolution, float boxMin, float boxMax,
		const ImplicitFunction_t& f);

	enum class ImplicitEquation
	{
		MARSCHNER_LOBB, //params "fM", "alpha"
		CUBE, //param "scale"
		SPHERE,
		INVERSE_SPHERE,
		DING_DONG,
		ENDRASS,
		BARTH,
		HEART,
		KLEINE,
		CASSINI,
		STEINER,
		CROSS_CAP,
		KUMMER,
		BLOBBY,
		TUBE,
		_NUM_IMPLICIT_EQUATION_
	};

	static std::unique_ptr<Volume> createImplicitDataset(
		int resolution, ImplicitEquation equation,
		const std::unordered_map<std::string, float>& params = {});

	static void registerPybindModules(pybind11::module& m);

	/**
	 * \brief clears all GPU resources of the mipmap levels.
	 * Calls \ref MipmapLevel::clearGpuResources() for all levels.
	 */
	void clearGpuResources();

	/**
	 * Estimates the memory requirements to store this volume.
	 * This is used for the LRU-cache in VolumeEnsembleFactory
	 */
	size_t estimateMemory() const;

	/**
	 * Creates a scaled version of this volume with the same features and world resolution,
	 * but new voxel resolution.
	 * Adaptive average pooling is used.
	 */
	std::shared_ptr<Volume> createScaled(int X, int Y, int Z) const;
	
};
typedef std::shared_ptr<Volume> Volume_ptr;

namespace internal {
	struct EnsembleTimeHash
	{
		std::size_t operator() (const std::pair<int, int>& pair) const {
			return std::hash<int>()(pair.first * 7919 + pair.second);
		}
	};
}

/**
 * Factory class for \ref Volume instances that are part of a timeseries or ensemble.
 *
 * The ensemble has two parameters, "ensemble" and "timestep".
 * The mapping to the filename of the actual volume file is done via the format string,
 * it is called via positional arguments:
 * <code>tinyformat::format(formatString, ensemble, timestep)</code>
 * Examples:
 *  - both ensemble and timestep: "files/ensemble%1$03d/time%2$03d.cvol"
 *	- only ensemble: "files/ensemble%1$03d.cvol"
 *	- only timestep: "files/time%2$03d.cvol"
 * See also: https://github.com/c42f/tinyformat
 *
 * Relative filenames are resolved relative to the loaded json file.
 *
 * The ensemble configuration is saved in a json file.
 */
class VolumeEnsembleFactory
{
	std::string root_;
	std::string formatString_;
	int startEnsemble_ = 0;
	int stepEnsemble_ = 1;
	int numEnsembles_ = 1;
	int startTimestep_ = 0;
	int stepTimestep_ = 1;
	int numTimesteps_ = 1;

	typedef std::pair<int, int> EnsembleTime_t;
	typedef LRUCache<EnsembleTime_t, Volume_ptr, internal::EnsembleTimeHash> Cache_t;
	std::unique_ptr<Cache_t> cache_;

public:
	VolumeEnsembleFactory() = default;

	VolumeEnsembleFactory(const std::string& formatString, int startEnsemble, int numEnsembles, int startTimestep,
		int numTimesteps)
		: formatString_(formatString),
		  startEnsemble_(startEnsemble),
		  numEnsembles_(numEnsembles),
		  startTimestep_(startTimestep),
		  numTimesteps_(numTimesteps),
	      root_("./")
	{
	}

	/**
	 * Loads the ensemble factory settings from the json file specified by the given filename.
	 */
	explicit VolumeEnsembleFactory(const std::string& filename);
	/**
	 * Saves the ensemble factory settings to the file specified by the given filename.
	 */
	void save(const std::string& filename);

	/**
	 * Loads the volume at the given ensemble and time index.
	 * If the volume could not be found (i.e. illegal ensemble or time index, or the dataset is
	 * missing), an empty pointer is returned.
	 */
	Volume_ptr loadVolume(int ensemble, int time);

	std::string getVolumeFilename(int ensemble, int time);

    [[nodiscard]] virtual std::string root() const
    {
        return root_;
    }

    virtual void setRoot(const std::string& root)
    {
        root_ = root;
    }

    [[nodiscard]] virtual std::string formatString() const
	{
		return formatString_;
	}

	virtual void setFormatString(const std::string& formatString)
	{
		formatString_ = formatString;
	}

	[[nodiscard]] virtual int startTimestep() const
	{
		return startTimestep_;
	}

	virtual void setStartTimestep(const int startTimestep)
	{
		startTimestep_ = startTimestep;
	}

    [[nodiscard]] virtual int stepEnsemble() const
    {
        return stepEnsemble_;
    }

    virtual void setStepEnsemble(const int stepEnsemble)
    {
        stepEnsemble_ = stepEnsemble;
    }

    [[nodiscard]] virtual int numTimesteps() const
	{
		return numTimesteps_;
	}

	virtual void setNumTimesteps(const int numTimesteps)
	{
		numTimesteps_ = numTimesteps;
	}

	[[nodiscard]] virtual int startEnsemble() const
	{
		return startEnsemble_;
	}

	virtual void setStartEnsemble(const int startEnsemble)
	{
		startEnsemble_ = startEnsemble;
	}

    [[nodiscard]] virtual int stepTimestep() const
    {
        return stepTimestep_;
    }

    virtual void setStepTimestep(const int stepTimestep)
    {
        stepTimestep_ = stepTimestep;
    }

    [[nodiscard]] virtual int numEnsembles() const
	{
		return numEnsembles_;
	}

	virtual void setNumEnsembles(const int numEnsembles)
	{
		numEnsembles_ = numEnsembles;
	}

	static void registerPybindModules(pybind11::module& m);
};
typedef std::shared_ptr<VolumeEnsembleFactory> VolumeEnsembleFactory_ptr;

END_RENDERER_NAMESPACE

#ifdef _WIN32
#pragma warning( pop )
#endif
