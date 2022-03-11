#pragma once

#include <cuda.h>
#include <imgui.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <json.hpp>
#include <vector>
#include <memory>
#include <optional>
#include <c10/core/ScalarType.h>
#include <unordered_map>
#include <any>
#include <filesystem>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>

#include "commons.h"

BEGIN_RENDERER_NAMESPACE

void throwOnError(CUresult err, const char* file, int line);
#define CU_SAFE_CALL( err ) RENDERER_NAMESPACE ::throwOnError( err, __FILE__, __LINE__ )

class ModuleRegistry;
class IModule;
class IModuleContainer;

struct ILoadingContext
{
	virtual ~ILoadingContext() = default;

	/**
	 * Returns the module for the specified tag and name if found.
	 */
	virtual std::shared_ptr<IModule> getModule(const std::string& tag, const std::string& name) const = 0;

	/**
	 * Returns the root path during loading to resolve relative imports.
	 */
	virtual std::filesystem::path getRootPath() const = 0;

	//Space to add more functions
};

struct ISavingContext
{
	virtual ~ISavingContext() = default;
	
	/**
	 * Returns the root path during saving to resolve relative imports.
	 */
	virtual std::filesystem::path getRootPath() const = 0;

	//Space to add more functions
};

/**
 * Context for rasterizing.
 */
struct RasterizingContext
{
	glm::vec3 origin;
	glm::mat4 projection;
	glm::mat4 view;
	glm::mat4 normal;

	/**
	 * The root module, can be used to query what other modules are
	 * currently used to validate and adapt the current module.
	 */
	std::shared_ptr<IModuleContainer> root;
};

/**
 * \brief the base class of all Modules.
 * In the UI, modules are singletones.
 *
 * Module Hierarchy:
 * (|- inheritance; -> contains)
 * <pre>
 * IImageEvaluator
 *  |- ImageEvaluatorNeuralTexture
 *  |  -> ICamera
 *  |
 *  |- ImageEvaluatorSimple
 *	|  -> ICamera
 *	|  -> IVolumeInterpolation
 *	|  -> IRayEvaluation
 *	|  -> IRasterization
 *
 * ICamera
 *  |- CameraOnASphere
 *
 * IVolumeInterpolation
 *  |- VolumeInterpolationGrid
 *
 * IRayEvaluation
 *  |- RayEvaluationStepping
 *	|  |- RayEvaluatorSteppingIso
 *	|  |- RayEvaluationSteppingDvr
 *	|  |   -> Blending
 *	|  |   -> ITransferFunction
 *	|  |   -> IBRDF
 *	|- RayEvaluatorMonteCarlo
 *	|   -> ITransferFunction
 *	|   -> IPhaseFunction
 *
 * Blending
 *  (no subclasses)
 *
 * ITransferFunction
 *  |- TransferFunctionIdentity
 *	|- TransferFunctionTexture
 *	|- TransferFunctionPiecewiseLinear
 *	|- TransferFunctionGaussian
 *
 * IBRDF
 *  |- BRDFLambert
 *
 * IPhaseFunction
 *  |- PhaseFunctionHenyeyGreenstein
 *	|- PhaseFunctionRayleigh
 *
 *
 * IRasterization
 *  |- ParticleIntegration
 *
 * </pre>
 */
class IModule
{
public:
	virtual ~IModule() = default;

	/**
	 * The tag declares the logical group of this module. Only one instance of a tag
	 * can be used in a rendering.
	 * Examples: TF, Volume.
	 *
	 * TODO: make it return std::string_view?
	 */
	[[nodiscard]] virtual std::string getTag() const = 0;
	/**
	 * The name of this particular module inside the logical group (tag).
	 * Example for tag "TF": TF-Gaussian, TF-Texture, TF-Identity
	 */
	[[nodiscard]] virtual std::string getName() const = 0;

	typedef std::unordered_map<std::string, std::any> UIStorage_t;
	/**
	 * Draws an updates the ImGui-UI.
	 * Returns true iff there are some changes and the scene should be redrawn.
	 *
	 * <b>This function must call \ref drawUI(UIStorage_t&) of the children</b>.
	 * 
	 * \param storage persistent storage to exchange data between different modules.
	 *   For example: the volume module stores the histogram that is used by the TF
	 *   in the background
	 */
	[[nodiscard]] virtual bool drawUI(UIStorage_t& storage) = 0;
	/**
	 * Draws extra information on the current frame in the information box on the top right.
	 * Called for all active modules.
	 */
	virtual void drawExtraInfo(UIStorage_t& storage) {}
	/**
	 * on-update of the UI. Called for all active modules prior to drawUI, regardless
	 * of if the module is visible or collapsed (drawUI called or not).
	 * \return true if changed and the image should be redrawn
	 */
	[[nodiscard]] virtual bool updateUI(UIStorage_t& storage) { return false; }

	template<typename T>
	static T get_or(const UIStorage_t& storage, const std::string& key, const T& alternative)
	{
		if (const auto& it = storage.find(key);
			it != storage.end())
		{
			return std::any_cast<T>(it->second);
		}
		return alternative;
	}

	virtual void load(const nlohmann::json& json, const ILoadingContext* fetcher) = 0;
	virtual void save(nlohmann::json& json, const ISavingContext* context) const = 0;

	/**
	 * Returns true if this module (or any child module) also wants to do some
	 * rasterization instead of only raytracing.
	 * This is used to draw spheres at the position of the lights for visualization
	 * or to include streamlines together with the direct volume rendering.
	 *
	 * If at least one module returns true, \ref performRasterization() is called.
	 *
	 * This function should be re-entrant, it might be called multiple times
	 * before \ref performRasterization() is invoked.
	 */
	virtual bool hasRasterizing() const { return false; }

	/**
	 * Performs rasterization.
	 * Rasterization is done prior to the volume raytracing into an offscreen framebuffer
	 * (already set with correct viewports).
	 * The depth buffer is used as end point for the raytracing for proper blending.
	 */
	virtual void performRasterization(const RasterizingContext* context) {}

private:
	friend class ModuleRegistry;
	/**
	 * Called by ModuleRegistry when initializing the pybind module.
	 * Do not call manually! Only this way it is ensured that is called only once per implementation.
	 */
	virtual void registerPybindModule(pybind11::module& m) = 0;
};
typedef std::shared_ptr<IModule> IModule_ptr;

/**
 * A module that contains other modules.
 * For example, the entry module contains the volume, tf, ..., as children.
 * Its main use here is to provide a reference to the
 * selected module for a specific tag, so that modules can exchange additional data.
 *
 * Note on drawUI, load/save, kernel modules:
 * - drawUI: manually call \ref IModule::drawUI from the subclasses.
 *   This allows for better placement of the UI elements and
 *   allows the module container to decide if the selection
 *   of the child module should be shared across different instances
 *   (via UIStorage) or not in the UI
 * - load/save: automatically called on all registered modules in
 *   \ref ModuleRegistry
 * - kernel creation methods in \ref IKernelModule: automatically
 *   called when assembling the kernel. The selected modules for each tag
 *   is queried from the containers -> Only one instance per tag
 *   can be used for rendering at once.
 */
class IModuleContainer : public virtual IModule
{
public:
	/**
	 * Returns the selected module in the current processing pipeline
	 * for the selected tag, if found.
	 * This function needs to be transitive: If the module container
	 * contains other containers, those have to queried for the
	 * specified tag as well if not found directly.
	 */
	[[nodiscard]] virtual IModule_ptr getSelectedModuleForTag(const std::string& tag) const = 0;
	/**
	 * Returns the list of tags for the modules that are contained
	 * in this container.
	 */
	[[nodiscard]] virtual std::vector<std::string> getSupportedTags() const = 0;

	//Default implementation that simply checks every child.
    bool hasRasterizing() const override
    {
		bool b = false;
		for (const auto& tag : getSupportedTags())
		{
			if (getSelectedModuleForTag(tag)->hasRasterizing())
				b = true;
		}
		return b;
    }
	//Default implementation that simply checks every child.
    void performRasterization(const RasterizingContext* context) override
    {
		for (const auto& tag : getSupportedTags())
		{
			const auto m = getSelectedModuleForTag(tag);
			if (m->hasRasterizing())
				m->performRasterization(context);
		}
    }
};
typedef std::shared_ptr<IModuleContainer> IModuleContainer_ptr;

/**
 * A module that is part of a CUDA kernel.
 * Contains methods for code generation.
 *
 * A kernel module is called in the following way, where
 * the variables in brackets {} are filled by the respective 
 * <pre>
 * #include "{getIncludeFileNames()}"
 *
 * //constant memory with the parameters
 * __constant__ {getParameterType()} c_{getTag()};
 *
 * __global__/__device__ kernel(...)
 * {
 *    //thread-local structure for caching
 *    {getPerThreadType()} l_{getTag()};
 *
 *    //calling
 *    auto ret = l_{getTag()}(params);
 * }
 *
 * </pre>
 */
class IKernelModule : public virtual IModule
{
public:

	/**
	 * Settings populated by the master module that launches the kernel
	 * and is passed down to the kernel modules
	 */
	struct GlobalSettings
	{
		//c10::kFloat or c10::kDouble
		c10::ScalarType scalarType;
		static constexpr c10::ScalarType kFloat = c10::kFloat;
		static constexpr c10::ScalarType kDouble = c10::kDouble;

		enum VolumeOutput
		{
		    Density, //Density -> real_t
			Velocity, //Velocity -> real3
			Color //RGBA-Color -> real4
		};
		/**
		 * The output from the volume interpolation, defining the expected datatype
		 */
		VolumeOutput volumeOutput = Density;

		// Transfer Function -> Volume Interpolation
		bool volumeShouldProvideNormals = false;
		// Transfer Function -> Volume Interpolation
		// The VolumeInterpolation should be prepared to be asked
		// to provide the first and second principal curvatures.
		bool volumeShouldProvideCurvature = false;
		// Ray Evaluation -> Volume Interpolation
		// True: position is in object space [0,resX]x[0,resY]x[0,resY]
		// False: position in world space [boxMinX, boxMaxX]x...
		bool interpolationInObjectSpace = false;

		/**
		 * The root module, can be used to query what other modules are
		 * currently used to validate and adapt the current module.
		 */
		IModuleContainer_ptr root;

		/**
		 * If true, the threads in a warp are synchronized,
		 * meaning they don't terminate early.
		 * Tracing is only stopped if all threads per warp reach termination.
		 * This is needed for the scene representation networks.
		 */
		bool synchronizedThreads = false;

		/**
		 * If set to non-zero, a fixed block size is requested.
		 * The rendering fails if that block size can't be fullfilled
		 */
		int fixedBlockSize = 0;
	};

	/**
	 * Called before all other methods below to allow
	 * the kernel modules to update settings (i.e.
	 * \ref volumeShouldProvideNormals and \ref interpolationInObjectSpace.
	 */
	virtual void prepareRendering(GlobalSettings& s) const {};
	/**
	 * If the parameters of this kernel module contains a batch dimension,
	 * returns the number of batches. If not, return an empty optional.
	 * This is used to decide how many batches are rendered. If batches are used,
	 * they must coincide across all modules.
	 */
	[[nodiscard]] virtual std::optional<int> getBatches(const GlobalSettings& s) const { return {}; }
	/**
	 * Additional string that is included at the very beginning for global
	 * #define's.
	 */
	[[nodiscard]] virtual std::string getDefines(const GlobalSettings& s) const { return ""; }
	/**
	 * The list of filenames to include
	 */
	[[nodiscard]] virtual std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const = 0;
	/**
	 * Allows the kernel module to insert arbitrary code (code generation).
	 * This code is inserted before the include files
	 */
	virtual void fillExtraSourceCode(const GlobalSettings& s, std::stringstream& ss) const {}
	/**
	 * the name of the __constant__ memory declaration with the parameters.
	 * If an empty string is returned, no constant memory is declared
	 * and \ref fillConstantMemory is not called
	 */
	[[nodiscard]] virtual std::string getConstantDeclarationName(const GlobalSettings& s) const = 0;
	/**
	 * The type of the per-thread instance
	 */
	[[nodiscard]] virtual std::string getPerThreadType(const GlobalSettings& s) const = 0;
	/**
	 * Fills the constant memory with the current settings.
	 * This method is only called if \ref getParameterType() is not empty.
	 *
	 * Not const, as some pre-computations (with caching) might be needed.
	 * Also perfom validations here.
	 */
	virtual void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) = 0;
};
typedef std::shared_ptr<IKernelModule> IKernelModule_ptr;


END_RENDERER_NAMESPACE
