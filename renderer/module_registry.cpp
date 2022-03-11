#include "module_registry.h"
#include "parameter.h"

#include <deque>

#include "volume_interpolation_grid.h"
#include "transfer_function.h"
#include "blending.h"
#include "camera.h"
#include "ray_evaluation_stepping.h"
#include "ray_evaluation_monte_carlo.h"
#include "image_evaluator_simple.h"
#include "brdf.h"
#include "phase_function.h"
#include "spherical_harmonics.h"
#include "irasterization.h"
#include "pytorch_functions.h"
#include "volume_interpolation_network.h"
#include "volume_interpolation_implicit.h"

renderer::ModuleRegistry::ModuleRegistry()
	: workingDirectory_(absolute(std::filesystem::path(".")))
{
}

renderer::ModuleRegistry& renderer::ModuleRegistry::Instance()
{
	static ModuleRegistry INSTANCE;
	return INSTANCE;
}

void renderer::ModuleRegistry::registerModule(const std::string& tag, IModule_ptr module, ModuleCreationFunction_t f)
{
	map_[tag].push_back(std::make_pair(module, f));
}

void renderer::ModuleRegistry::registerPybindModules(pybind11::module& m)
{
	//register parameters
	namespace py = pybind11;

#define PARAMETER_NON_DIFFERENTIABLE(T)										\
	py::class_<Parameter<T>>(m, "Parameter_" C10_STRINGIZE(T))				\
		.def_readwrite("value", &Parameter<T>::value)						\
		.def_property_readonly("supports_gradients", []()					\
			{return static_cast<bool>(Parameter<T>::supportsGradients); });
	PARAMETER_NON_DIFFERENTIABLE(bool);
	PARAMETER_NON_DIFFERENTIABLE(int);
	PARAMETER_NON_DIFFERENTIABLE(int2);
	PARAMETER_NON_DIFFERENTIABLE(int3);
	PARAMETER_NON_DIFFERENTIABLE(int4);
#undef PARAMETER_NON_DIFFERENTIABLE

#define PARAMETER_DIFFERENTIABLE(T)											\
	py::class_<Parameter<T>>(m, "Parameter_" C10_STRINGIZE(T))				\
		.def_readwrite("value", &Parameter<T>::value)						\
		.def_readwrite("grad", &Parameter<T>::grad)							\
		.def_readwrite("forwardIndex", &Parameter<T>::forwardIndex)			\
		.def_property_readonly("supports_gradients", []()					\
			{return static_cast<bool>(Parameter<T>::supportsGradients); });
	PARAMETER_DIFFERENTIABLE(double);
	PARAMETER_DIFFERENTIABLE(double2);
	PARAMETER_DIFFERENTIABLE(double3);
	PARAMETER_DIFFERENTIABLE(double4);
	using torch::Tensor;
	PARAMETER_DIFFERENTIABLE(Tensor);
#undef PARAMETER_DIFFERENTIABLE
	
	//register modules
	for (const auto& x : map_)
		for (const auto& v : x.second)
			v.first->registerPybindModule(m);

	//loading function
	m.def("load_from_json", [this](const std::string& filename)
		{
			std::ifstream i(filename);
            if (!i.is_open())
            {
                std::cerr << "Unable to open file " << filename << std::endl;
                throw std::runtime_error("Unable to open file");
            }
			nlohmann::json settings;
            try {
                i >> settings;
            } catch (const std::exception& ex)
            {
                std::cerr << "Unable to load json with the settings (" << filename << "): " << ex.what() << std::endl;
                throw;
            }
			std::string rootName = settings.value("root", "");
			std::filesystem::path rootPath = absolute(std::filesystem::path(filename)).parent_path();
			return std::dynamic_pointer_cast<IImageEvaluator>(this->loadTree(
				settings, IImageEvaluator::Tag(), rootName, rootPath));
		}, py::arg("filename"), py::doc(R"(
	Loads the module tree from the specified filename.
	Only the selected image evaluator and the dependent modules are loaded.
	The resulting module and child modules are new instances.
)"));

	//other utilities
	SphericalHarmonics::registerPybindModules(m);
	Volume::registerPybindModules(m);
	VolumeEnsembleFactory::registerPybindModules(m);
	PytorchFunctions::registerPybindModule(m);
}

void renderer::ModuleRegistry::populateModules()
{
#define REGISTER(T)	\
	RegisterModule([](){return std::make_shared<T>();})

	REGISTER(RasterizationContainer);
	REGISTER(VolumeInterpolationGrid);
	REGISTER(TransferFunctionIdentity);
	REGISTER(TransferFunctionTexture);
	REGISTER(TransferFunctionPiecewiseLinear);
	REGISTER(TransferFunctionGaussian);
	REGISTER(Blending);
	REGISTER(CameraOnASphere);
	REGISTER(RayEvaluationSteppingIso);
	REGISTER(RayEvaluationSteppingDvr);
	REGISTER(RayEvaluationMonteCarlo);
	REGISTER(ImageEvaluatorSimple);
	REGISTER(BRDFLambert);
	REGISTER(PhaseFunctionHenyeyGreenstein);
	REGISTER(PhaseFunctionRayleigh);
	REGISTER(VolumeInterpolationNetwork);
	REGISTER(VolumeInterpolationImplicit);
}

const std::vector<std::pair<renderer::IModule_ptr, renderer::ModuleRegistry::ModuleCreationFunction_t>>& 
	renderer::ModuleRegistry::getModulesForTag(const std::string& tag) const
{
	auto it = map_.find(tag);
	TORCH_CHECK(it != map_.end(), "Tag ", tag, " not found!");
	return it->second;
}

std::shared_ptr<renderer::IModule> renderer::ModuleRegistry::getModule(const std::string& tag,
	const std::string& name) const
{
	for (const auto& m : getModulesForTag(tag))
	{
		if (m.first->getName() == name)
			return m.first;
	}
	return {};
}

std::shared_ptr<renderer::IModule> renderer::ModuleRegistry::createModule(const std::string& tag,
	const std::string& name) const
{
	for (const auto& m : getModulesForTag(tag))
	{
		if (m.first->getName() == name)
			return m.second();
	}
	return {};
}

namespace {
	class GlobalLoadingContext : public renderer::ILoadingContext
	{
		const renderer::ModuleRegistry* const registry_;
		std::filesystem::path rootPath_;
	public:

		GlobalLoadingContext(const renderer::ModuleRegistry* const registry,
			const std::filesystem::path rootPath)
			: registry_(registry),
			rootPath_(rootPath)
		{
		}

		std::shared_ptr<renderer::IModule> getModule(const std::string& tag, const std::string& name) const override
		{
			return registry_->getModule(tag, name);
		}

		std::filesystem::path getRootPath() const override
		{
			return rootPath_;
		}
	};
	class BasicSavingContext : public renderer::ISavingContext
	{
		std::filesystem::path rootPath_;
	public:

		explicit BasicSavingContext(const std::filesystem::path& rootPath)
			: rootPath_(rootPath)
		{
		}

		std::filesystem::path getRootPath() const override
		{
			return rootPath_;
		}
	};
}

void renderer::ModuleRegistry::loadAll(const nlohmann::json& root, const std::filesystem::path& rootPath)
{
	GlobalLoadingContext f(this, rootPath);
	for (const auto& x : map_)
	{
		const std::string& tag = x.first;
		if (!root.contains(tag))
		{
			std::cerr << "No object for tag '" << tag << "' found in the save file" << std::endl;
			continue;
		}
		const auto& tagObj = root[tag];
		for (const auto & v : x.second)
		{
			const std::string& name = v.first->getName();
			if (!tagObj.contains(name))
			{
				std::cerr << "No object for module '" << name << "' of tag '" << tag << "' found in the save file" << std::endl;
				continue;
			}
			
			v.first->load(tagObj[name], &f);
		}
	}
}

void renderer::ModuleRegistry::saveAll(nlohmann::json& root, const std::filesystem::path& rootPath)
{
	BasicSavingContext ctx(rootPath);
	for (const auto& x : map_)
	{
		const std::string& tag = x.first;
		nlohmann::json tagObj = nlohmann::json::object();
		for (const auto& v : x.second)
		{
			const std::string& name = v.first->getName();
			nlohmann::json moduleObj = nlohmann::json::object();
			v.first->save(moduleObj, &ctx);
			tagObj[name] = moduleObj;
		}
		root[tag] = tagObj;
	}
}

namespace {
	class LocalLoadingContext : public renderer::ILoadingContext
	{
		const renderer::ModuleRegistry* const registry_;
		std::deque<renderer::IModule_ptr>* modulesToLoad_;
		std::filesystem::path rootPath_;
		mutable std::map<std::pair<std::string, std::string>, std::weak_ptr<renderer::IModule>> loadedModules_;
	public:

		LocalLoadingContext(const renderer::ModuleRegistry* const registry,
			std::deque<renderer::IModule_ptr>* modulesToLoad, const std::filesystem::path rootPath)
			: registry_(registry),
			  modulesToLoad_(modulesToLoad),
			  rootPath_(rootPath)
		{
		}

		std::shared_ptr<renderer::IModule> getModule(const std::string& tag, const std::string& name) const override
		{
			//check if already loaded to prevent infinite loops
			const auto key = std::make_pair(tag, name);
			auto it = loadedModules_.find(key);
			if (it != loadedModules_.end())
			{
				auto m = it->second.lock();
				if (m) return m;
			}

			auto m = registry_->createModule(tag, name);
			loadedModules_.insert(std::make_pair(key, std::weak_ptr<renderer::IModule>(m)));
			if (m) modulesToLoad_->push_back(m);
			return m;
		}

		std::filesystem::path getRootPath() const override
		{
			return rootPath_;
		}
	};
}

renderer::IModule_ptr renderer::ModuleRegistry::loadTree(
	const nlohmann::json& root, const std::string& tag,	const std::string& name,
	const std::filesystem::path& rootPath) const
{
	std::deque<renderer::IModule_ptr> modulesToLoad;
	LocalLoadingContext fetcher(this, &modulesToLoad, rootPath);
	IModule_ptr rootModule = fetcher.getModule(tag, name);
	if (rootModule)
	{
		modulesToLoad.push_back(rootModule);
		while (!modulesToLoad.empty())
		{
			auto m = modulesToLoad.front(); modulesToLoad.pop_front();
			m->load(root[m->getTag()][m->getName()], &fetcher);
		}
	}
	return rootModule;
}

void renderer::ModuleRegistry::mapTree(IModuleContainer_ptr root, const std::function<void(IModule_ptr)>& f)
{
	f(root);
	for (const auto& tag : root->getSupportedTags())
	{
		auto m = root->getSelectedModuleForTag(tag);
		if (m) {
			if (auto mc = std::dynamic_pointer_cast<IModuleContainer>(m))
				mapTree(mc, f);
			else
				f(m); //leave
		}
	}
}
