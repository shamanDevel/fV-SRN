#pragma once

#include <unordered_map>
#include <functional>
#include "imodule.h"

BEGIN_RENDERER_NAMESPACE


class ModuleRegistry
{
public:
	typedef std::function<IModule_ptr()> ModuleCreationFunction_t;
	
private:
	ModuleRegistry();
	const std::filesystem::path workingDirectory_;
	std::unordered_map<std::string, std::vector<std::pair<IModule_ptr, ModuleCreationFunction_t>>> map_;

public:
	static ModuleRegistry& Instance();

	void registerModule(const std::string& tag, IModule_ptr module, ModuleCreationFunction_t f);
	static void RegisterModule(ModuleCreationFunction_t f)
	{
		IModule_ptr m = f(); //the global instance for the UI and python registry
		Instance().registerModule(m->getTag(), m, f);
	}

	void registerPybindModules(pybind11::module& m);

	/**
	 * \brief call upon startup to populate the modules.
	 * Auto-registration does not work because the renderer is
	 * usually compiled as static library.
	 */
	static void populateModules();

	const std::vector<std::pair<IModule_ptr, ModuleCreationFunction_t>>& getModulesForTag(const std::string& tag) const;

	std::shared_ptr<IModule> getModule(const std::string& tag, const std::string& name) const;
	[[nodiscard]] std::shared_ptr<IModule> createModule(const std::string& tag, const std::string& name) const;
	
	void loadAll(const nlohmann::json& root, const std::filesystem::path& rootPath);
	void saveAll(nlohmann::json& root, const std::filesystem::path& rootPath);

	/**
	 * \brief Loads the module tree starting from the specified start 'tag' and 'name'
	 * from the json setting file 'root'.
	 * 
	 * This function is used in the python side. The returned modules are freshly created.
	 * If all modules should be loaded in the global instances (for the UI), use \ref loadAll.
	 */
	IModule_ptr loadTree(const nlohmann::json& root, 
		const std::string& tag, const std::string& name,
		const std::filesystem::path& rootPath) const;

	/**
	 * Calls \c f for all children in the module tree spanned by \c root.
	 */
	static void mapTree(IModuleContainer_ptr root, const std::function<void(IModule_ptr)>& f);
};

END_RENDERER_NAMESPACE
