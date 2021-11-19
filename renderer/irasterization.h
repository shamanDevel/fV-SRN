#pragma once

#include "imodule.h"
#include "volume_interpolation.h"

BEGIN_RENDERER_NAMESPACE

class RasterizationContainer;

/**
 * Base class for modules that perform rasterization to draw their information.
 * Because this is not included in the raytracing, they don't inherit from IModule,
 * instead, they are listed in \ref RasterizationContainer, a subclass of IModule.
 */
class IRasterization : public std::enable_shared_from_this<IRasterization>
{
public:
	IRasterization() = default;
	virtual ~IRasterization() = default;

	virtual [[nodiscard]] std::string getName() const = 0;
	virtual [[nodiscard]] bool drawUI(IModule::UIStorage_t& storage) = 0;
	virtual void drawExtraInfo(IModule::UIStorage_t& storage) {}
	virtual [[nodiscard]] bool updateUI(IModule::UIStorage_t& storage) { return false; }
	virtual void load(const nlohmann::json& json, const ILoadingContext* fetcher) = 0;
	virtual void save(nlohmann::json& json, const ISavingContext* context) const = 0;
	virtual void performRasterization(const RasterizingContext* context) = 0;
protected:
	virtual void registerPybindModule(pybind11::module& m);
	friend class RasterizationContainer;

	/**
	 * Returns the selected volume for the UI, might be null.
	 */
	IVolumeInterpolation_ptr getSelectedVolume(const IModule::UIStorage_t& s) const;
	/**
	 * Returns the selected volume during rendering.
	 */
	IVolumeInterpolation_ptr getSelectedVolume(const RasterizingContext* c) const;
};
typedef std::shared_ptr<IRasterization> IRasterization_ptr;

/**
 * Contains the IRasterization instances.
 */
class RasterizationContainer : public IModule
{
	std::list<IRasterization_ptr> rasterizations_;

	typedef std::function<IRasterization_ptr()> Factory_t;
	static std::unordered_map<std::string, Factory_t> Implementations_;
	static void RegisterImplementation();
public:
	RasterizationContainer();

	static constexpr std::string_view TAG = "Rasterization";
	std::string getTag() const override { return std::string(TAG); }
	static std::string Tag() { return std::string(TAG); }

	static constexpr std::string_view NAME = "Container";
	std::string getName() const override { return std::string(NAME); }
	static std::string Name() { return std::string(NAME); }

    [[nodiscard]] bool drawUI(UIStorage_t& storage) override;
    void drawExtraInfo(UIStorage_t& storage) override;
    [[nodiscard]] bool updateUI(UIStorage_t& storage) override;
    void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
    void save(nlohmann::json& json, const ISavingContext* context) const override;
    bool hasRasterizing() const override;
    void performRasterization(const RasterizingContext* context) override;

	int numRasterizations() const { return rasterizations_.size(); }
    static std::vector<std::string> ImplementationNames();
	IRasterization_ptr addImplementation(const std::string& name);
private:
    void registerPybindModule(pybind11::module& m) override;
};
typedef std::shared_ptr<RasterizationContainer> RasterizationContainer_ptr;

END_RENDERER_NAMESPACE
