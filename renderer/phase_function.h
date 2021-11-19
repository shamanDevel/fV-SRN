#pragma once

#include "imodule.h"
#include "parameter.h"

BEGIN_RENDERER_NAMESPACE

/**
 * Defines the phase function for volumetric scattering,
 * It defines two functions:
 * <code>
 *  real_t prob(real3 dirIn, real3 dirOut, real3 pos, int batch);
 *  template<typename Sampler>
 *  real3 sample(real3 dirIn, real3 pos, Sampler& sampler, int batch);
 * </code>
 *
 * The first method computes the probability that the incoming ray 'dirIn'
 * at position 'pos' is scattered into the outgoing direction 'dirOut'.
 * The second method randomly samples such an outgoing direction.
 *
 * By convention, 'dirIn' points to the point 'pos', 'dirOut' points away.
 * Hence the angle between the directions is
 * <code>cos(theta) = dot(-dirIn, dirOut)</code>.
 * 
 */
class IPhaseFunction : public IKernelModule, public std::enable_shared_from_this<IPhaseFunction>
{

public:
	static constexpr std::string_view TAG = "phase";
	std::string getTag() const override { return std::string(TAG); }
	static std::string Tag() { return std::string(TAG); }
protected:
	void registerPybindModule(pybind11::module& m) override;
};
typedef std::shared_ptr<IPhaseFunction> IPhaseFunction_ptr;

//built-in Phase functions

/*
 * Henyey-Greenstein
 */
class PhaseFunctionHenyeyGreenstein : public IPhaseFunction
{
	Parameter<double> g_;

public:
	PhaseFunctionHenyeyGreenstein();

	[[nodiscard]] std::string getName() const override;
	[[nodiscard]] bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
public:
	void prepareRendering(GlobalSettings& s) const override;
	[[nodiscard]] std::optional<int> getBatches(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getDefines(const GlobalSettings& s) const override;
	[[nodiscard]] std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getPerThreadType(const GlobalSettings& s) const override;
	void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;

	Parameter<double>& g() { return g_; }
	const Parameter<double>& g() const { return g_; }
};

/*
 * Rayleigh
 */
class PhaseFunctionRayleigh : public IPhaseFunction
{
	Parameter<double> g_;

public:
	PhaseFunctionRayleigh();

	[[nodiscard]] std::string getName() const override;
	[[nodiscard]] bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
public:
	void prepareRendering(GlobalSettings& s) const override;
	[[nodiscard]] std::optional<int> getBatches(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getDefines(const GlobalSettings& s) const override;
	[[nodiscard]] std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getPerThreadType(const GlobalSettings& s) const override;
	void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;

	Parameter<double>& g() { return g_; }
	const Parameter<double>& g() const { return g_; }
};


END_RENDERER_NAMESPACE
