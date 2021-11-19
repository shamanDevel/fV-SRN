#pragma once

#include "imodule.h"
#include "parameter.h"

BEGIN_RENDERER_NAMESPACE

/**
 * Defines the blending
 * <pre>
 *  real4 rgb_alpha = blend.eval(real4 rgb_alpha_previous, real4 rgb_absorption_new_contribution)
 * </pre>
 * The scaling with the stepsize is done in the TF.
 */
class Blending : public IKernelModule
{
public:
	enum class BlendMode
	{
		Alpha,
		BeerLambert
	};

	Blending() = default;
	virtual ~Blending() = default;

	static constexpr std::string_view TAG = "blending";
	std::string getTag() const override { return std::string(TAG); }
	std::string getName() const override;
	bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* context) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
private:
	void registerPybindModule(pybind11::module& m) override;
public:
	std::optional<int> getBatches(const GlobalSettings& s) const override;
	std::string getDefines(const GlobalSettings& s) const override;
	std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	std::string getPerThreadType(const GlobalSettings& s) const override;
	void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;

	[[nodiscard]] virtual BlendMode blendMode() const
	{
		return blendMode_;
	}

	virtual void setBlendMode(const BlendMode blendMode)
	{
		blendMode_ = blendMode;
	}

private:
	BlendMode blendMode_ = BlendMode::BeerLambert;
};
typedef std::shared_ptr<Blending> Blending_ptr;

END_RENDERER_NAMESPACE