#pragma once

#include "imodule.h"
#include "opengl_mesh.h"
#include "opengl_shader.h"
#include "parameter.h"

BEGIN_RENDERER_NAMESPACE

/**
 * Defines the bidirection reflection distribution function,
 * i.e. it modulates the rgb-absorption based on the current position and gradient.
 * It is called in the following way:
 * <code>
 *  real4 rgb_absorption = brdf.eval(real4 rgb_absorption, real3 position, real3 gradient, real3 rayDir, int batch)
 * </code>
 * The gradient is computed only if \ref IKernelModule::GlobalSettings::volumeShouldProvideNormals
 * is true.
 */
class IBRDF : public IKernelModule, public std::enable_shared_from_this<IBRDF>
{

public:
	static constexpr std::string_view TAG = "brdf";
	std::string getTag() const override { return std::string(TAG); }
	static std::string Tag() { return std::string(TAG); }
};
typedef std::shared_ptr<IBRDF> IBRDF_ptr;

//built-in BRDF

/*
 * Simple lambert diffuse BRDF.
 *
 * It first scales the absorption alpha according to
 * \f$ alpha = alpha * (1-exp(-magnitudeScaling_ * ||gradient||_2^2))\f$, 
 * if enableMagnitudeScaling_ is true.
 *
 * Second, it performs Phong shading according to
 * \f$
 *  real_t ambient = Ambient light scrength in [0,1]
 *  real_t specular = Specular light strength in [0,1]
 *  real_t minMagnitude, maxMagnitude = min-max boundaries for when to use shading
 *
 *  real_t phongStrength = smoothstep(||gradient||, minMagnitude, maxMagnitude)
 *  real3 baseColor = rgb_absorption.rgb
 *  real3 outColor = lerp(1,ambient,phongStrength) * baseColor +
 *    (1-lerp(1,ambient,phongStrength)) * [normal.light*baseColor + specular(ray.reflect)^n]
 * \f$
 *
 * All parameters are either all scalar or all batched with tensors of shape (B,)
 */
class BRDFLambert : public IBRDF
{
	bool enableMagnitudeScaling_;
	Parameter<double> magnitudeScaling_;

	bool enablePhong_;
	Parameter<double> ambient_;
	Parameter<double> specular_;
	Parameter<double> magnitudeCenter_; //for the smoothstep of the phong influence
	Parameter<double> magnitudeRadius_;
	Parameter<int> specularExponent_;

	enum class LightType
	{
		Point, Directional
	};
	bool lightFollowsCamera_;
	LightType lightType_;
	Parameter<double3> lightPosition_;
	Parameter<double3> lightDirection_;

	//light UI
	bool showLight_;
#if RENDERER_OPENGL_SUPPORT==1
	Mesh lightMesh_;
	Shader lightShader_;
#endif

public:
	BRDFLambert();


	[[nodiscard]] std::string getName() const override;
	[[nodiscard]] bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* context) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
private:
	void registerPybindModule(pybind11::module& m) override;
public:
	void prepareRendering(GlobalSettings& s) const override;
	[[nodiscard]] std::optional<int> getBatches(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getDefines(const GlobalSettings& s) const override;
	[[nodiscard]] std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	[[nodiscard]] std::string getPerThreadType(const GlobalSettings& s) const override;
	void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;
    bool hasRasterizing() const override;
    void performRasterization(const RasterizingContext* context) override;

private:
	bool updateLightFromCamera(const GlobalSettings& s);
};


END_RENDERER_NAMESPACE
