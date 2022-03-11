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

	/**
	 * Evaluates the BRDF at the given position and color.
	 * All tensors are of shape (B,C)
	 */
	virtual torch::Tensor evaluate(
		const torch::Tensor& rgba,
		const torch::Tensor& position,
		const torch::Tensor& gradient,
		const torch::Tensor& rayDir,
		CUstream stream);
protected:
    void registerPybindModule(pybind11::module& m) override;
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

public:
	enum class LightType
	{
		Point, Directional
	};
private:
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

public:
    [[nodiscard]] bool enableMagnitudeScaling() const
    {
        return enableMagnitudeScaling_;
    }

    [[nodiscard]] Parameter<double> magnitudeScaling() const
    {
        return magnitudeScaling_;
    }

    [[nodiscard]] bool enablePhong() const
    {
        return enablePhong_;
    }

    [[nodiscard]] Parameter<double> ambient() const
    {
        return ambient_;
    }

    [[nodiscard]] Parameter<double> specular() const
    {
        return specular_;
    }

    [[nodiscard]] Parameter<double> magnitudeCenter() const
    {
        return magnitudeCenter_;
    }

    [[nodiscard]] Parameter<double> magnitudeRadius() const
    {
        return magnitudeRadius_;
    }

    [[nodiscard]] Parameter<int> specularExponent() const
    {
        return specularExponent_;
    }

    [[nodiscard]] bool lightFollowsCamera() const
    {
        return lightFollowsCamera_;
    }

    [[nodiscard]] LightType lightType() const
    {
        return lightType_;
    }

    [[nodiscard]] Parameter<double3> lightPosition() const
    {
        return lightPosition_;
    }

    [[nodiscard]] Parameter<double3> lightDirection() const
    {
        return lightDirection_;
    }

    void setEnableMagnitudeScaling(const bool enable_magnitude_scaling)
    {
        enableMagnitudeScaling_ = enable_magnitude_scaling;
    }

    void setMagnitudeScaling(const Parameter<double>& magnitude_scaling)
    {
        magnitudeScaling_ = magnitude_scaling;
    }

    void setEnablePhong(const bool enable_phong)
    {
        enablePhong_ = enable_phong;
    }

    void setAmbient(const Parameter<double>& ambient)
    {
        ambient_ = ambient;
    }

    void setSpecular(const Parameter<double>& specular)
    {
        specular_ = specular;
    }

    void setMagnitudeCenter(const Parameter<double>& magnitude_center)
    {
        magnitudeCenter_ = magnitude_center;
    }

    void setMagnitudeRadius(const Parameter<double>& magnitude_radius)
    {
        magnitudeRadius_ = magnitude_radius;
    }

    void setSpecularExponent(const Parameter<int>& specular_exponent)
    {
        specularExponent_ = specular_exponent;
    }

    void setLightFollowsCamera(const bool light_follows_camera)
    {
        lightFollowsCamera_ = light_follows_camera;
    }

    void setLightType(const LightType light_type)
    {
        lightType_ = light_type;
    }

    void setLightPosition(const Parameter<double3>& light_position)
    {
        lightPosition_ = light_position;
    }

    void setLightDirection(const Parameter<double3>& light_direction)
    {
        lightDirection_ = light_direction;
    }
};


END_RENDERER_NAMESPACE
