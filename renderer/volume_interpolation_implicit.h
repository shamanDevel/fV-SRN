#pragma once

#include <unordered_map>
#include <functional>
#include <json.hpp>

#include "volume_interpolation.h"
#include "commons.h"
#include "volume.h"

BEGIN_RENDERER_NAMESPACE

/**
 * Defines an implicit volume that can be evaluated at any point in space and time.
 *
 * The volume is specified by a function with the following syntax:
 * <code>
 * auto implicitVolume(const float x, const float y, const float z, const float t)
 * {
 *     const auto p = ....; //the parameter struct with members given by the keys of 'parameters'
 *     //BEGIN 'codeForward' that is passed in the constructor
 *     return 1-sqrt(x*x+y*y+z*z); //example
 *     //END 'code'
 * }
 * </code>
 * The returned datatype is either a single float, a float3 for vector fields or
 * a float4 (x,y,z,density) for combined vector-scalar fields.
 *
 * Furthermore, a gradient function for scalar fields can be specified:
 * <code>
 * float3 gradient(const float x, const float y, const float z, const float t)
 * {
 *     const auto p = ....; //the parameter struct with members given by the keys of 'parameters'
 *     //BEGIN 'codeGradient' that is passed in the constructor
 *     float a = sqrtf(x*x+y*y+z*z); //example
 *     float denum = a<1e-6 ? 1 : -1.0f/a;
 *     return make_float3(x*denum, y*denum, z*denum);
 *     //END 'code'
 * }
 * </code>
 * If no such code is passed, automatic differentiation is used.
 * This requires, that all intermediate variables in the forward code
 * are declared as \c auto.
 *
 * Available helper functions for the code:
 * <code>
 *  static inline float sqr(float s) { return s * s; }
 *  static inline float cb(float s) { return s * s * s; }
 *  static inline float implicit2Density(float i) { return std::max(0.0f, std::min(1.0f, -i + 0.5f)); }
 *  M_PI
 * </code>
 *
 * X,Y,Z values are in the range (-boxSize/2, +boxSize/2)
 */
class ImplicitVolume
{
public:
    enum class DataType
    {
        FLOAT,
        FLOAT3,
        FLOAT4
    };

private:
    const std::string name_;
    const DataType dataType_;
    const float valueMin_;
    const float valueMax_;
    const float boxSize_;
    const std::string codeForward_;
    const std::string codeGradient_;

    std::unordered_map<std::string, float> parameters_;

public:
    ImplicitVolume(const std::string& name, DataType data_type, float value_min, float value_max,
        float box_size, const std::string& codeForward, const std::string& codeGradient, const std::unordered_map<std::string, float>& parameters)
        : name_(name),
          dataType_(data_type),
          valueMin_(value_min),
          valueMax_(value_max),
          boxSize_(box_size),
          codeForward_(codeForward),
          codeGradient_(codeGradient),
          parameters_(parameters)
    {
    }

    /**
     * Loads the volume from the JSON setting
     */
    ImplicitVolume(const nlohmann::json& json);

    [[nodiscard]] std::string name() const
    {
        return name_;
    }

    [[nodiscard]] DataType dataType() const
    {
        return dataType_;
    }

    [[nodiscard]] float valueMin() const
    {
        return valueMin_;
    }

    [[nodiscard]] float valueMax() const
    {
        return valueMax_;
    }

    [[nodiscard]] float boxSize() const
    {
        return boxSize_;
    }

    [[nodiscard]] std::string codeForward() const
    {
        return codeForward_;
    }

    [[nodiscard]] std::string codeGradient() const
    {
        return codeGradient_;
    }

    [[nodiscard]] bool hasGradientCode() const
    {
        return !codeGradient_.empty();
    }

    /**
     * Does the volume have extra parameters that can be modified?
     */
    [[nodiscard]] bool hasParameters() const
    {
        return !parameters_.empty();
    }

    [[nodiscard]] const std::unordered_map<std::string, float>& parameters() const
    {
        return parameters_;
    }
    [[nodiscard]] std::unordered_map<std::string, float>& parameters()
    {
        return parameters_;
    }

    [[nodiscard]] std::vector<std::string> parameterKeys() const;

    [[nodiscard]] float getParameter(const std::string& key) const;

    void setParameter(const std::string& key, float value);

    /**
     * Clones this implicit function.
     * Only useful if there are parameters that can be modified.
     */
    [[nodiscard]] std::shared_ptr<ImplicitVolume> clone() const;
};
typedef std::shared_ptr<ImplicitVolume> ImplicitVolume_ptr;


class VolumeInterpolationImplicit : public IVolumeInterpolation
{
    static std::vector<std::string> ImplicitFunctionNames_;
    static std::vector<ImplicitVolume_ptr> DefaultImplicitFunctions_;
    
    static void InitImplicitFunctions();

    //The stepsize for the hessian estimation (central differences on the gradients)
    const float hessianStepsize_;

    int selectedImplicitFunctionIndex_; //UI
    ImplicitVolume_ptr selectedImplicitFunction_;

public:
    VolumeInterpolationImplicit();
    void setSource(ImplicitVolume_ptr v);
    [[nodiscard]] ImplicitVolume_ptr getSource() const { return selectedImplicitFunction_; }

    static const std::vector<ImplicitVolume_ptr>& DefaultImplicitFunctions()
    {
        return DefaultImplicitFunctions_;
    }

    [[nodiscard]] bool supportsCurvatureEstimation() const override { return true; }
    [[nodiscard]] std::string getName() const override;
    [[nodiscard]] bool drawUI(UIStorage_t& storage) override;
    void load(const nlohmann::json& json, const ILoadingContext* fetcher) override;
    void save(nlohmann::json& json, const ISavingContext* context) const override;
    void prepareRendering(GlobalSettings& s) const override;
    [[nodiscard]] std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
    [[nodiscard]] std::string getConstantDeclarationName(const GlobalSettings& s) const override;
    [[nodiscard]] std::string getPerThreadType(const GlobalSettings& s) const override;
    void fillExtraSourceCode(const GlobalSettings& s, std::stringstream& ss) const override;
    void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;
    [[nodiscard]] GlobalSettings::VolumeOutput outputType() const override;
protected:
    void registerPybindModule(pybind11::module& m) override;
};
typedef std::shared_ptr<VolumeInterpolationImplicit> VolumeInterpolationImplicit_ptr;


END_RENDERER_NAMESPACE
