#include "volume_interpolation_implicit.h"

#include <magic_enum.hpp>

#include <cmrc/cmrc.hpp>

#include "volume_interpolation_grid.h"
CMRC_DECLARE(shaders);

renderer::ImplicitVolume::ImplicitVolume(const nlohmann::json& json)
    : name_(json.at("name")),
      dataType_(magic_enum::enum_cast<DataType>(json.at("dtype").get<std::string>()).value()),
      valueMin_(json.at("valuemin")),
      valueMax_(json.at("valuemax")),
      boxSize_(json.at("boxsize")),
      codeForward_(json.at("codeForward")),
      codeGradient_(json.value("codeGradient", ""))
{
    auto p = json.at("parameters");
    for (auto& [key, value] : p.items())
    {
        parameters_.emplace(key, value.get<float>());
    }
}

std::vector<std::string> renderer::ImplicitVolume::parameterKeys() const
{
    std::vector<std::string> keys;
    keys.reserve(parameters_.size());
    for (const auto& e : parameters_)
        keys.push_back(e.first);
    return keys;
}

float renderer::ImplicitVolume::getParameter(const std::string& key) const
{
    const auto it = parameters_.find(key);
    if (it == parameters_.cend())
        throw std::runtime_error("Key not found! " + key);
    return it->second;
}

void renderer::ImplicitVolume::setParameter(const std::string& key, float value)
{
    auto it = parameters_.find(key);
    if (it == parameters_.end()) {
        throw std::runtime_error("Key not found! " + key);
    }
    it->second = value;
}

std::shared_ptr<renderer::ImplicitVolume> renderer::ImplicitVolume::clone() const
{
    auto v = std::make_shared<ImplicitVolume>(*this);
    return v;
}


std::vector<std::string> renderer::VolumeInterpolationImplicit::ImplicitFunctionNames_;
std::vector<renderer::ImplicitVolume_ptr> renderer::VolumeInterpolationImplicit::DefaultImplicitFunctions_;

void renderer::VolumeInterpolationImplicit::InitImplicitFunctions()
{
    static bool initialized = false;
    if (initialized) return;
    initialized = true;

    try {
        auto fs = cmrc::shaders::get_filesystem();
        auto specification = fs.open("shaders/VolumeImplicit.json");
        auto j = nlohmann::json::parse(specification.begin(), specification.end());
        for (auto obj : j)
        {
            auto v = std::make_shared<ImplicitVolume>(obj);
            ImplicitFunctionNames_.push_back(v->name());
            DefaultImplicitFunctions_.push_back(v);
        }
    } catch (const std::exception& ex)
    {
        std::cerr << "Error while reading implicit functions:\n" << ex.what();
        throw;
    }

    if (ImplicitFunctionNames_.empty())
        throw std::runtime_error("No implicit functions defined!!");
    else
        std::cout << ImplicitFunctionNames_.size() << " implicit functions loaded" << std::endl;
}

renderer::VolumeInterpolationImplicit::VolumeInterpolationImplicit()
    : IVolumeInterpolation(false)
    , hessianStepsize_(1/256.f)
{
    InitImplicitFunctions();

    selectedImplicitFunctionIndex_ = 0;
    selectedImplicitFunction_ = DefaultImplicitFunctions_[selectedImplicitFunctionIndex_];

    //unit cube
    setBoxMin(make_double3(-0.5));
    setBoxMax(make_double3(+0.5));
}

void renderer::VolumeInterpolationImplicit::setSource(ImplicitVolume_ptr v)
{
    if (!v)
        throw std::runtime_error("Attempt to set null implicit volume");
    selectedImplicitFunction_ = v;
}

std::string renderer::VolumeInterpolationImplicit::getName() const
{
    return "Implicit";
}

bool renderer::VolumeInterpolationImplicit::drawUI(UIStorage_t& storage)
{
    bool changed = false;

    //select implicit function
    if (ImGui::BeginCombo("Function##VolumeInterpolationImplicit",
        selectedImplicitFunction_->name().c_str()))
    {
        for (int i=0; i< ImplicitFunctionNames_.size(); ++i)
        {
            bool isSelected = i == selectedImplicitFunctionIndex_;
            if (ImGui::Selectable(ImplicitFunctionNames_[i].c_str(), isSelected))
            {
                selectedImplicitFunctionIndex_ = i;
                setSource(DefaultImplicitFunctions_[i]);
                changed = true;
            }
            if (isSelected)
                ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    //parameters
    if (selectedImplicitFunction_->hasParameters())
    {
        ImGui::PushID("Parameters##VolumeInterpolationImplicit");
        for (auto& [key,value] : selectedImplicitFunction_->parameters())
        {
            if (ImGui::InputFloat(key.c_str(), &value)) {
                changed = true;
            }
        }
        ImGui::PopID();
    }

    //infos
    ImGui::Text("Value range: [%.2f, %.2f]\nBox size: %.2f",
        selectedImplicitFunction_->valueMin(),
        selectedImplicitFunction_->valueMax(),
        selectedImplicitFunction_->boxSize());

    //UI Storage
    storage[VolumeInterpolationGrid::UI_KEY_HISTOGRAM] = nullptr;
    storage[VolumeInterpolationGrid::UI_KEY_MIN_DENSITY] = static_cast<float>(selectedImplicitFunction_->valueMin());
    storage[VolumeInterpolationGrid::UI_KEY_MAX_DENSITY] = static_cast<float>(selectedImplicitFunction_->valueMax());

    return changed;
}

void renderer::VolumeInterpolationImplicit::load(const nlohmann::json& json, const ILoadingContext* fetcher)
{
    std::string functionName = json.value("function", "");
    for (int i=0; i<ImplicitFunctionNames_.size(); ++i)
    {
        if (ImplicitFunctionNames_[i] == functionName)
        {
            selectedImplicitFunctionIndex_ = i;
            setSource(DefaultImplicitFunctions_[i]);
            for (auto& [key, value] : selectedImplicitFunction_->parameters())
            {
                if (auto it = json.find(key); it != json.end())
                {
                    value = it.value().get<float>();
                }
            }
        }
    }
}

void renderer::VolumeInterpolationImplicit::save(nlohmann::json& json, const ISavingContext* context) const
{
    for (const auto& [key, value] : selectedImplicitFunction_->parameters())
    {
        json[key] = value;
    }
    json["function"] = selectedImplicitFunction_->name();
}

void renderer::VolumeInterpolationImplicit::prepareRendering(GlobalSettings& s) const
{
    IVolumeInterpolation::prepareRendering(s);
}

std::vector<std::string> renderer::VolumeInterpolationImplicit::getIncludeFileNames(const GlobalSettings& s) const
{
    return { "renderer_volume_implicit.cuh" };
}

std::string renderer::VolumeInterpolationImplicit::getConstantDeclarationName(const GlobalSettings& s) const
{
    return "volumeInterpolationImplicitParameters";
}

std::string renderer::VolumeInterpolationImplicit::getPerThreadType(const GlobalSettings& s) const
{
    return "::kernel::VolumeInterpolationImplicit";
}

void renderer::VolumeInterpolationImplicit::fillExtraSourceCode(const GlobalSettings& s, std::stringstream& ss) const
{
    IVolumeInterpolation::fillExtraSourceCode(s, ss);

    //code generation
    ss << R"(
#include "helper_math.cuh"
#include <forward.h>
#include <forward_float.h>

namespace kernel
{
    struct VolumeInterpolationImplicitParameters
    {
        float4 sourceBoxMin;
        float4 sourceBoxSize;
        float targetBoxMin;
        float targetBoxSize;
        float hessianStepsize;
    )";
    for (const auto& [key, value] : selectedImplicitFunction_->parameters())
    {
        ss << "\t\tfloat " << key << ";\n";
    }
    ss << R"(
    };
}
__constant__ kernel::VolumeInterpolationImplicitParameters volumeInterpolationImplicitParameters;

namespace kernel
{
    template<typename T> __host__ __device__ __forceinline__ T sqr(T s) { return s * s; }
    template<typename T> __host__ __device__ __forceinline__ T cb(T s) { return s * s * s; }
    template<typename T> __host__ __device__ __forceinline__ T implicit2Density(T i) {
        //return fmaxf(0.0f, fminf(1.0f, 0.5f - i));
        return 0.5f - i;
    }

    __host__ __device__ __forceinline__
    float VolumeInterpolationImplicit_forward(const float x, const float y, const float z, const float t)
    {
        const auto& p = volumeInterpolationImplicitParameters;
        using namespace std;
    )";
    ss << selectedImplicitFunction_->codeForward();
    ss << R"(
    }
    )";

    if (s.volumeShouldProvideNormals)
    {
        if (selectedImplicitFunction_->hasGradientCode())
        {
            ss << R"(
    __host__ __device__ __forceinline__
    float3 VolumeInterpolationImplicit_gradient(const float x, const float y, const float z, const float t)
    {
        const auto& p = volumeInterpolationImplicitParameters;
        using namespace std;
            )";
            ss << selectedImplicitFunction_->codeGradient();
            ss << "\t}\n";
        }
        else
        {
            //use AD

            ss << R"(
    __host__ __device__ __forceinline__
    CUDAD_NAMESPACE fvar<float, 3> VolumeInterpolationImplicit_gradient_helper(
        const CUDAD_NAMESPACE fvar<float, 3>& x, const CUDAD_NAMESPACE fvar<float, 3>& y,
        const CUDAD_NAMESPACE fvar<float, 3>& z, const float t)
    {
        const auto& p = volumeInterpolationImplicitParameters;
        using namespace std;
            )";
            ss << selectedImplicitFunction_->codeForward();
            ss << R"(
    }
    __host__ __device__ __forceinline__
    float3 VolumeInterpolationImplicit_gradient(const float x, const float y, const float z, const float t)
    {
        auto g = VolumeInterpolationImplicit_gradient_helper(
            CUDAD_NAMESPACE fvar<float, 3>::input<0>(x),
            CUDAD_NAMESPACE fvar<float, 3>::input<1>(y),
            CUDAD_NAMESPACE fvar<float, 3>::input<2>(z),
            t);
        return make_float3(g.derivative<0>(), g.derivative<1>(), g.derivative<2>());
    }
            )";
        }
    } else
    {
        ss << R"(
    __host__ __device__ __forceinline__
    float3 VolumeInterpolationImplicit_gradient(const float x, const float y, const float z, const float t)
    { 
        return make_float3(0,0,0);
    }
            )";
    }

    ss << R"(
    }
#define VOLUME_INTERPOLATION_IMPLICIT__CODE_GENERATION 1
    )";
}

void renderer::VolumeInterpolationImplicit::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr,
                                                               CUstream stream)
{
    std::vector<float> mem;
    mem.push_back(static_cast<float>(boxMin().x));
    mem.push_back(static_cast<float>(boxMin().y));
    mem.push_back(static_cast<float>(boxMin().z));
    mem.push_back(0.0f);
    mem.push_back(static_cast<float>(boxSize().x));
    mem.push_back(static_cast<float>(boxSize().y));
    mem.push_back(static_cast<float>(boxSize().z));
    mem.push_back(0.0f);
    mem.push_back(-selectedImplicitFunction_->boxSize() / 2);
    mem.push_back(selectedImplicitFunction_->boxSize());
    mem.push_back(hessianStepsize_);
    for (const auto& [key, value] : selectedImplicitFunction_->parameters())
    {
        mem.push_back(value);
    }
    CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, mem.data(), sizeof(float)*mem.size(), stream));
}

renderer::IKernelModule::GlobalSettings::VolumeOutput renderer::VolumeInterpolationImplicit::outputType() const
{
    //TODO: support vector types
    return GlobalSettings::Density;
}

void renderer::VolumeInterpolationImplicit::registerPybindModule(pybind11::module& m)
{
    IVolumeInterpolation::registerPybindModule(m);

    //guard double registration
    static bool registered = false;
    if (registered) return;
    registered = true;

    namespace py = pybind11;
    py::class_<VolumeInterpolationImplicit, IVolumeInterpolation, VolumeInterpolationImplicit_ptr> c(m, "VolumeInterpolationImplicit");

    py::enum_<ImplicitVolume::DataType>(c, "DataType")
        .value("FLOAT", ImplicitVolume::DataType::FLOAT)
        .value("FLOAT3", ImplicitVolume::DataType::FLOAT3)
        .value("FLOAT4", ImplicitVolume::DataType::FLOAT4)
    ;

    py::class_<ImplicitVolume, ImplicitVolume_ptr>(c, "ImplicitVolume")
        .def(py::init<const std::string&, ImplicitVolume::DataType, float, float, float, const std::string&, const std::string&,
            const std::unordered_map<std::string, float>&>(), py::doc(R"(
        Defines an implicit volume that can be evaluated at any point in space and time.
        
        The volume is specified by a function with the following syntax:
        <code>
        auto implicitVolume(const float x, const float y, const float z, const float t)
        {
            const auto p = ....; //the parameter struct with members given by the keys of 'parameters'
            //BEGIN 'codeForward' that is passed in the constructor
            return 1-sqrt(x*x+y*y+z*z); //example
            //END 'code'
        }
        </code>
        The returned datatype is either a single float, a float3 for vector fields or
        a float4 (x,y,z,density) for combined vector-scalar fields.
        
        Furthermore, a gradient function for scalar fields can be specified:
        <code>
        float3 gradient(const float x, const float y, const float z, const float t)
        {
            const auto p = ....; //the parameter struct with members given by the keys of 'parameters'
            //BEGIN 'codeGradient' that is passed in the constructor
            float a = sqrtf(x*x+y*y+z*z); //example
            float denum = a<1e-6 ? 1 : -1.0f/a;
            return make_float3(x*denum, y*denum, z*denum);
            //END 'code'
        }
        </code>
        If no such code is passed, automatic differentiation is used.
        This requires, that all intermediate variables in the forward code
        are declared as \c auto.
        
        Available helper functions for the code:
        <code>
         static inline float sqr(float s) { return s * s; }
         static inline float cb(float s) { return s * s * s; }
         static inline float implicit2Density(float i) { return std::max(0.0f, std::min(1.0f, -i + 0.5f)); }
        </code>
        
        X,Y,Z values are in the range (-boxSize/2, +boxSize/2)
        )"),
            py::arg("name"), py::arg("data_type"), py::arg("value_min"), py::arg("value_max"),
            py::arg("box_size"), py::arg("code_forward"), py::arg("code_gradient"),
            py::arg("parameters"))
        .def_property_readonly("name", &ImplicitVolume::name)
        .def_property_readonly("data_type", &ImplicitVolume::dataType)
        .def_property_readonly("value_min", &ImplicitVolume::valueMin)
        .def_property_readonly("value_max", &ImplicitVolume::valueMax)
        .def_property_readonly("box_size", &ImplicitVolume::boxSize)
        .def_property_readonly("code_forward", &ImplicitVolume::codeForward)
        .def_property_readonly("code_gradient", &ImplicitVolume::codeGradient)
        .def_property_readonly("has_gradient_code", &ImplicitVolume::hasGradientCode)
        .def_property_readonly("has_parameters", &ImplicitVolume::hasParameters)
        .def("parameterKeys", &ImplicitVolume::parameterKeys)
        .def("getParameter", &ImplicitVolume::getParameter)
        .def("setParameter", &ImplicitVolume::setParameter)
        .def("clone", &ImplicitVolume::clone)
        ;

    c.def(py::init<>())
        .def("set_source", &VolumeInterpolationImplicit::setSource)
        .def("get_source", &VolumeInterpolationImplicit::getSource)
        ;
}
