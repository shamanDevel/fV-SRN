#include "transfer_function_tex2D.h"

renderer::TransferFunctionTex2D::TransferFunctionTex2D()
    : volumeInfoCache_(16)
{
    throw std::logic_error("Not implemented");
}

renderer::TransferFunctionTex2D::~TransferFunctionTex2D()
{
    throw std::logic_error("Not implemented");
}

std::string renderer::TransferFunctionTex2D::getName() const
{
    throw std::logic_error("Not implemented");
}

bool renderer::TransferFunctionTex2D::drawUI(UIStorage_t& storage)
{
    throw std::logic_error("Not implemented");
}

void renderer::TransferFunctionTex2D::load(const nlohmann::json& json, const ILoadingContext* fetcher)
{
    throw std::logic_error("Not implemented");
}

void renderer::TransferFunctionTex2D::save(nlohmann::json& json, const ISavingContext* context) const
{
    throw std::logic_error("Not implemented");
}

void renderer::TransferFunctionTex2D::prepareRendering(GlobalSettings& s) const
{
    ITransferFunction::prepareRendering(s);
}

std::optional<int> renderer::TransferFunctionTex2D::getBatches(const GlobalSettings& s) const
{
    return ITransferFunction::getBatches(s);
}

std::string renderer::TransferFunctionTex2D::getDefines(const GlobalSettings& s) const
{
    return ITransferFunction::getDefines(s);
}

std::vector<std::string> renderer::TransferFunctionTex2D::getIncludeFileNames(const GlobalSettings& s) const
{
    throw std::logic_error("Not implemented");
}

std::string renderer::TransferFunctionTex2D::getConstantDeclarationName(const GlobalSettings& s) const
{
    throw std::logic_error("Not implemented");
}

std::string renderer::TransferFunctionTex2D::getPerThreadType(const GlobalSettings& s) const
{
    throw std::logic_error("Not implemented");
}

void renderer::TransferFunctionTex2D::fillConstantMemoryTF(const GlobalSettings& s, CUdeviceptr ptr, double stepsize,
    CUstream stream)
{
    throw std::logic_error("Not implemented");
}

void renderer::TransferFunctionTex2D::registerPybindModule(pybind11::module& referenceses)
{
    ITransferFunction::registerPybindModule(referenceses);
}

bool renderer::TransferFunctionTex2D::canPaste(std::shared_ptr<ITransferFunction> transfer_function)
{
    return ITransferFunction::canPaste(transfer_function);
}

void renderer::TransferFunctionTex2D::doPaste(std::shared_ptr<ITransferFunction> transfer_function)
{
    ITransferFunction::doPaste(transfer_function);
}

double4 renderer::TransferFunctionTex2D::evaluate(double density) const
{
    throw std::logic_error("Not implemented");
}

double renderer::TransferFunctionTex2D::getMaxAbsorption() const
{
    throw std::logic_error("Not implemented");
}

bool renderer::TransferFunctionTex2D::requiresGradients() const
{
    return ITransferFunction::requiresGradients();
}

renderer::TransferFunctionTex2D::VolumeCache::VolumeCache(IVolumeInterpolation_ptr volume)
{
    throw std::logic_error("Not implemented");
}

renderer::TransferFunctionTex2D::VolumeCache::~VolumeCache()
{
    throw std::logic_error("Not implemented");
}
