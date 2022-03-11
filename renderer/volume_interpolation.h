#pragma once

#include "imodule.h"
#include "helper_math.cuh"
#include "transfer_function.h"

BEGIN_RENDERER_NAMESPACE

/**
 * Defines the volume interpolation.
 * The volume interpolation is called in the following way:
 * <pre>
 * using density_t = Volume::density_t; //typically real_t
 * const auto result = volume.eval(position, direction, batch);
 * density_t density = result.value; bool isInside = result.isInside;
 * real3 normal = volume.evalNormal(position, direction, result, batch);
 * real2 curvature = volume.evalCurvature(position, direction, result, batch);
 * real3 boxMin = volume.getBoxMin();
 * real3 boxSize = volume.getBoxSize();
 * </pre>
 * where position is either
 *  - in world space between \ref getBoxMin() and \ref getBoxMax()
 *  - in object space between [0,0,0] and \ref getObjectResolution(),
 *    if \ref supportsObjectSpaceIndexing() is true.
 * This is decided via \ref IKernelModule::GlobalSettings::interpolationInObjectSpace.
 *
 * Note that the returned normal is just the gradient of the density.
 * It is <b>not</b> normalized.
 */
class IVolumeInterpolation : public IKernelModule
{
protected:
    /**
     * if object-space indexing is supported,
     * i.e. objectResolution_ contains sensible data.
     */
    const bool supportsObjectSpaceIndexing_;
    //The voxel resolution
    int3 objectResolution_;

    //Bounding box
    double3 boxMin_, boxMax_;

    IVolumeInterpolation(bool supportsObjectSpaceIndexing)
        : supportsObjectSpaceIndexing_(supportsObjectSpaceIndexing)
    {}

public:
    static constexpr std::string_view TAG = "volume";
    std::string getTag() const override { return std::string(TAG); }
    static std::string Tag() { return std::string(TAG); }

    [[nodiscard]] virtual bool supportsObjectSpaceIndexing() const
    {
        return supportsObjectSpaceIndexing_;
    }

    [[nodiscard]] virtual int3 objectResolution() const
    {
        return objectResolution_;
    }

    [[nodiscard]] virtual double3 boxMin() const
    {
        return boxMin_;
    }

    [[nodiscard]] virtual double3 boxMax() const
    {
        return boxMax_;
    }

    [[nodiscard]] virtual double3 boxSize() const;
    [[nodiscard]] virtual double3 voxelSize() const;

    /**
     * Does the volume interpolation allows for curvature estimation?
     */
    [[nodiscard]] virtual bool supportsCurvatureEstimation() const
    {
        return false;
    }

    static const int OutputType2ChannelCount[3];
    /**
     * The output type of the currently selected volume.
     * Might differ from the one requested in the GlobalSettings
     */
    [[nodiscard]] virtual GlobalSettings::VolumeOutput outputType() const = 0;
    /**
     * \brief Returns the output channels of the volume interpolation.
     * Possible return values:
     * 1 -> density
     * 3 -> velocity xyz
     * 4 -> color rgbo
     */
    [[nodiscard]] virtual int outputChannels() const
    {
        return OutputType2ChannelCount[int(outputType())];
    }



    /**
     * Evaluates the values at the given positions in world space.
     * \param positions Tensor of shape (N,3) with the positions in [0,1]^3
     * \param directions Tensor of shape (N,3) with the directions in [0,1]^3.
     *   Can be a undefined tensor, in that case, the direction (0,0,0) is used.
     * \return the interpolated densities at that positions as a tensor of shape (N,C)
     *   with C=outputChannels()
     */
    [[nodiscard]] virtual torch::Tensor evaluate(
        const torch::Tensor& positions,
        const torch::Tensor& directions,
        CUstream stream);

    /**
     * Evaluates the density values and gradient at the given positions in world space.
     * If the volume does not represent a scalar field, i.e.
     * \ref outputType()!=VolumeOutput::Density, then an error is thrown.
     *
     * \param positions Tensor of shape (N,3) with the positions in [0,1]^3
     * \param directions Tensor of shape (N,3) with the directions in [0,1]^3.
     *   Can be a undefined tensor, in that case, the direction (0,0,0) is used.
     * \return a tuple:
     *  - the interpolated densities at that positions as a tensor of shape (N,1)
     *	- the gradient of the density field at that positions as a tensor of shape (N,3)
     */
    [[nodiscard]] virtual std::tuple<torch::Tensor, torch::Tensor> evaluateWithGradient(
        const torch::Tensor& positions,
        const torch::Tensor& directions,
        CUstream stream);

    /**
     * Evaluates the density values and gradient at the given positions in world space.
     * If the volume does not represent a scalar field, i.e.
     * \ref outputType()!=VolumeOutput::Density, then an error is thrown.
     *
     * \param positions Tensor of shape (N,3) with the positions in [0,1]^3
     * \param directions Tensor of shape (N,3) with the directions in [0,1]^3.
     *   Can be a undefined tensor, in that case, the direction (0,0,0) is used.
     * \return a tuple:
     *  - the interpolated densities at that positions as a tensor of shape (N,1)
     *	- the gradient of the density field at that positions as a tensor of shape (N,3)
     *	- the first and second principal curvature of the density field at that positions as a tensor of shape (N,2)
     */
    [[nodiscard]] virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
            evaluateWithGradientAndCurvature(
        const torch::Tensor& positions,
        const torch::Tensor& directions,
        CUstream stream);

    /**
     * \brief Performs importance sampling on the volume.
     * If the transfer function is null, the density values are used (and returned).
     * If the transfer function is defined, the alpha values are used and the color is returned.
     * \param numSamples the number of samples to draw
     * \param tf the transfer function, can be null
     * \param minProb the minimal probability of a sample in [0,1]
     * \param seed the seed for the rng
     * \param time the time increment for the rng
     * \param densityMin the minimal density for TF mappign
     * \param densityMax the maximal density for TF mapping
     * \param dtype the scalar type (float or double) of the output
     * \param stream the CUDA stream to use
     * \return a tuple (positions, densities, colors) where
     *	- positions is a tensor of shape (N,3) of xyz-positions in [0,1]^3, N=numSamples
     *	- densities is a tensor of shape (N,1) of the densities
     *	- colors is a tensor of shape (N,4) of the colors, but only defined if tf!=null
     */
    [[nodiscard]] virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        importanceSampling(int numSamples, ITransferFunction_ptr tf, double minProb, 
            int seed, int time, double densityMin, double densityMax,
            c10::ScalarType dtype, CUstream stream);

    /**
     * \brief Performs importance sampling on the volume,
     * using a separate grid for the probabilities.
     * If the transfer function is null, the density values are used (and returned).
     * If the transfer function is defined, the color is returned.
     * The dtype is inferred from \c probabilityGrid.
     * \param numSamples the number of samples to draw
     * \param tf the transfer function, can be null
     * \param probabilityGrid the grid of shape (X,Y,Z) with the probabilities to use
     * \param maxProbability the maximal value in probabilityGrid for normalization
     * \param minProb the minimal probability of a sample in [0,1]
     * \param seed the seed for the rng
     * \param time the time increment for the rng
     * \param densityMin the minimal density for TF mappign
     * \param densityMax the maximal density for TF mapping
     * \param stream the CUDA stream to use
     * \return a tuple (positions, densities, colors) where
     *	- positions is a tensor of shape (N,3) of xyz-positions in [0,1]^3, N=numSamples
     *	- densities is a tensor of shape (N,1) of the densities
     *	- colors is a tensor of shape (N,4) of the colors, but only defined if tf!=null
     */
    [[nodiscard]] virtual std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
        importanceSamplingWithProbabilityGrid(int numSamples, ITransferFunction_ptr tf, 
            torch::Tensor probabilityGrid, double maxProbability, double minProb,
            int seed, int time, double densityMin, double densityMax,
            CUstream stream);

protected:
    virtual void setObjectResolution(const int3& objectResolution)
    {
        objectResolution_ = objectResolution;
    }

public:
    virtual void setBoxMin(const double3& boxMin)
    {
        boxMin_ = boxMin;
    }
    
    virtual void setBoxMax(const double3& boxMax)
    {
        boxMax_ = boxMax;
    }
protected:
    void registerPybindModule(pybind11::module& m) override;
};
typedef std::shared_ptr<IVolumeInterpolation> IVolumeInterpolation_ptr;

END_RENDERER_NAMESPACE
