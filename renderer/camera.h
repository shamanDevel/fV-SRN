#pragma once

#include "imodule.h"
#include "parameter.h"

BEGIN_RENDERER_NAMESPACE

/**
 * Base class of camera implementations:
 * <code>
 * const auto [rayStart, rayDir] camera.eval(real2 screenPos, int batch)
 * </code>
 * The screen position is defined in normalized device coordinates [-1,+1]^2.
 */
class ICamera : public IKernelModule
{
protected:
	//width / height
	double aspectRatio_ = 1.0;

public:
	static constexpr std::string_view TAG = "camera";
	static std::string Tag() { return std::string(TAG); }
	std::string getTag() const override { return Tag(); }


	[[nodiscard]] virtual double aspectRatio() const
	{
		return aspectRatio_;
	}

	virtual void setAspectRatio(const double aspectRatio)
	{
		this->aspectRatio_ = aspectRatio;
	}

	/**
	 * Returns the eye position / camera origin at the specific batch.
	 */
	virtual double3 getOrigin(int batch = 0) const = 0;
	/**
	 * Returns the front vector, i.e. the direction in which the camera faces,
	 * at the specific batch.
	 */
	virtual double3 getFront(int batch = 0) const = 0;

	/**
	 * Generate the rays for the current camera settings.
	 * The batch size is determined by the settings, the width and height by
	 * the argument.
	 *
	 * \param width the width of the image
	 * \param height the height of the image
	 * \param doublePrecision true -> result is of dtype 'double'. False -> 'float'
	 * \param stream the cuda stream
	 * \return a tuple (ray start, ray direction), each of shape B*H*W*3.
	 *   <b>Note that channel is last!</b>
	 *   This is to allow for easy reshaping to (BHW)*3 for scene network
	 *   training
	 */
	virtual std::tuple<torch::Tensor, torch::Tensor> generateRays(
		int width, int height, bool doublePrecision, CUstream stream);

	/**
	 * Generate the rays for the current camera settings.
	 * The batch size is determined by the settings, the width and height by
	 * the argument.
	 *
	 * \param width the width of the image
	 * \param height the height of the image
	 * \param doublePrecision true -> result is of dtype 'double'. False -> 'float'
	 * \param numSamples the number of samples to generate
	 * \param time the time for the random number generator
	 * \param stream the cuda stream
	 * \return a tuple (ray start, ray direction), each of shape B*H*W*3.
	 *   <b>Note that channel is last!</b>
	 *   This is to allow for easy reshaping to (BHW)*3 for scene network
	 *   training
	 */
	virtual std::tuple<torch::Tensor, torch::Tensor> generateRaysMultisampling(
		int width, int height, bool doublePrecision, int numSamples, unsigned int time, CUstream stream);

	/**
	 * Returns the tensor of shape (B,...) that parameterizes the camera.
	 * The shape of the returned tensor must be equal for all instances of the
	 * same subclass of ICamera to allow batches, but might vary for different subclasses.
	 */
	virtual torch::Tensor getParameters() = 0;

	/**
	 * Sets/overwrites the parameters of the camera.
	 * The parameters are of shape (B, ...), see \ref getParameters().
	 * If an empty tensor is passed, the default values from the UI settings
	 * shall be used again.
	 */
	virtual void setParameters(const torch::Tensor& parameters) = 0;

	/**
	 * Computes the opengl matrices (view and projection) for batch zero
	 * of the current camera.
	 */
	virtual void computeOpenGlMatrices(int width, int height, 
		glm::mat4& viewOut, glm::mat4& projectionOut, glm::mat4& normalOut, glm::vec3& originOut) const = 0;

	/**
	 * Converts from world coordinates to normalized screen coordinates
	 */
	virtual std::vector<double3> world2screen(int width, int height, const std::vector<double3>& world) const;

protected:
	void registerPybindModule(pybind11::module& m) override;
};
typedef std::shared_ptr<ICamera> ICamera_ptr;

/**
 * Camera on a sphere around a center facing inward.
 * The pitch, yaw, distance and center is differentiable and can be batched.
 */
class CameraOnASphere : public ICamera
{
public:
	enum Orientation
	{
		Xp, Xm, Yp, Ym, Zp, Zm
	};
	static const char* OrientationNames[6];
	static const float3 OrientationUp[6];
	static const int3 OrientationPermutation[6];
	static const bool OrientationInvertYaw[6];
	static const bool OrientationInvertPitch[6];

protected:
	Orientation orientation_;

	Parameter<double3> center_;
	//(pitch in radians, yaw in radians, distance)
	Parameter<double3> pitchYawDistance_;
	//field of view along Y in radians
	double fovYradians_;

	torch::Tensor cachedCameraMatrix_;
	bool matrixFromExternalSource_;
	
	//UI
	const float baseDistance_ = 1.0f;
	const float rotateSpeed_ = 0.00872665f; // = deg2rad(0.5f);
	const float zoomSpeed_ = 1.1f;
	float zoomValue_ = 0;
	//cached for UI
	double3 cacheOrigin_;
	double3 cacheFront_;

public:
	CameraOnASphere();
	
	[[nodiscard]] virtual Orientation orientation() const
	{
		return orientation_;
	}

	virtual void setOrientation(const Orientation orientation)
	{
		orientation_ = orientation;
	}

	[[nodiscard]] virtual Parameter<double3> center() const
	{
		return center_;
	}
	[[nodiscard]] virtual Parameter<double3>& center()
	{
		return center_;
	}

	[[nodiscard]] virtual Parameter<double3> pitchYawDistance() const
	{
		return pitchYawDistance_;
	}
	[[nodiscard]] virtual Parameter<double3>& pitchYawDistance()
	{
		return pitchYawDistance_;
	}

	[[nodiscard]] virtual double fovYradians() const
	{
		return fovYradians_;
	}

	[[nodiscard]] virtual float zoomValue() const
	{
		return zoomValue_;
	}

	torch::Tensor getParameters() override;
	void setParameters(const torch::Tensor& parameters) override;
	
	std::string getName() const override;
	bool updateUI(UIStorage_t& storage) override;
	bool drawUI(UIStorage_t& storage) override;
	void load(const nlohmann::json& json, const ILoadingContext* context) override;
	void save(nlohmann::json& json, const ISavingContext* context) const override;
protected:
	void registerPybindModule(pybind11::module& m) override;
public:
	std::optional<int> getBatches(const GlobalSettings& s) const override;
	std::vector<std::string> getIncludeFileNames(const GlobalSettings& s) const override;
	std::string getConstantDeclarationName(const GlobalSettings& s) const override;
	std::string getPerThreadType(const GlobalSettings& s) const override;
	void fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream) override;

private:
	void updateCameraMatrix(c10::ScalarType scalarType);
	void computeParameters(
		double3& origin, double3& up, double& distance) const;

public:
	double3 getOrigin(int batch) const override;
	double3 getFront(int batch) const override;
    void computeOpenGlMatrices(int width, int height, 
		glm::mat4& viewOut, glm::mat4& projectionOut, glm::mat4& normalOut, glm::vec3& originOut) const override;

    static double3 eulerToCartesian(
		double pitch, double yaw, double distance,
		Orientation orientation);
};

END_RENDERER_NAMESPACE
