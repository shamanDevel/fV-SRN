#include "volume_interpolation.h"

#include <c10/cuda/CUDAStream.h>
#include <cuMat/src/DevicePointer.h>
#include <cuMat/src/Macros.h>

#include "pytorch_utils.h"
#include "renderer_commons.cuh"
#include "kernel_loader.h"
#include "renderer_tensor.cuh"

const int renderer::IVolumeInterpolation::OutputType2ChannelCount[3] = {
	1, 3, 4
};

double3 renderer::IVolumeInterpolation::boxSize() const
{
	return boxMax_ - boxMin_;
}

double3 renderer::IVolumeInterpolation::voxelSize() const
{
	return boxSize() / make_double3(objectResolution() - make_int3(1));
}

torch::Tensor renderer::IVolumeInterpolation::evaluate(
	const torch::Tensor& positions, const torch::Tensor& direction, CUstream stream)
{
	CHECK_CUDA(positions, true);
	CHECK_DIM(positions, 2);
	CHECK_SIZE(positions, 1, 3);
	bool hasDirection = false;
	if (direction.defined())
	{
		hasDirection = true;
		CHECK_CUDA(direction, true);
		CHECK_DIM(direction, 2);
		CHECK_SIZE(direction, 1, 3);
	}

	GlobalSettings s{};
	s.scalarType = positions.scalar_type();
	s.volumeShouldProvideNormals = false;
	s.interpolationInObjectSpace = false;
	const auto oldBoxMax = boxMax();
	const auto oldBoxMin = boxMin();
	setBoxMin(make_double3(0, 0, 0));
	setBoxMax(make_double3(1, 1, 1));
	int channels = outputChannels();

	//kernel
	this->prepareRendering(s);
	const std::string kernelName = "EvaluateNoBatches";
	std::vector<std::string> constantNames;
	if (const auto c = getConstantDeclarationName(s); !c.empty())
		constantNames.push_back(c);
	std::stringstream extraSource;
	extraSource << "#define KERNEL_DOUBLE_PRECISION "
		<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
	extraSource << "#define KERNEL_SYNCHRONIZED_TRACING "
		<< (s.synchronizedThreads ? 1 : 0)
		<< "\n";
	extraSource << getDefines(s) << "\n";
	for (const auto& i : getIncludeFileNames(s))
		extraSource << "#include \"" << i << "\"\n";
	extraSource << "#define VOLUME_INTERPOLATION_T " <<
		getPerThreadType(s) << "\n";
	extraSource << "#define OUTPUT_CHANNELS " << channels << "\n";
	extraSource << "#define VOLUME_USE_DIRECTION " << (hasDirection ? 1 : 0) << "\n";
	extraSource << "#include \"renderer_volume_kernels1.cuh\"\n";
	const auto fun0 = KernelLoader::Instance().getKernelFunction(
		kernelName, extraSource.str(), constantNames, false, false);
	if (!fun0.has_value())
		throw std::runtime_error("Unable to compile kernel");
	const auto fun = fun0.value();
	if (auto c = getConstantDeclarationName(s); !c.empty())
	{
		CUdeviceptr ptr = fun.constant(c);
		fillConstantMemory(s, ptr, stream);
	}

	//output tensors
	int batches = positions.size(0);
	auto densities = torch::empty({ batches, channels },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));

	//launch kernel
	int blockSize;
	if (s.fixedBlockSize>0)
	{
		if (s.fixedBlockSize > fun.bestBlockSize())
			throw std::runtime_error("larger block size requested that can be fullfilled");
		blockSize = s.fixedBlockSize;
	} else
	{
		blockSize = fun.bestBlockSize();
	}
	int minGridSize = std::min(
		int(CUMAT_DIV_UP(batches, blockSize)),
		fun.minGridSize());
	dim3 virtual_size{
		static_cast<unsigned int>(batches), 1, 1 };
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "IVolumeInterpolation::evaluate", [&]()
		{
			const auto accPosition = accessor< ::kernel::Tensor2Read<scalar_t>>(positions);
			const auto accDirection = hasDirection
				? accessor< ::kernel::Tensor2Read<scalar_t>>(direction)
				: ::kernel::Tensor2Read<scalar_t>();
			const auto accDensity = accessor< ::kernel::Tensor2RW<scalar_t>>(densities);
			const void* args[] = { &virtual_size, &accPosition, &accDirection, &accDensity };
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, blockSize, 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return printError(result, kernelName);
			return true;
		});

	setBoxMin(oldBoxMin);
	setBoxMax(oldBoxMax);
	
	if (!success) throw std::runtime_error("Error during rendering!");
	
	return densities;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> renderer::IVolumeInterpolation::importanceSampling(
	int numSamples, ITransferFunction_ptr tf,
    double minProb, int seed, int time,
	double densityMin, double densityMax,
	c10::ScalarType dtype, CUstream stream)
{
	TORCH_CHECK(numSamples > 0, "numSamples must be positive");
	TORCH_CHECK(dtype == c10::kFloat || dtype == c10::kDouble, "dtype must be float or double");
    double maxValue = tf == nullptr ? 1 : tf->getMaxAbsorption();

	GlobalSettings s{};
	s.scalarType = dtype;
	s.volumeShouldProvideNormals = false;
	s.interpolationInObjectSpace = false;
	const auto oldBoxMax = boxMax();
	const auto oldBoxMin = boxMin();
	setBoxMin(make_double3(0, 0, 0));
	setBoxMax(make_double3(1, 1, 1));

	//kernel (volume)
	this->prepareRendering(s);
	const std::string kernelName = "ImportanceSampling";
	std::vector<std::string> constantNames;
	if (const auto c = getConstantDeclarationName(s); !c.empty())
		constantNames.push_back(c);
	std::stringstream extraSource;
	extraSource << "#define KERNEL_DOUBLE_PRECISION "
		<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
	extraSource << "#define KERNEL_SYNCHRONIZED_TRACING "
		<< (s.synchronizedThreads ? 1 : 0)
		<< "\n";
	extraSource << getDefines(s) << "\n";
	for (const auto& i : getIncludeFileNames(s))
		extraSource << "#include \"" << i << "\"\n";
	extraSource << "#define VOLUME_INTERPOLATION_T " <<
		getPerThreadType(s) << "\n";

	//kernel (TF)
	if (tf)
	{
		tf->prepareRendering(s);
		if (const auto c = tf->getConstantDeclarationName(s); !c.empty())
			constantNames.push_back(c);
		extraSource << tf->getDefines(s) << "\n";
		for (const auto& i : tf->getIncludeFileNames(s))
			extraSource << "#include \"" << i << "\"\n";
		extraSource << "#define TRANSFER_FUNCTION_T " <<
			tf->getPerThreadType(s) << "\n";
		extraSource << "#define HAS_TRANSFER_FUNCTION 1\n";
	}
	else
	{
		extraSource << "#define HAS_TRANSFER_FUNCTION 0\n";
	}

	//compile and fill constants
	extraSource << "#include \"renderer_volume_kernels2.cuh\"\n";
	const auto& fun = KernelLoader::Instance().getKernelFunction(
		kernelName, extraSource.str(), constantNames, false, false).value();
	if (auto c = getConstantDeclarationName(s); !c.empty())
	{
		CUdeviceptr ptr = fun.constant(c);
		fillConstantMemory(s, ptr, stream);
	}
	if (tf)
	{
		if (auto c = tf->getConstantDeclarationName(s); !c.empty())
		{
			CUdeviceptr ptr = fun.constant(c);
			tf->fillConstantMemory(s, ptr, stream);
		}
	}

	//output tensors
	auto positions = torch::empty({numSamples, 3}, 
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	auto densities = torch::empty({ numSamples, 1 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	torch::Tensor colors;
	if (tf)
		colors = torch::empty({ numSamples, 4 },
			at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	cuMat::DevicePointer<unsigned int> counter(1);
	CUMAT_SAFE_CALL(cudaMemset(counter.pointer(), 0, sizeof(unsigned int)));
	unsigned int* counterPtr = counter.pointer();

	//launch kernel
	int minGridSize = fun.minGridSize();
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "IVolumeInterpolation::importanceSampling", [&]()
		{
		    //cast variables
			unsigned int numSamples_u = static_cast<unsigned int>(numSamples);
			const auto accPosition = accessor< ::kernel::Tensor2RW<scalar_t>>(positions);
			const auto accDensities = accessor< ::kernel::Tensor2RW<scalar_t>>(densities);
			const auto accColors = tf ? accessor< ::kernel::Tensor2RW<scalar_t>>(colors) : ::kernel::Tensor2RW<scalar_t>();
			scalar_t maxValue_r = static_cast<scalar_t>(maxValue);
			scalar_t minProb_r = static_cast<scalar_t>(minProb);
			scalar_t densityMin_r = static_cast<scalar_t>(densityMin);
			scalar_t densityMax_r = static_cast<scalar_t>(densityMax);
		    
			const void* args[] = {
			    &numSamples_u, &accPosition, &accDensities, &accColors,
			    &maxValue_r, &minProb_r, &counterPtr,
			    &seed, &time, &densityMin_r, &densityMax_r};
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, fun.bestBlockSize(), 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return printError(result, kernelName);
			return true;
		});

	setBoxMin(oldBoxMin);
	setBoxMax(oldBoxMax);

	if (!success) throw std::runtime_error("Error during rendering!");

	return {positions, densities, colors};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> renderer::IVolumeInterpolation::importanceSamplingWithProbabilityGrid(
	int numSamples, ITransferFunction_ptr tf,
	torch::Tensor probabilityGrid, double maxProbability, 
	double minProb, int seed, int time,
	double densityMin, double densityMax,
	CUstream stream)
{
	CHECK_DIM(probabilityGrid, 3);
	CHECK_CUDA(probabilityGrid, true);
	c10::ScalarType dtype = probabilityGrid.scalar_type();

	TORCH_CHECK(numSamples > 0, "numSamples must be positive");
	TORCH_CHECK(dtype == c10::kFloat || dtype == c10::kDouble, "dtype must be float or double");

	GlobalSettings s{};
	s.scalarType = dtype;
	s.volumeShouldProvideNormals = false;
	s.interpolationInObjectSpace = false;
	const auto oldBoxMax = boxMax();
	const auto oldBoxMin = boxMin();
	setBoxMin(make_double3(0, 0, 0));
	setBoxMax(make_double3(1, 1, 1));

	//kernel (volume)
	this->prepareRendering(s);
	const std::string kernelName = "ImportanceSamplingWithProbabilityGrid";
	std::vector<std::string> constantNames;
	if (const auto c = getConstantDeclarationName(s); !c.empty())
		constantNames.push_back(c);
	std::stringstream extraSource;
	extraSource << "#define KERNEL_DOUBLE_PRECISION "
		<< (s.scalarType == GlobalSettings::kDouble ? 1 : 0)
		<< "\n";
	extraSource << "#define KERNEL_SYNCHRONIZED_TRACING "
		<< (s.synchronizedThreads ? 1 : 0)
		<< "\n";
	extraSource << getDefines(s) << "\n";
	for (const auto& i : getIncludeFileNames(s))
		extraSource << "#include \"" << i << "\"\n";
	extraSource << "#define VOLUME_INTERPOLATION_T " <<
		getPerThreadType(s) << "\n";

	//kernel (TF)
	if (tf)
	{
		tf->prepareRendering(s);
		if (const auto c = tf->getConstantDeclarationName(s); !c.empty())
			constantNames.push_back(c);
		extraSource << tf->getDefines(s) << "\n";
		for (const auto& i : tf->getIncludeFileNames(s))
			extraSource << "#include \"" << i << "\"\n";
		extraSource << "#define TRANSFER_FUNCTION_T " <<
			tf->getPerThreadType(s) << "\n";
		extraSource << "#define HAS_TRANSFER_FUNCTION 1\n";
	}
	else
	{
		extraSource << "#define HAS_TRANSFER_FUNCTION 0\n";
	}

	//compile and fill constants
	extraSource << "#include \"renderer_volume_kernels3.cuh\"\n";
	const auto& fun = KernelLoader::Instance().getKernelFunction(
		kernelName, extraSource.str(), constantNames, false, false).value();
	if (auto c = getConstantDeclarationName(s); !c.empty())
	{
		CUdeviceptr ptr = fun.constant(c);
		fillConstantMemory(s, ptr, stream);
	}
	if (tf)
	{
		if (auto c = tf->getConstantDeclarationName(s); !c.empty())
		{
			CUdeviceptr ptr = fun.constant(c);
			tf->fillConstantMemory(s, ptr, stream);
		}
	}

	//output tensors
	auto positions = torch::empty({ numSamples, 3 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	auto densities = torch::empty({ numSamples, 1 },
		at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	torch::Tensor colors;
	if (tf)
		colors = torch::empty({ numSamples, 4 },
			at::TensorOptions().dtype(s.scalarType).device(c10::kCUDA));
	cuMat::DevicePointer<unsigned int> counter(1);
	CUMAT_SAFE_CALL(cudaMemset(counter.pointer(), 0, sizeof(unsigned int)));
	unsigned int* counterPtr = counter.pointer();

	//launch kernel
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());
	int minGridSize = fun.minGridSize();
	bool success = RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "IVolumeInterpolation::importanceSamplingWithProbabilityGrid", [&]()
		{
			//cast variables
			unsigned int numSamples_u = static_cast<unsigned int>(numSamples);
			const auto accProbabilityGrid = accessor< ::kernel::Tensor3Read<scalar_t>>(probabilityGrid);
			const auto accPosition = accessor< ::kernel::Tensor2RW<scalar_t>>(positions);
			const auto accDensities = accessor< ::kernel::Tensor2RW<scalar_t>>(densities);
			const auto accColors = tf ? accessor< ::kernel::Tensor2RW<scalar_t>>(colors) : ::kernel::Tensor2RW<scalar_t>();
			scalar_t maxValue_r = static_cast<scalar_t>(maxProbability);
			scalar_t minProb_r = static_cast<scalar_t>(minProb);
			scalar_t densityMin_r = static_cast<scalar_t>(densityMin);
			scalar_t densityMax_r = static_cast<scalar_t>(densityMax);

			const void* args[] = {
				&numSamples_u, &accProbabilityGrid, &accPosition, &accDensities, &accColors,
				&maxValue_r, &minProb_r, &counterPtr,
				&seed, &time, &densityMin_r, &densityMax_r };
			auto result = cuLaunchKernel(
				fun.fun(), minGridSize, 1, 1, fun.bestBlockSize(), 1, 1,
				0, stream, const_cast<void**>(args), NULL);
			if (result != CUDA_SUCCESS)
				return printError(result, kernelName);
			return true;
		});
	CUMAT_SAFE_CALL(cudaDeviceSynchronize());

	setBoxMin(oldBoxMin);
	setBoxMax(oldBoxMax);

	if (!success) throw std::runtime_error("Error during rendering!");

	return { positions, densities, colors };
}

void renderer::IVolumeInterpolation::registerPybindModule(pybind11::module& m)
{
	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;

	namespace py = pybind11;
	py::class_<IVolumeInterpolation, IVolumeInterpolation_ptr>(m, "IVolumeInterpolation")
		.def("supports_object_space_indexing", &IVolumeInterpolation::supportsObjectSpaceIndexing)
		.def("object_resolution", &IVolumeInterpolation::objectResolution)
		.def("box_min", &IVolumeInterpolation::boxMin)
		.def("box_max", &IVolumeInterpolation::boxMax)
		.def("box_size", &IVolumeInterpolation::boxSize)
		.def("voxel_size", &IVolumeInterpolation::voxelSize)
	    .def("output_channels", &IVolumeInterpolation::outputChannels)
		.def("evaluate", [](IVolumeInterpolation* self, torch::Tensor positions, const std::optional<torch::Tensor>& direction)
			{
				return self->evaluate(positions, direction.value_or(torch::Tensor()), c10::cuda::getCurrentCUDAStream());
			},
			py::doc("Evaluates the volume on the given position array of shape (B,3) and returns the interpolated densities of shape (B,1)"),
				py::arg("positions"), py::arg("direction")=std::optional<torch::Tensor>{})
		.def("importance_sampling", [](IVolumeInterpolation* self, int numSamples, ITransferFunction_ptr tf, double minProb,
			int seed, int time, double densityMin, double densityMax, const std::string& dtype)
		    {
				c10::ScalarType dtype2;
				if (dtype == "float")
					dtype2 = c10::ScalarType::Float;
				else if (dtype == "double")
					dtype2 = c10::ScalarType::Double;
				else
					throw std::runtime_error("Unknown scalar type");
				return self->importanceSampling(numSamples, tf, minProb, seed, time, densityMin, densityMax, dtype2, c10::cuda::getCurrentCUDAStream());
		    },
			py::doc(R"(
	 Performs importance sampling on the volume.
	 If the transfer function is null, the density values are used (and returned).
	 If the transfer function is defined, the alpha values are used and the color is returned.
	 :param num_samples: the number of samples to draw
	 :param tf: the transfer function, can be null
	 :param min_prob: the minimal probability of a sample in [0,1]
	 :param seed: the seed for the rng
	 :param time: the time increment for the rng
     :param min_density: the minimal density for the TF mapping
     :param max_density: the minimal density for the TF mapping
	 :param dtype: the scalar type ("float" or "double") of the output
	 :return: a tuple (positions, densities, colors) where
	 - positions is a tensor of shape (N,3) of xyz-positions in [0,1]^3, N=numSamples
	 - densities is a tensor of shape (N,1) of the densities
	 - colors is a tensor of shape (N,4) of the colors, but only defined if tf!=null
        )"), py::arg("num_samples"), py::arg("tf"), py::arg("min_prob"), 
				py::arg("seed"), py::arg("time"),
				py::arg("min_density"), py::arg("max_density"), py::arg("dtype"))
	    .def("importance_sampling_with_probability_grid", [](
			IVolumeInterpolation* self, int numSamples, ITransferFunction_ptr tf, 
			torch::Tensor probabilityGrid, double maxProbability, double minProb,
		    int seed, int time, double densityMin, double densityMax)
		    {
			    return self->importanceSamplingWithProbabilityGrid(
					numSamples, tf, probabilityGrid, maxProbability, minProb, seed, time, densityMin, densityMax, c10::cuda::getCurrentCUDAStream());
		    }, py::arg("num_samples"), py::arg("tf"), 
				py::arg("probability_grid"), py::arg("max_probability"),
				py::arg("min_prob"),
				py::arg("seed"), py::arg("time"),
				py::arg("min_density"), py::arg("max_density"))
		;
}

