#include "volume_interpolation_network.h"

#include <magic_enum.hpp>
#include <cuMat/src/Errors.h>
#include <cuda_runtime.h>
#include <iomanip>

#include "IconsFontAwesome5.h"
#include "portable-file-dialogs.h"
#include "pytorch_utils.h"
#include "renderer_utils.cuh"
#include "sha1.h"
#include "tinyformat.h"

static void writeString(std::ostream& o, const std::string& s)
{
	int l = s.length();
	o.write(reinterpret_cast<const char*>(&l), sizeof(int));
	o.write(s.data(), l);
}
static std::string loadString(std::istream& i)
{
	int l;
	i.read(reinterpret_cast<char*>(&l), sizeof(int));
	std::string s;
	s.resize(l);
	i.read(s.data(), l);
	return s;
}

int renderer::InputParametrization::channelsOut() const
{
	if (numFourierFeatures > 0) {
		return 4 + (hasDirection ? 4 : 0) + 2 * numFourierFeatures;
	}
	else
	{
		return 3 + (hasDirection ? 3 : 0);
	}
}

bool renderer::InputParametrization::valid() const
{
	if (useDirectionInFourierFeatures && !hasDirection)
	{
		std::cerr << "useDirectionInFourierFeatures==true requires hasDirection==true, but hasDirection is false" << std::endl;
		return false;
	}
	if (fourierMatrix.size() % 3 != 0)
	{
		std::cerr << "Fourier matrix size not divisible by three" << std::endl;
		return false;
	}
	int fourierChannels = useDirectionInFourierFeatures ? 6 : 3;
	if (numFourierFeatures >= 0 && (numFourierFeatures != fourierMatrix.size() / fourierChannels))
	{
		std::cerr << "Fourier features specified, but number of rows in 'fourierMatrix' does not match 'numFourierFeatures" << std::endl;
		return false;
	}
	if ((numFourierFeatures % 2) != 0)
	{
		std::cerr << "The number of fourier features must be divisible by 2, but is " << numFourierFeatures << std::endl;
		return false;
	}
	return true;
}

const int renderer::InputParametrization::VERSION = 3;

renderer::InputParametrization_ptr renderer::InputParametrization::load(std::istream& in)
{
	auto p = std::make_shared<InputParametrization>();

	int version;
	in.read(reinterpret_cast<char*>(&version), sizeof(int));

	if (version == 1)
	{
		in.read(reinterpret_cast<char*>(&p->hasDirection), sizeof(bool));
		in.read(reinterpret_cast<char*>(&p->numFourierFeatures), sizeof(int));
		p->useDirectionInFourierFeatures = false;
		FourierMatrix_t m(p->numFourierFeatures * 3);
		if (p->numFourierFeatures > 0)
			in.read(reinterpret_cast<char*>(m.data()), sizeof(half) * p->numFourierFeatures * 3);
		p->fourierMatrix = m;
	}
	else if (version == 2)
	{
		in.read(reinterpret_cast<char*>(&p->hasDirection), sizeof(bool));
		in.read(reinterpret_cast<char*>(&p->numFourierFeatures), sizeof(int));
		in.read(reinterpret_cast<char*>(&p->useDirectionInFourierFeatures), sizeof(bool));
		int C = p->useDirectionInFourierFeatures ? 6 : 3;
		FourierMatrix_t m(p->numFourierFeatures * C);
		if (p->numFourierFeatures > 0)
			in.read(reinterpret_cast<char*>(m.data()), sizeof(half) * p->numFourierFeatures * C);
		p->fourierMatrix = m;
	}
	else if (version == 3)
	{
		in.read(reinterpret_cast<char*>(&p->hasTime), sizeof(bool));
		in.read(reinterpret_cast<char*>(&p->hasDirection), sizeof(bool));
		in.read(reinterpret_cast<char*>(&p->numFourierFeatures), sizeof(int));
		in.read(reinterpret_cast<char*>(&p->useDirectionInFourierFeatures), sizeof(bool));
		int C = p->useDirectionInFourierFeatures ? 6 : 3;
		FourierMatrix_t m(p->numFourierFeatures * C);
		if (p->numFourierFeatures > 0)
			in.read(reinterpret_cast<char*>(m.data()), sizeof(half) * p->numFourierFeatures * C);
		p->fourierMatrix = m;
	}
	else
		throw std::runtime_error("Unknown version for InputParametrization " + std::to_string(version));

	return p;
}

void renderer::InputParametrization::save(std::ostream& out) const
{
	out.write(reinterpret_cast<const char*>(&VERSION), sizeof(int));
	out.write(reinterpret_cast<const char*>(&hasTime), sizeof(bool));
	out.write(reinterpret_cast<const char*>(&hasDirection), sizeof(bool));
	out.write(reinterpret_cast<const char*>(&numFourierFeatures), sizeof(int));
	out.write(reinterpret_cast<const char*>(&useDirectionInFourierFeatures), sizeof(bool));
	int C = useDirectionInFourierFeatures ? 6 : 3;
	TORCH_CHECK(numFourierFeatures*C == fourierMatrix.size());
	if (numFourierFeatures > 0)
		out.write(reinterpret_cast<const char*>(fourierMatrix.data()), sizeof(half) * numFourierFeatures * C);
}

void renderer::InputParametrization::setFourierMatrixFromTensor(const torch::Tensor& t, bool premultiplied)
{
	CHECK_DIM(t, 2);
	int C = t.size(1);
	if (C == 3)
	{
		useDirectionInFourierFeatures = false;
	}
	else if (C == 6)
	{
		if (!hasDirection)
			throw std::runtime_error("hasDirection==false, but the fourier matrix has input channels for the direction");
		useDirectionInFourierFeatures = true;
	}
	else
	{
		throw std::runtime_error(tinyformat::format("Unrecognized number of input channels. Actual: %d, expected: 3 or 6", C));
	}
	at::Tensor t2 = t.cpu();
	CHECK_DTYPE(t2, c10::kFloat);
	const auto tAcc = t2.accessor<float, 2>();
	numFourierFeatures = t.size(0);
	fourierMatrix.resize(numFourierFeatures * C);
	for (int cout = 0; cout < numFourierFeatures; ++cout)
		for (int cin = 0; cin < C; ++cin)
			fourierMatrix[cout + numFourierFeatures*cin] = __float2half(
				(premultiplied ? 1 : 2 * M_PI) * tAcc[cout][cin]);
}

void renderer::InputParametrization::disableFourierFeatures()
{
	numFourierFeatures = 0;
	useDirectionInFourierFeatures = false;
	fourierMatrix = FourierMatrix_t();
}

const std::string renderer::OutputParametrization::OutputModeNames[] = {
	"density",
	"density:direct",
	"rgbo",
	"rgbo:direct",
	"densitygrad",
	"densitygrad:direct",
	"densitygrad:cubic",
	"densitycurvature",
	"densitycurvature:direct"
};
const int renderer::OutputParametrization::OutputModeNumChannelsIn[] = {
	1, 1, 4, 4, 4, 4, 4, 6, 6
};
const int renderer::OutputParametrization::OutputModeNumChannelsOut[] = {
	1, 1, 4, 4, 1, 1, 1, 1, 1
};

renderer::OutputParametrization::OutputMode renderer::OutputParametrization::OutputModeFromString(const std::string& s)
{
	for (int i=0; i<_NUM_OUTPUT_MODES_; ++i)
	{
		if (OutputModeNames[i] == s) return OutputMode(i);
	}
	throw std::runtime_error("No output mode found matching string " + s);
}

int renderer::OutputParametrization::channelsIn() const
{
	return OutputModeNumChannelsIn[outputMode];
}

int renderer::OutputParametrization::channelsOut() const
{
	return OutputModeNumChannelsOut[outputMode];
}


const int renderer::OutputParametrization::VERSION = 1;

renderer::OutputParametrization_ptr renderer::OutputParametrization::load(std::istream& in)
{
	int version;
	in.read(reinterpret_cast<char*>(&version), sizeof(int));
	if (version != VERSION)
		throw std::runtime_error("Unknown version for OutputParametrization " + std::to_string(version));

	auto p = std::make_shared<OutputParametrization>();
	p->outputMode = OutputModeFromString(loadString(in));
	return p;
}

void renderer::OutputParametrization::save(std::ostream& out) const
{
	out.write(reinterpret_cast<const char*>(&VERSION), sizeof(int));
	writeString(out, OutputModeNames[outputMode]);
}

const std::string renderer::Layer::ActivationNames[] = {
	"ReLU",
	"Sine",
	"Snake",
	"SnakeAlt",
	"Sigmoid",
	"None"
};

renderer::Layer::Activation renderer::Layer::ActivationFromString(const std::string& s)
{
	for (int i = 0; i < _NUM_ACTIVATIONS_; ++i)
	{
		if (ActivationNames[i] == s) return Activation(i);
	}
	throw std::runtime_error("No output mode found matching string " + s);
}

bool renderer::Layer::valid(bool isOutputLayer) const
{
	return (weights.size() == channelsIn*channelsOut) &&
		(bias.size() == channelsOut) &&
		(isOutputLayer || (bias.size() % 4 == 0));
}

const int renderer::Layer::VERSION = 2;

renderer::Layer_ptr renderer::Layer::load(std::istream& in)
{
	int version;
	in.read(reinterpret_cast<char*>(&version), sizeof(int));
	if (version != 1 && version != 2)
		throw std::runtime_error("Unknown version for InputParametrization " + std::to_string(version));

	int rows, cols;
	in.read(reinterpret_cast<char*>(&rows), sizeof(int));
	in.read(reinterpret_cast<char*>(&cols), sizeof(int));

	weights_t fw(rows * cols);
	bias_t fb(rows);
	in.read(reinterpret_cast<char*>(fw.data()), sizeof(half) * rows * cols);
	in.read(reinterpret_cast<char*>(fb.data()), sizeof(half) * rows);

	Activation a = ActivationFromString(loadString(in));
	float activationParameter = 1;
	if (version == 2)
		in.read(reinterpret_cast<char*>(&activationParameter), sizeof(float));

	return std::make_shared<Layer>(cols, rows, fw, fb, a, activationParameter);
}

void renderer::Layer::save(std::ostream& out) const
{
	out.write(reinterpret_cast<const char*>(&VERSION), sizeof(int));

	int rows = channelsOut;
	int cols = channelsIn;
	out.write(reinterpret_cast<const char*>(&rows), sizeof(int));
	out.write(reinterpret_cast<const char*>(&cols), sizeof(int));

	out.write(reinterpret_cast<const char*>(weights.data()), sizeof(half) * rows * cols);
	out.write(reinterpret_cast<const char*>(bias.data()), sizeof(half) * rows);

	writeString(out, ActivationNames[activation]);
	out.write(reinterpret_cast<const char*>(&activationParameter), sizeof(float));
}

renderer::LatentGrid::LatentGrid(const torch::Tensor& t_, Encoding encoding)
{
	CHECK_DIM(t_, 5);
	CHECK_SIZE(t_, 0, 1);
	torch::Tensor t = t_.to(c10::kCPU, c10::kFloat);

	gridChannels = t.size(1);
	gridSizeZ = t.size(2);
	gridSizeY = t.size(3);
	gridSizeX = t.size(4);

	this->encoding = encoding;
	switch (encoding)
	{
	case FLOAT:
		initEncodingFloat(t);
		break;
	case BYTE_LINEAR:
		initEncodingByteLinear(t);
		break;
	case BYTE_GAUSSIAN:
		initEncodingByteGaussian(t);
		break;
	default:
		throw std::runtime_error("Unsupported encoding");
	}
}

double renderer::LatentGrid::LastEncodingError = 0;

void renderer::LatentGrid::initEncodingFloat(const torch::Tensor& t)
{
	static_assert(sizeof(float) == 4, "What compiler did you smoke?");
	const auto tAcc = t.accessor<float, 5>();
	grid_t g(t.numel() * 4);
	float* data = reinterpret_cast<float*>(g.data());
	for (int c = 0; c < gridChannels; ++c) for (int z = 0; z < gridSizeZ; ++z) for (int y = 0; y < gridSizeY; ++y) for (int x = 0; x < gridSizeX; ++x)
	{
		int cHigh = c / 4;
		int cLow = c % 4;
		data[idx(cHigh, z, y, x, cLow)] = tAcc[0][c][z][y][x];
	}
	this->grid = g;
	LastEncodingError = 0;
}

void renderer::LatentGrid::initEncodingByteLinear(const torch::Tensor& t)
{
	//compute minimal and maximal value
	std::vector<float> minValues(gridChannels);
	std::vector<float> maxValues(gridChannels);
	std::cout << "Add LatentGrid with linear encoding, min/max per channel:";
	for (int c = 0; c < gridChannels; ++c)
	{
		minValues[c] = torch::min(t.select(1, c)).item<float>();
		maxValues[c] = torch::max(t.select(1, c)).item<float>();
		std::cout << " " << std::setprecision(3) << minValues[c] << "/" << maxValues[c];
	}
	std::cout << std::endl;
	//transform and save
	gridOffsetOrMean.resize(gridChannels);
	gridScaleOrStd.resize(gridChannels);
	grid.resize(t.numel());
	const auto tAcc = t.accessor<float, 5>();
	double encodingError = 0;
	for (int c = 0; c < gridChannels; ++c)
	{
		gridOffsetOrMean[c] = minValues[c];
		gridScaleOrStd[c] = maxValues[c] - minValues[c];
		float invScale = 1.0f / std::max(1e-5f, maxValues[c] - minValues[c]);
		for (int z = 0; z < gridSizeZ; ++z) for (int y = 0; y < gridSizeY; ++y) for (int x = 0; x < gridSizeX; ++x)
		{
			float value = tAcc[0][c][z][y][x];
			float x01 = (value - minValues[c]) * invScale;
			int x255 = std::max(0, std::min(255, static_cast<int>(std::roundf(255 * x01))));
			int cHigh = c / 4;
			int cLow = c % 4;
			auto xFinal = static_cast<uint8_t>(x255);
			grid[idx(cHigh, z, y, x, cLow)] = xFinal;
			//reverse-conversion to validate error
			float valueReconstructed = gridOffsetOrMean[c] + xFinal / 255.0f * gridScaleOrStd[c];
			//std::cout << "v=" << std::setprecision(4) << value << " -> " << int(xFinal) << " -> " << valueReconstructed << std::endl;
			encodingError += std::abs(value - valueReconstructed);
		}
	}
	std::cout << "Average absolute encoding error: " << std::setprecision(6) <<
		(encodingError / t.numel()) << std::endl;
	LastEncodingError = encodingError / t.numel();
}

void renderer::LatentGrid::initEncodingByteGaussian(const torch::Tensor& t)
{
	//compute mean and variance
	std::vector<float> meanValues(gridChannels);
	std::vector<float> stdValues(gridChannels);
	std::cout << "Add LatentGrid with gaussian encoding, mean/std per channel:";
	for (int c = 0; c < gridChannels; ++c)
	{
		auto [std, mean] = torch::std_mean(t.select(1, c));
		meanValues[c] = mean.item<float>();
		stdValues[c] = std.item<float>();
		std::cout << " " << std::setprecision(2) << meanValues[c] << "/" << 
			std::setprecision(3) << stdValues[c];
	}
	std::cout << std::endl;
	//transform and save
	gridOffsetOrMean.resize(gridChannels);
	gridScaleOrStd.resize(gridChannels);
	grid.resize(t.numel());
	const auto tAcc = t.accessor<float, 5>();
	double encodingError = 0;
	for (int c = 0; c < gridChannels; ++c)
	{
		gridOffsetOrMean[c] = meanValues[c];
		gridScaleOrStd[c] = stdValues[c];
		float invStd = 1.0f / std::max(1e-5f, stdValues[c]);
		for (int z = 0; z < gridSizeZ; ++z) for (int y = 0; y < gridSizeY; ++y) for (int x = 0; x < gridSizeX; ++x)
		{
			float vx = tAcc[0][c][z][y][x]; // vx \prop N(mean, std)
			float vxHat = (vx - meanValues[c]) * invStd; // vxHat \prop N(0,1)
			static constexpr float invSqrt2 = 0.7071067811865475244008443621048f; // 1/sqrt(2)
			float theta01 = 0.5f * (1 + std::erf(vxHat * invSqrt2));
			int theta255 = std::max(0, std::min(255, static_cast<int>(std::roundf(255 * theta01))));
			int cHigh = c / 4;
			int cLow = c % 4;
			auto xFinal = static_cast<uint8_t>(theta255);
			grid[idx(cHigh, z, y, x, cLow)] = xFinal;
			//reverse-conversion to validate error
			//constants copied from renderer_volume_tensorcores.cuh
			static constexpr float ENCODING_GAUSSIAN_EPSILON = 1e-4f;
			static constexpr float ENCODING_GAUSSIAN_2_MINUS_EPSILON = 2 - 1e-4f;
			static constexpr float ENCODING_GAUSSIAN_SQRT2 = 1.4142135623730950488016887242096980f;
			float tmp = ENCODING_GAUSSIAN_SQRT2 * myErfInv(ENCODING_GAUSSIAN_2_MINUS_EPSILON * (xFinal / 255.0f - 0.5f));
			float valueReconstructed = gridOffsetOrMean[c] + tmp * gridScaleOrStd[c];
			//std::cout << "v=" << std::setprecision(4) << vx << " -> " << int(xFinal) << " -> " << valueReconstructed << std::endl;
			encodingError += std::abs(vx - valueReconstructed);
		}
	}
	std::cout << "Average absolute encoding error: " << std::setprecision(6) <<
		(encodingError / t.numel()) << std::endl;
	LastEncodingError = encodingError / t.numel();
}

bool renderer::LatentGrid::isValid() const
{
	if (gridChannels<=0 || gridSizeX<=0 || gridSizeY<=0 || gridSizeZ<=0)
	{
		std::cerr << "Error, LatentGrid: all dimensions must be positive" << std::endl;
		return false;
	}
	if (gridChannels%16 != 0)
	{
		std::cerr << "Error, LatentGrid: the number of channels must be divisible by 16" << std::endl;
		return false;
	}

	size_t entries = grid.size();
	size_t expected = bytesPerEntry() * gridChannels * gridSizeZ * gridSizeY * gridSizeX;
	if (entries != expected)
	{
		std::cerr << "Error, LatentGrid: illegal grid size" << std::endl;
		return false;
	}
	if (encoding != FLOAT)
	{
		if (gridOffsetOrMean.size() != gridChannels)
		{
			std::cerr << "Error, LatentGrid: gridOffsetOrMean must contain gridChannels entries" << std::endl;
			return false;
		}
		if (gridScaleOrStd.size() != gridChannels)
		{
			std::cerr << "Error, LatentGrid: gridScaleOrStd must contain gridChannels entries" << std::endl;
			return false;
		}
	}

	return true;
}

renderer::LatentGrid::GPUArray::GPUArray(int sizeX, int sizeY, int sizeZ, bool isFloat, const char* data)
	: array(nullptr), texture(0)
{
	//create array
	cudaExtent extent = make_cudaExtent(sizeX, sizeY, sizeZ);
	int bytesPerType = isFloat ? 4 : 1;
	int bitsPerType = 8 * bytesPerType;
	auto format = isFloat ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
		bitsPerType, bitsPerType, bitsPerType, bitsPerType, format);
	CUMAT_SAFE_CALL(cudaMalloc3DArray(&array, &channelDesc, extent));

	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = make_cudaPitchedPtr(const_cast<char*>(data),
		bytesPerType * sizeX * 4, sizeX, sizeY);
	params.dstArray = array;
	params.extent = extent;
	params.kind = cudaMemcpyHostToDevice;
	CUMAT_SAFE_CALL(cudaMemcpy3D(&params));

	//create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = array;
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = isFloat ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&texture, &resDesc, &texDesc, NULL));
}

renderer::LatentGrid::GPUArray::~GPUArray()
{
	if (texture) {
		CUMAT_SAFE_CALL_NO_THROW(cudaDestroyTextureObject(texture));
		texture = 0;
	}
	if (array)
	{
		CUMAT_SAFE_CALL_NO_THROW(cudaFreeArray(array));
		array = nullptr;
	}
}

void renderer::LatentGrid::clearGPUResources()
{
	gpuResources_.clear();
}

void renderer::LatentGrid::copyGridToGPU(bool skipIfAlreadyInitialized)
{
	if (skipIfAlreadyInitialized && !gpuResources_.empty()) return;
	TORCH_CHECK(gridChannels % 4 == 0, "gridChannels must be divisible by four");
	gpuResources_.resize(gridChannels / 4);
	size_t stride = bytesPerEntry() * 4 * gridSizeX * gridSizeY * gridSizeZ;
	for (int i=0; i<gridChannels/4; ++i)
	{
		gpuResources_[i] = std::make_shared<GPUArray>(
			gridSizeX, gridSizeY, gridSizeZ, encoding == FLOAT, grid.data() + (i * stride));
	}
}

cudaTextureObject_t renderer::LatentGrid::getTexture(int index) const
{
	TORCH_CHECK(index >= 0 && index < gridChannels / 4, "index of of bounds");
	TORCH_CHECK(!gpuResources_.empty(), "GPU resources not yet created");
	return gpuResources_[index]->texture;
}

float4 renderer::LatentGrid::getOffsetOrMean(int index) const
{
	TORCH_CHECK(index >= 0 && index < gridChannels / 4, "index of of bounds");
	TORCH_CHECK(encoding != FLOAT);
	index *= 4;
	return make_float4(
		gridOffsetOrMean[index], gridOffsetOrMean[index + 1],
		gridOffsetOrMean[index + 2], gridOffsetOrMean[index + 3]);
}

float4 renderer::LatentGrid::getScaleOrStd(int index) const
{
	TORCH_CHECK(index >= 0 && index < gridChannels / 4, "index of of bounds");
	TORCH_CHECK(encoding != FLOAT);
	index *= 4;
	return make_float4(
		gridScaleOrStd[index], gridScaleOrStd[index + 1],
		gridScaleOrStd[index + 2], gridScaleOrStd[index + 3]);
}

const int renderer::LatentGrid::VERSION = 1;

std::shared_ptr<renderer::LatentGrid> renderer::LatentGrid::load(std::istream& in)
{
	int version;
	in.read(reinterpret_cast<char*>(&version), sizeof(int));
	if (version != VERSION)
		throw std::runtime_error("Unknown version for LatentGrid " + std::to_string(version));

	auto g = std::make_shared<LatentGrid>();
	in.read(reinterpret_cast<char*>(&g->encoding), sizeof(int));
	in.read(reinterpret_cast<char*>(&g->gridChannels), sizeof(int));
	in.read(reinterpret_cast<char*>(&g->gridSizeZ), sizeof(int));
	in.read(reinterpret_cast<char*>(&g->gridSizeY), sizeof(int));
	in.read(reinterpret_cast<char*>(&g->gridSizeX), sizeof(int));

	size_t entries = g->bytesPerEntry() * g->gridChannels * g->gridSizeZ * g->gridSizeY * g->gridSizeX;
	grid_t data(entries);
	in.read(data.data(), entries);
	g->grid = data;
	if (g->encoding != FLOAT)
	{
		g->gridOffsetOrMean.resize(g->gridChannels);
		g->gridScaleOrStd.resize(g->gridChannels);
		in.read(reinterpret_cast<char*>(g->gridOffsetOrMean.data()), sizeof(float) * g->gridChannels);
		in.read(reinterpret_cast<char*>(g->gridScaleOrStd.data()), sizeof(float) * g->gridChannels);
	}

	return g;
}

void renderer::LatentGrid::save(std::ostream& out) const
{
	if (!isValid())
		throw std::runtime_error("LatentGridTimeAndEnsemble is not valid, cannot save");
	out.write(reinterpret_cast<const char*>(&VERSION), sizeof(int));
	int e = static_cast<int>(encoding);
	out.write(reinterpret_cast<const char*>(&e), sizeof(int));
	out.write(reinterpret_cast<const char*>(&gridChannels), sizeof(int));
	out.write(reinterpret_cast<const char*>(&gridSizeZ), sizeof(int));
	out.write(reinterpret_cast<const char*>(&gridSizeY), sizeof(int));
	out.write(reinterpret_cast<const char*>(&gridSizeX), sizeof(int));
	size_t memory = bytesPerEntry() * gridChannels * gridSizeZ * gridSizeY * gridSizeX;
	const char* data = grid.data();
	out.write(data, memory);
	if (encoding != FLOAT)
	{
		out.write(reinterpret_cast<const char*>(gridOffsetOrMean.data()), sizeof(float)*gridChannels);
		out.write(reinterpret_cast<const char*>(gridScaleOrStd.data()), sizeof(float) * gridChannels);
	}
}

double renderer::LatentGridTimeAndEnsemble::setTimeGridFromTorch(int index, const torch::Tensor& t,
															   LatentGrid::Encoding encoding)
{
	TORCH_CHECK(index >= 0 && index < timeGrids.size(), "index out of bounds!");
	timeGrids[index] = std::make_shared<LatentGrid>(t, encoding);
	return LatentGrid::LastEncodingError;
}

double renderer::LatentGridTimeAndEnsemble::setEnsembleGridFromTorch(int index, const torch::Tensor& t,
	LatentGrid::Encoding encoding)
{
	TORCH_CHECK(index >= 0 && index < ensembleGrids.size(), "index out of bounds!");
	ensembleGrids[index] = std::make_shared<LatentGrid>(t, encoding);
	return LatentGrid::LastEncodingError;
}

bool renderer::LatentGridTimeAndEnsemble::isValid() const
{
	if (timeGrids.empty() && ensembleGrids.empty())
	{
		std::cerr << "Either time or ensemble grids must be specified!" << std::endl;
		return false;
	}

	//check common encoding
	LatentGrid::Encoding encoding = LatentGrid::FLOAT;
	bool encodingSet = false;
	for (auto g : timeGrids)
	{
		if (!g)
		{
			std::cerr << "One latent grid was null" << std::endl;
			return false;
		}
		if (!g->isValid())
			return false;
		if (encodingSet)
		{
			if (encoding != g->encoding)
			{
				std::cerr << "All latent grids must share the same encoding modes, but this grid uses " <<
					magic_enum::enum_name(g->encoding) << ", while previous grids used " <<
					magic_enum::enum_name(encoding) << std::endl;
				return false;
			}
		}
		else
			encoding = g->encoding;
	}
	for (auto g : ensembleGrids)
	{
		if (!g)
		{
			std::cerr << "One latent grid was null" << std::endl;
			return false;
		}
		if (!g->isValid())
			return false;
		if (encodingSet)
		{
			if (encoding != g->encoding)
			{
				std::cerr << "All latent grids must share the same encoding modes, but this grid uses " <<
					magic_enum::enum_name(g->encoding) << ", while previous grids used " <<
					magic_enum::enum_name(encoding) << std::endl;
				return false;
			}
		}
		else
			encoding = g->encoding;
	}

	//check common channel count
	if (!timeGrids.empty())
	{
		int c = timeGrids[0]->gridChannels;
		for (size_t i=1; i<timeGrids.size(); ++i)
		{
			if (timeGrids[i]->gridChannels != c)
			{
				std::cerr << "Time grid " << i << " uses a different channel count of " <<
					timeGrids[i]->gridChannels << " than previous grids with a channel count of " <<
					c << std::endl;
				return false;
			}
		}
	}
	if (!ensembleGrids.empty())
	{
		int c = ensembleGrids[0]->gridChannels;
		for (size_t i = 1; i < ensembleGrids.size(); ++i)
		{
			if (ensembleGrids[i]->gridChannels != c)
			{
				std::cerr << "Time grid " << i << " uses a different channel count of " <<
					ensembleGrids[i]->gridChannels << " than previous grids with a channel count of " <<
					c << std::endl;
				return false;
			}
		}
	}
	return true;
}

void renderer::LatentGridTimeAndEnsemble::clearGPUResources()
{
	for (auto g : timeGrids) g->clearGPUResources();
	for (auto g : ensembleGrids) g->clearGPUResources();
}

int renderer::LatentGridTimeAndEnsemble::getResolution() const
{
	assert(isValid());
	if (!timeGrids.empty()) return timeGrids[0]->gridSizeX;
	if (!ensembleGrids.empty()) return ensembleGrids[0]->gridSizeX;
	throw std::runtime_error("at least one grid must be active!");
}

renderer::LatentGrid::Encoding renderer::LatentGridTimeAndEnsemble::getCommonEncoding() const
{
	assert(isValid());
	if (!timeGrids.empty()) return timeGrids[0]->encoding;
	if (!ensembleGrids.empty()) return ensembleGrids[0]->encoding;
	throw std::runtime_error("at least one grid must be active!");
}

int renderer::LatentGridTimeAndEnsemble::getTimeChannels() const
{
	assert(isValid());
	if (timeGrids.empty()) return 0;
	return timeGrids[0]->gridChannels;
}

int renderer::LatentGridTimeAndEnsemble::getEnsembleChannels() const
{
	assert(isValid());
	if (ensembleGrids.empty()) return 0;
	return ensembleGrids[0]->gridChannels;
}

const int renderer::LatentGridTimeAndEnsemble::VERSION = 1;

renderer::LatentGridTimeAndEnsemble_ptr renderer::LatentGridTimeAndEnsemble::load(std::istream& in)
{
	int version;
	in.read(reinterpret_cast<char*>(&version), sizeof(int));
	if (version > VERSION)
		throw std::runtime_error("Unknown version for LatentGridTimeAndEnsemble " + std::to_string(version));

	auto g = std::make_shared<LatentGridTimeAndEnsemble>();
	in.read(reinterpret_cast<char*>(&g->timeMin), sizeof(int));
	in.read(reinterpret_cast<char*>(&g->timeNum), sizeof(int));
	in.read(reinterpret_cast<char*>(&g->timeStep), sizeof(int));
	in.read(reinterpret_cast<char*>(&g->ensembleMin), sizeof(int));
	in.read(reinterpret_cast<char*>(&g->ensembleNum), sizeof(int));
	std::cout << "Load " << g->timeNum << " time grids and " << g->ensembleNum << " ensemble grids" << std::endl;
	g->timeGrids.resize(g->timeNum);
	for (int i = 0; i < g->timeNum; ++i)
		g->timeGrids[i] = LatentGrid::load(in);
	g->ensembleGrids.resize(g->ensembleNum);
	for (int i = 0; i < g->ensembleNum; ++i)
		g->ensembleGrids[i] = LatentGrid::load(in);

	return g;
}

void renderer::LatentGridTimeAndEnsemble::save(std::ostream& out) const
{
	if (!isValid())
		throw std::runtime_error("LatentGridTimeAndEnsemble is not valid, cannot save");
	out.write(reinterpret_cast<const char*>(&VERSION), sizeof(int));
	out.write(reinterpret_cast<const char*>(&timeMin), sizeof(int));
	out.write(reinterpret_cast<const char*>(&timeNum), sizeof(int));
	out.write(reinterpret_cast<const char*>(&timeStep), sizeof(int));
	out.write(reinterpret_cast<const char*>(&ensembleMin), sizeof(int));
	out.write(reinterpret_cast<const char*>(&ensembleNum), sizeof(int));
	for (int i = 0; i < timeNum; ++i)
		timeGrids[i]->save(out);
	for (int i = 0; i < ensembleNum; ++i)
		ensembleGrids[i]->save(out);
}

renderer::SceneNetwork::SceneNetwork()
	: boxMin_{make_float3(-5.f)}
	, boxSize_{make_float3(1.f)}
	, input_{std::make_shared<InputParametrization>()}
	, output_{std::make_shared<OutputParametrization>()}
{
}

void renderer::SceneNetwork::addLayer(Layer_ptr layer)
{
	if (hidden_.empty() && input_->numFourierFeatures>0)
	{
		//first layer, modify to include skipped inputs
		Layer::weights_t wOld = layer->weights;
		Layer::weights_t wNew;
		int newChannelsIn = layer->channelsIn;
		if (!input_->hasTime) {
			if (input_->hasDirection)
			{
				newChannelsIn = layer->channelsIn + 2;
				wNew.resize((layer->channelsIn + 2) * layer->channelsOut);
				//position
				for (int cin = 0; cin < 3; cin++)
					for (int cout = 0; cout < layer->channelsOut; ++cout)
						wNew[cout * newChannelsIn + cin] = wOld[cout * layer->channelsIn + cin];
				//direction
				for (int cin = 0; cin < 3; cin++)
					for (int cout = 0; cout < layer->channelsOut; ++cout)
						wNew[cout * newChannelsIn + cin + 4] = wOld[cout * layer->channelsIn + cin + 3];
				//fourier + extra
				int numExtra = layer->channelsIn - 6;
				for (int cin = 0; cin < numExtra; cin++)
					for (int cout = 0; cout < layer->channelsOut; ++cout)
						wNew[cout * newChannelsIn + cin + 8] = wOld[cout * layer->channelsIn + cin + 6];
			}
			else
			{
				newChannelsIn = layer->channelsIn + 1;
				wNew.resize((layer->channelsIn + 1) * layer->channelsOut);
				//position
				for (int cin = 0; cin < 3; cin++)
					for (int cout = 0; cout < layer->channelsOut; ++cout)
						wNew[cout * newChannelsIn + cin] = wOld[cout * layer->channelsIn + cin];
				//fourier + extra
				int numExtra = layer->channelsIn - 3;
				for (int cin = 0; cin < numExtra; cin++)
					for (int cout = 0; cout < layer->channelsOut; ++cout)
						wNew[cout * newChannelsIn + cin + 4] = wOld[cout * layer->channelsIn + cin + 3];
			}
			hidden_.push_back(std::make_shared<Layer>(newChannelsIn, layer->channelsOut, wNew, layer->bias, layer->activation, layer->activationParameter));
		} else
		{
			//time adds an extra input after position -> only modify if we also have direction
			if (input_->hasDirection)
			{
				newChannelsIn = layer->channelsIn + 1;
				wNew.resize((layer->channelsIn + 1) * layer->channelsOut);
				//position + time
				for (int cin = 0; cin < 4; cin++)
					for (int cout = 0; cout < layer->channelsOut; ++cout)
						wNew[cout * newChannelsIn + cin] = wOld[cout * layer->channelsIn + cin];
				//direction
				for (int cin = 0; cin < 3; cin++)
					for (int cout = 0; cout < layer->channelsOut; ++cout)
						wNew[cout * newChannelsIn + cin + 4] = wOld[cout * layer->channelsIn + cin + 4];
				//fourier + extra
				int numExtra = layer->channelsIn - 6;
				for (int cin = 0; cin < numExtra; cin++)
					for (int cout = 0; cout < layer->channelsOut; ++cout)
						wNew[cout * newChannelsIn + cin + 8] = wOld[cout * layer->channelsIn + cin + 7];
				//add modified
				hidden_.push_back(std::make_shared<Layer>(newChannelsIn, layer->channelsOut, wNew, layer->bias, layer->activation, layer->activationParameter));
			} else
			{
				//no change needed
				hidden_.push_back(layer);
			}
		}
	}
	else if (layer->channelsIn < 16 || layer->channelsOut < 16)
	{
		//first layer (no fourier features) or last layer (to density/color)
		//transpose
		Layer::weights_t wOld = layer->weights;
		Layer::weights_t wNew;
		wNew.resize(wOld.size());
		for (int cout = 0; cout < layer->channelsOut; ++cout)
			for (int cin = 0; cin < layer->channelsIn; ++cin)
				wNew[cout + layer->channelsOut * cin] = wOld[cout * layer->channelsIn + cin];
		hidden_.push_back(std::make_shared<Layer>(layer->channelsIn, layer->channelsOut, wNew, layer->bias, layer->activation, layer->activationParameter));
	}
	else
	{
		//regular hidden layers, simply add
		hidden_.push_back(layer);
	}
}

void renderer::SceneNetwork::addLayerFromTorch(const torch::Tensor& weights, const torch::Tensor& bias,
	Layer::Activation activation, float activationParameter)
{
	CHECK_DIM(weights, 2);
	CHECK_DIM(bias, 1);
	CHECK_SIZE(weights, 0, bias.size(0));

	at::Tensor weights2 = weights.cpu();
	at::Tensor bias2 = bias.cpu();
	const auto weightsAcc = weights2.accessor<float, 2>();
	const auto biasAcc = bias2.accessor<float, 1>();

	int channelsIn = weights.size(1);
	int channelsOut = weights.size(0);

	Layer::weights_t w(channelsIn * channelsOut);
	for (int cout = 0; cout < channelsOut; ++cout)
		for (int cin = 0; cin < channelsIn; ++cin)
			w[cout * channelsIn + cin] = __float2half(weightsAcc[cout][cin]);

	Layer::bias_t b(bias.size(0));
	for (int col = 0; col < b.size(); ++col)
		b[col] = __float2half(biasAcc[col]);

	addLayer(std::make_shared<Layer>(channelsIn, channelsOut, w, b, activation, activationParameter));
}

void renderer::SceneNetwork::setTimeAndEnsemble(float time, int ensemble)
{
	if (!latentGrid())
	{
		std::cerr << "No latent grid specified, setTimeAndEnsemble has no effect" << std::endl;
		return;
	}
	//silently clamp
	currentTime_ = clamp(time, 
		static_cast<float>(latentGrid()->timeMin), 
		static_cast<float>(latentGrid()->timeMaxInclusive()));
	currentEnsemble_ = clamp(ensemble, latentGrid()->ensembleMin, latentGrid()->ensembleMaxInclusive());

	//clear cache, as time and ensemble are written to constant memory
	cacheConstantMemory_.clear();
}

bool renderer::SceneNetwork::valid() const
{
	if (!input()->valid())
	{
		std::cerr << "Input parametrization is invalid" << std::endl;
		return false;
	}
	if (latentGrid() && !latentGrid()->isValid())
	{
		std::cerr << "LatentGrid is invalid" << std::endl;
		return false;
	}
	if (latentGrid() && input()->numFourierFeatures==0)
	{
		std::cerr << "Currently, LatentGrid requires fourier features as well" << std::endl;
		return false;
	}

	int current_channel = input()->channelsOut();
	if (latentGrid()) current_channel += latentGrid()->getTotalChannels();
	for (size_t i = 0; i < hidden_.size(); ++i)
	{
		auto l = hidden_[i];
		if (l->channelsIn != current_channel)
		{
			std::cerr << "Invalid input channels at hidden layer " << i << ", expected " << current_channel
				<< ", got " << l->channelsIn << std::endl;
			return false;
		}
		if (!l->valid(i == hidden_.size()-1))
		{
			std::cerr << "Invalid hidden layer " << i <<
				", probably weights and bias don't match or are not a multiple of 4" << std::endl;
			return false;
		}
		current_channel = l->channelsOut;
	}
	if (current_channel != output()->channelsIn())
	{
		std::cerr <<
			"Output channels from the hidden layers don't match the expected channels for the output parametrization. "
			<< "Expected " << output()->channelsIn() << ", got " << current_channel << std::endl;
		return false;
	}
	return true;
}

int renderer::SceneNetwork::computeMaxWarps(bool onlySharedMemory, bool adjoint) const
{
	static const int maxShared = 48 * 1024;
	static const int maxConstant = 16 * 1024;
	static const int bytesPerEntry = sizeof(half);
	static const int warpSize = 32;
	static_assert(sizeof(half) == 2, "What compiler did you smoke?");

	int numShared = 0; //num entries
	int numConst = 0;
	//input
	if (input_->numFourierFeatures)
		numConst += input_->fourierMatrix.size();
	//hidden
	int lastChannels = input_->channelsOut();
	int maxChannels = lastChannels;
	for (const auto& l : hidden_)
	{
		//input layer (if not fourier features) and output layer are in constant memory
		if (l->channelsIn<16 || l->channelsOut<16)
		{
			numConst += l->weights.size() + l->bias.size();
		}
		else {
			numShared += l->weights.size() + l->bias.size();
		}
		lastChannels = l->channelsOut;
		maxChannels = max(maxChannels, lastChannels);
	}

	if (onlySharedMemory)
	{
		numShared += numConst;
		numConst = 0;
	}

	int entriesPerThread = maxChannels;
	if (adjoint)
	{
		entriesPerThread += (static_cast<int>(hidden_.size()) - 1) * maxChannels;
	}

	//scale with bytes per entry
	numShared *= bytesPerEntry;
	numConst *= bytesPerEntry;
	maxChannels *= bytesPerEntry;
	entriesPerThread *= bytesPerEntry;

	if (numConst > maxConstant)
		return -1; //constants out of bounds
	int numWarps = static_cast<int>(std::floor((maxShared - numShared) / static_cast<float>(entriesPerThread * warpSize)));
	if (numWarps <= 0)
		return -1; //shared memory out of bounds
	return numWarps;
}

int renderer::SceneNetwork::numParameters() const
{
	int numParameters = 0;
	//input
	if (input_->numFourierFeatures)
		numParameters += input_->fourierMatrix.size();
	//hidden
	for (const auto& l : hidden_)
	{
		numParameters += l->weights.size() + l->bias.size();
	}
	return numParameters;
}

const int renderer::SceneNetwork::VERSION = 2;

renderer::SceneNetwork_ptr renderer::SceneNetwork::load(std::istream& in)
{
	int version;
	in.read(reinterpret_cast<char*>(&version), sizeof(int));
	if (version > VERSION)
		throw std::runtime_error("Unknown version for InputParametrization " + std::to_string(version));

	SceneNetwork_ptr p = std::make_shared<SceneNetwork>();
	p->input_ = InputParametrization::load(in);
	p->output_ = OutputParametrization::load(in);
	int numLayers;
	in.read(reinterpret_cast<char*>(&numLayers), sizeof(int));
	for (int i = 0; i < numLayers; ++i)
		p->hidden_.push_back(Layer::load(in));
	in.read(reinterpret_cast<char*>(&p->boxMin_.x), sizeof(float3));
	in.read(reinterpret_cast<char*>(&p->boxSize_.x), sizeof(float3));
	if (version == 2)
	{
		//include latent grid
		char hasLatentGrid;
		in.read(&hasLatentGrid, 1);
		if (hasLatentGrid>0)
		{
			p->latentGrid_ = LatentGridTimeAndEnsemble::load(in);
		}
	}
	return p;
}

void renderer::SceneNetwork::save(std::ostream& out) const
{
	if (!valid())
		throw std::runtime_error("scene network is not valid, cannot save");
	out.write(reinterpret_cast<const char*>(&VERSION), sizeof(int));
	input_->save(out);
	output_->save(out);
	int numLayers = hidden_.size();
	out.write(reinterpret_cast<const char*>(&numLayers), sizeof(int));
	for (int i = 0; i < numLayers; ++i)
		hidden_[i]->save(out);
	out.write(reinterpret_cast<const char*>(&boxMin_.x), sizeof(float3));
	out.write(reinterpret_cast<const char*>(&boxSize_.x), sizeof(float3));
	char hasLatentGrid = latentGrid_ ? 1 : 0;
	out.write(&hasLatentGrid, 1);
	if (latentGrid_) latentGrid_->save(out);
}

void renderer::SceneNetwork::clearGPUResources()
{
	cacheConstantMemory_.clear();
	cacheDefines_.clear();
	if (latentGrid())
		latentGrid()->clearGPUResources();
}

std::string renderer::SceneNetwork::codeReturnType() const
{
	switch (output()->outputMode)
	{
	case OutputParametrization::DENSITY:
	case OutputParametrization::DENSITY_DIRECT:
		return "real_t";
	case OutputParametrization::RGBO:
	case OutputParametrization::RGBO_DIRECT:
		return "real4";
	case OutputParametrization::DENSITY_GRADIENT:
	case OutputParametrization::DENSITY_GRADIENT_DIRECT:
	case OutputParametrization::DENSITY_GRADIENT_CUBIC:
	case OutputParametrization::DENSITY_CURVATURE:
	case OutputParametrization::DENSITY_CURVATURE_DIRECT:
		return "real_t";
	default: throw std::runtime_error("Unknown output mode");
	}
}

bool renderer::SceneNetwork::supportsNormals() const
{
	return output_->isDensityGradient() || output_->isDensityCurvature();
}

std::string renderer::SceneNetwork::getDefines(
	const IKernelModule::GlobalSettings& s, 
	int numWarps, bool firstAndLastInSharedMemory,
	GradientMode gradientMode) const
{
	// For time measures: keep the gradient mode (e.g. adjoint), even if no gradients
	// are needed in the end.
	//if (!s.volumeShouldProvideNormals) gradientMode = GradientMode::OFF_OR_DIRECT;

	if (output_->isColor()) gradientMode = GradientMode::OFF_OR_DIRECT;
	if (!cacheDefines_.empty() &&
		cacheNumWarps_ == numWarps &&
		cacheFirstAndLastInSharedMemory_ == firstAndLastInSharedMemory &&
		cacheGradientMode_ == gradientMode) {
		return cacheDefines_;
	}
	cacheNumWarps_ = numWarps;
	cacheFirstAndLastInSharedMemory_ = firstAndLastInSharedMemory;
	cacheGradientMode_ = gradientMode;

	bool hasVolumetricFeatures = latentGrid_ != nullptr;
	bool hasFourierFeatures = input_->numFourierFeatures > 0;

	int hiddenChannels = hasFourierFeatures ? hidden_[0]->channelsIn : hidden_[0]->channelsOut;

	int latentGridChannelsDiv16 = 0;
	int latentGridEncoding = 0;
	if (hasVolumetricFeatures)
	{
		if (!latentGrid_->isValid()) throw std::runtime_error("Latent Grid invalid");
		latentGridChannelsDiv16 = latentGrid_->getTotalChannels() / 16;
		latentGridEncoding = static_cast<int>(latentGrid_->getCommonEncoding());
		hiddenChannels -= latentGridChannelsDiv16 * 16;
	}

	int numHiddenLayers = static_cast<int>(hidden_.size()) - 1; //last layer with scalar or color output is handled separately
	if (!hasFourierFeatures) numHiddenLayers--; //special first layer from position
	if (hasVolumetricFeatures) numHiddenLayers--; //first layer is explicitly handled
	for (int i = 1; i < hidden_.size(); ++i)
		if (hidden_[i]->channelsIn != hiddenChannels)
			throw std::runtime_error("Currently, all hidden layers must have the same size");
	if (hiddenChannels % 16 != 0)
		throw std::runtime_error("Hidden channels must be a multiple of 16");
	if (numHiddenLayers<0)
		throw std::runtime_error("at least one hidden layer needed");

	auto activation = hidden_[0]->activation;
	for (int i = 1; i < hidden_.size() - 1; ++i)
		if (hidden_[i]->activation != activation)
			throw std::runtime_error("Currently, all hidden layers must have the same activation function");
	if (hidden_[hidden_.size() - 1]->activation != Layer::None)
		throw std::runtime_error("The last layer must have activation 'None'");

	int baseChannels = input_->hasDirection ? 8 : 4;
	if (hasFourierFeatures && input_->numFourierFeatures != ((hidden_[0]->channelsIn - baseChannels - latentGridChannelsDiv16*16) / 2))
		throw std::runtime_error(tinyformat::format(
			"If fourier features are defined, 2*num_fourier+%d==hidden[0].channelsIn() must hold. num_fourier=%d, channelsIn=%d",
			baseChannels, input_->numFourierFeatures, hidden_[0]->channelsIn));

	int directionMode = 0;
	if (input_->hasDirection)
		directionMode = input_->useDirectionInFourierFeatures ? 2 : 1;

	std::stringstream ss;
	ss << "#define BLOCK_SIZE " << (numWarps * 32) << "\n";
	ss << "#define NUM_HIDDEN_LAYERS " << numHiddenLayers << "\n";
	ss << "#define HIDDEN_CHANNELS_DIV16 " << (hiddenChannels / 16) << "\n";
	ss << "#define HAS_FOURIER_FEATURES " << (input_->numFourierFeatures > 0 ? 1 : 0) << "\n";
	ss << "#define NUM_FOURIER_FEATURES " << input_->numFourierFeatures << "\n";
	ss << "#define USE_DIRECTION " << directionMode << "\n";
	ss << "#define ACTIVATION " << Layer::ActivationNames[activation] << "\n";
	ss << "#define OUTPUT_MODE " << int(output_->outputMode) << "\n";
	ss << "#define FIRST_AND_LAST_IN_SHARED_MEMORY " << (firstAndLastInSharedMemory ? 1 : 0) << "\n";
	ss << "#define LATENT_GRID_CHANNELS_DIV16 " << latentGridChannelsDiv16 << "\n";
	ss << "#define LATENT_GRID_ENCODING " << latentGridEncoding << "\n";
	ss << "#define PASS_TIME_TO_NETWORK " << (input_->hasTime?1:0) << "\n";
	ss << "#define GRADIENT_MODE " << static_cast<int>(gradientMode) << "\n";
	//std::cout << "DEFINES:\n" << ss.str() << std::endl;
	cacheDefines_ = ss.str();
	return cacheDefines_;
}

std::vector<std::string> renderer::SceneNetwork::getIncludeFileNames(const IKernelModule::GlobalSettings& s) const
{
	return { "renderer_volume_tensorcores.cuh" };
}

std::string renderer::SceneNetwork::getConstantDeclarationName(const IKernelModule::GlobalSettings& s) const
{
	return "volumeInterpolationTensorcoresParameters";
}

std::string renderer::SceneNetwork::getPerThreadType(const IKernelModule::GlobalSettings& s) const
{
	return "::kernel::VolumeInterpolationTensorcores";
}

void renderer::SceneNetwork::fillConstantMemory(
	const IKernelModule::GlobalSettings& s, float fdStepsize, CUdeviceptr ptr,
	CUstream stream)
{
	if (!cacheConstantMemory_.empty() && fdStepsize==cacheFDStepsize_)
	{
		CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, cacheConstantMemory_.data(), cacheConstantMemory_.size(), stream));
		return;
	}
	cacheFDStepsize_ = fdStepsize;

	static std::vector<char> MEMORY(1024 * 1024);

	const bool hasFourierFeatures = input_->numFourierFeatures > 0;
	const bool hasDirection = input_->hasDirection;
	const bool hasColorOutput = output_->outputMode == OutputParametrization::RGBO || output_->outputMode == OutputParametrization::RGBO_DIRECT;
	
	size_t index = 0;
	const auto addWithPadding = [&](const void* mem, size_t len, int padding = 32)
	{
		//add padding
		index = kernel::roundUpPower2(index, padding);
		if (len > 0) {
			release_assert(index + len < MEMORY.size());
			memcpy(MEMORY.data() + index, mem, len);
			index += len;
		}
	};

	if (hasFourierFeatures)
	{
		addWithPadding(input_->fourierMatrix.data(), sizeof(half) * input_->fourierMatrix.size()); //cWeightsFourier
	}
	else
	{
		int C = hasDirection ? 6 : 3;
		release_assert(hidden_[0]->channelsIn == C);
		addWithPadding(hidden_[0]->weights.data(), sizeof(half) * C * hidden_[0]->channelsOut); //cWeightsFirst
		addWithPadding(hidden_[0]->bias.data(), sizeof(half) * hidden_[0]->channelsOut); //cBiasFirst
	}

#define DUMP_ARRAY_HALF(ax, count)	\
	do {printf(#ax ":"); for (int ii=0; ii<(count); ++ii) {printf(" %.2f", __half2float(ax[ii]));} printf("\n"); } while(0)
#define DUMP_ARRAY_INT(ax, count)	\
	do {printf(#ax ":"); for (int ii=0; ii<(count); ++ii) {printf(" %d", int(ax[ii]));} printf("\n"); } while(0)
#define DUMP_ARRAY_FLOAT4(ax, count)	\
	do {printf(#ax ":"); for (int ii=0; ii<(count); ++ii) {	\
		printf(" %.2f", ax[ii].x);	\
		printf(" %.2f", ax[ii].y);	\
		printf(" %.2f", ax[ii].z);	\
		printf(" %.2f", ax[ii].w);	\
	} printf("\n"); } while(0)

	bool hasLatentGrid = latentGrid() != nullptr;
	if (hasLatentGrid)
	{
		int gridTimeChannels = latentGrid()->getTimeChannels();
		int gridEnsembleChannels = latentGrid()->getEnsembleChannels();
		int gridTotalChannels = latentGrid()->getTotalChannels();
		int gridTotalChannelsDiv4 = gridTotalChannels / 4;
		bool hasOffsetScale = latentGrid()->getCommonEncoding() != LatentGrid::FLOAT;

		//memory to be stored in the constant buffer
		std::vector<cudaTextureObject_t> cLatentGridA(gridTotalChannelsDiv4);
		std::vector<cudaTextureObject_t> cLatentGridB(gridTotalChannelsDiv4);
		std::vector<float4> cLatentGridOffsetA(gridTotalChannelsDiv4);
		std::vector<float4> cLatentGridOffsetB(gridTotalChannelsDiv4);
		std::vector<float4> cLatentGridScaleA(gridTotalChannelsDiv4);
		std::vector<float4> cLatentGridScaleB(gridTotalChannelsDiv4);
		std::vector<float4> cLatentGridInterpolation(gridTotalChannelsDiv4);

		//time interpolation
		if (latentGrid()->hasTimeGrids()) {
			float time = latentGrid()->interpolateTime(currentTime_);
			int timeLow = std::min(static_cast<int>(time), latentGrid()->timeNum - 1);
			int timeHigh = std::min(timeLow + 1, latentGrid()->timeNum-1);
			LatentGrid_ptr gridLow = latentGrid()->getTimeGrid(timeLow);
			LatentGrid_ptr gridHigh = latentGrid()->getTimeGrid(timeHigh);
			gridLow->copyGridToGPU(true);
			gridHigh->copyGridToGPU(true);
			for (int i = 0; i < gridTimeChannels / 4; ++i)
			{
				cLatentGridA[i] = gridLow->getTexture(i);
				cLatentGridB[i] = gridHigh->getTexture(i);
				if (hasOffsetScale)
				{
					cLatentGridOffsetA[i] = gridLow->getOffsetOrMean(i);
					cLatentGridOffsetB[i] = gridHigh->getOffsetOrMean(i);
					cLatentGridScaleA[i] = gridLow->getScaleOrStd(i);
					cLatentGridScaleB[i] = gridHigh->getScaleOrStd(i);
				}
				cLatentGridInterpolation[i] = make_float4(time);
			}
		}

		//ensemble interpolation (currently: no interpolation)
		if (latentGrid()->hasEnsembleGrids()) {
			int ensemble = latentGrid()->interpolateEnsemble(currentEnsemble_);
			LatentGrid_ptr grid = latentGrid()->getEnsembleGrid(ensemble);
			grid->copyGridToGPU(true);
			int iOff = gridTimeChannels / 4;
			for (int i = 0; i < gridEnsembleChannels / 4; ++i)
			{
				cLatentGridA[i + iOff] = grid->getTexture(i);
				cLatentGridB[i + iOff] = grid->getTexture(i);
				if (hasOffsetScale)
				{
					cLatentGridOffsetA[i + iOff] = grid->getOffsetOrMean(i);
					cLatentGridOffsetB[i + iOff] = grid->getOffsetOrMean(i);
					cLatentGridScaleA[i + iOff] = grid->getScaleOrStd(i);
					cLatentGridScaleB[i + iOff] = grid->getScaleOrStd(i);
				}
				cLatentGridInterpolation[i + iOff] = make_float4(0);
			}
		}

		//copy to data
		addWithPadding(cLatentGridA.data(), cLatentGridA.size() * sizeof(cudaTextureObject_t));
		addWithPadding(cLatentGridB.data(), cLatentGridA.size() * sizeof(cudaTextureObject_t));
		if (hasOffsetScale)
		{
			addWithPadding(cLatentGridOffsetA.data(), cLatentGridOffsetA.size() * sizeof(float4));
			addWithPadding(cLatentGridOffsetB.data(), cLatentGridOffsetB.size() * sizeof(float4));
			addWithPadding(cLatentGridScaleA.data(), cLatentGridScaleA.size() * sizeof(float4));
			addWithPadding(cLatentGridScaleB.data(), cLatentGridScaleB.size() * sizeof(float4));
		}
		addWithPadding(cLatentGridInterpolation.data(), cLatentGridInterpolation.size() * sizeof(float4));

		//weights for the first layer
		auto layer = hidden_[0];
		int channelsIn = layer->channelsIn;
		int channelsOut = layer->channelsOut;
		addWithPadding(layer->weights.data(), sizeof(half) * channelsIn * channelsOut);
		addWithPadding(layer->bias.data(), sizeof(half) * channelsOut);

		//DUMP_ARRAY_HALF(layer->weights, channelsIn* channelsOut);
		//DUMP_ARRAY_HALF(layer->bias, channelsOut);
	}

	int startHidden = (hasFourierFeatures && !hasLatentGrid) ? 0 : 1;
	int numHidden = static_cast<int>(hidden_.size()) - startHidden - 1;
	if (numHidden < 0) throw std::runtime_error("Number of (real) hidden channels is negative. How did that happen?");
	int hiddenChannels = hasFourierFeatures ? hidden_[startHidden]->channelsIn : hidden_[startHidden]->channelsOut;
	std::vector<half> weightsHidden(max(1, numHidden * hiddenChannels * hiddenChannels));
	std::vector<half> biasHidden(max(1, numHidden * hiddenChannels));
	for (int i=startHidden; i<hidden_.size()-1; ++i)
	{
		memcpy(
			weightsHidden.data() + ((i - startHidden) * hiddenChannels * hiddenChannels),
			hidden_[i]->weights.data(),
			sizeof(half) * hiddenChannels * hiddenChannels);
		memcpy(
			biasHidden.data() + ((i - startHidden) * hiddenChannels),
			hidden_[i]->bias.data(),
			sizeof(half) * hiddenChannels);
	}
	addWithPadding(weightsHidden.data(), weightsHidden.size() * sizeof(half)); //cWeightsHidden
	addWithPadding(biasHidden.data(), biasHidden.size() * sizeof(half)); //cBiasHidden
	//DUMP_ARRAY_HALF(weightsHidden, weightsHidden.size());
	//DUMP_ARRAY_HALF(biasHidden, biasHidden.size());

	int lastHidden = static_cast<int>(hidden_.size()) - 1;
	addWithPadding(hidden_[lastHidden]->weights.data(), sizeof(half) * hidden_[lastHidden]->weights.size()); //cWeightsLast
	addWithPadding(hidden_[lastHidden]->bias.data(), sizeof(half) * hidden_[lastHidden]->bias.size()); //cBiasLiast

	addWithPadding(&boxMin_.x, sizeof(float3));
	addWithPadding(&boxSize_.x, sizeof(float3));

	float activationParameter = hidden_[0]->activationParameter;
	for (int i = 1; i < hidden_.size() - 1; ++i)
		if (hidden_[startHidden]->activationParameter != activationParameter)
			throw std::runtime_error("Extra parameter of the activation must be the same over all layers");
	addWithPadding(&activationParameter, sizeof(float));
	addWithPadding(&fdStepsize, sizeof(float)); //finiteDifferencesStepsize
	addWithPadding(&fdStepsize, sizeof(float)); //latentGridDifferencesStepSize

	addWithPadding(nullptr, 0); //pad whole struct
	//write out
	cacheConstantMemory_.resize(index);
	std::copy_n(MEMORY.begin(), index, cacheConstantMemory_.begin());
	CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, MEMORY.data(), index, stream));
}

renderer::VolumeInterpolationNetwork::VolumeInterpolationNetwork()
	: IVolumeInterpolation(false)
	, selectedNetwork_(0)
	, gradientMode_(SceneNetwork::GradientMode::OFF_OR_DIRECT)
	, finiteDifferencesStepsize_(1/256.f)
    , adjointLatentGridCentralDifferencesStepsizeScale_(4)
	, onlySharedMemory_(false)
{
}

void renderer::VolumeInterpolationNetwork::addNetwork(
	SceneNetwork_ptr network, const std::string& filename)
{
	int numWarpsSharedOnly = network->computeMaxWarps(true, false);
	int numWarpsMixed = network->computeMaxWarps(false, false);
	if (numWarpsSharedOnly < 0 && numWarpsMixed < 0)
		throw std::runtime_error("The network is too large!");
	networks_.push_back({
		network,
		numWarpsSharedOnly,
		numWarpsMixed,
		network->computeMaxWarps(true, true),
		network->computeMaxWarps(false, true),
		filename,
		filename.empty() ? "" : std::filesystem::path(filename).filename().string()
		});
	selectNetwork(networks_.size() - 1);
}

void renderer::VolumeInterpolationNetwork::selectNetwork(int index)
{
	selectedNetwork_ = index;
	auto net = networks_[index].network;
	//call the super-methods to not overwite the network settings
	IVolumeInterpolation::setBoxMin(make_double3(net->boxMin()));
	IVolumeInterpolation::setBoxMax(make_double3(net->boxSize()+net->boxMin()));
	if (net->latentGrid())
	{
		hasTimesteps_ = net->latentGrid()->hasTimeGrids();
		currentTimestep_ = hasTimesteps_
			? clamp(currentTimestep_, static_cast<float>(net->latentGrid()->timeMin), static_cast<float>(net->latentGrid()->timeMaxInclusive()))
			: 0;
		hasEnsembles_ = net->latentGrid()->hasEnsembleGrids();
		currentEnsemble_ = hasEnsembles_
			? clamp(currentEnsemble_, net->latentGrid()->ensembleMin, net->latentGrid()->ensembleMaxInclusive())
			: 0;
	} else
	{
		hasTimesteps_ = false;
		currentTimestep_ = 0;
		hasEnsembles_ = false;
		currentEnsemble_ = 0;
	}
}

void renderer::VolumeInterpolationNetwork::loadNetwork(const std::string& filename)
{
	std::ifstream in(filename, std::ofstream::binary);
	if (!in.is_open()) throw std::runtime_error("Unable to open the file");
	addNetwork(SceneNetwork::load(in), filename);
}

void renderer::VolumeInterpolationNetwork::setNetwork(SceneNetwork_ptr network)
{
	networks_.clear();
	addNetwork(network, "");
}

renderer::SceneNetwork_ptr renderer::VolumeInterpolationNetwork::currentNetwork() const
{
	if (networks_.empty() || selectedNetwork_ < 0)
		throw std::runtime_error("No network loaded");
	return networks_[selectedNetwork_].network;;
}

void renderer::VolumeInterpolationNetwork::setTimeAndEnsemble(float time, int ensemble)
{
	auto net = networks_[selectedNetwork_].network;
	net->setTimeAndEnsemble(time, ensemble);
}

void renderer::VolumeInterpolationNetwork::setBoxMin(const double3& box_min)
{
    IVolumeInterpolation::setBoxMin(box_min);
	for (auto& net : networks_)
	{
		net.network->setBoxMin(make_float3(boxMin()));
		net.network->setBoxSize(make_float3(boxSize()));
		net.network->clearGPUResources();
	}
}

void renderer::VolumeInterpolationNetwork::setBoxMax(const double3& box_max)
{
    IVolumeInterpolation::setBoxMax(box_max);
	for (auto& net : networks_)
	{
		net.network->setBoxMin(make_float3(boxMin()));
		net.network->setBoxSize(make_float3(boxSize()));
		net.network->clearGPUResources();
	}
}

std::string renderer::VolumeInterpolationNetwork::getName() const
{
	return "SRN";
}

void renderer::VolumeInterpolationNetwork::loadNetworkDialog()
{
	std::cout << "Open file dialog" << std::endl;

	// open file dialog
	auto results = pfd::open_file(
		"Load scene network",
		"",
		{ "Scene Networks", "*.volnet" },
		false
	).result();
	if (results.empty())
		return;
	std::string fileNameStr = results[0];

	//auto fileNamePath = std::filesystem::path(fileNameStr);
	//sceneNetworkDirectory_ = fileNamePath.string();
	//ImGui::MarkIniSettingsDirty();
	//ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);

	//load the file
	std::cout << "Load " << fileNameStr << std::endl;
	loadNetwork(fileNameStr);
}

bool renderer::VolumeInterpolationNetwork::drawUI(UIStorage_t& storage)
{
	bool changed = false;

	std::vector<const char*> networkNames(networks_.size());
	for (int i = 0; i < networks_.size(); ++i)
		networkNames[i] = networks_[i].humanname.c_str();
	if (ImGui::ListBox("", &selectedNetwork_, networkNames.data(), networks_.size()))
	{
		selectNetwork(selectedNetwork_);
		changed = true;
	}
	ImGui::SameLine();
	ImGui::BeginGroup();
	if (ImGui::Button(ICON_FA_FOLDER_OPEN "##Network"))
	{
		loadNetworkDialog();
		changed = true;
	}
	if (ImGui::ButtonEx(ICON_FA_MINUS "##Network", ImVec2(0, 0),
		networks_.empty() ? ImGuiButtonFlags_Disabled : 0))
	{
		networks_.erase(networks_.begin() + selectedNetwork_);
		selectNetwork(std::max(0, selectedNetwork_ - 1));
		changed = true;
	}
	ImGui::EndGroup();

	if (hasTimesteps_)
	{
		auto grid = networks_[selectedNetwork_].network->latentGrid();
		if (ImGui::SliderFloat("Time##Network", &currentTimestep_,
			static_cast<float>(grid->timeMin), static_cast<float>(grid->timeMaxInclusive())))
		{
			setTimeAndEnsemble(currentTimestep_, currentEnsemble_);
			changed = true;
		}
	}
	if (hasEnsembles_)
	{
		auto grid = networks_[selectedNetwork_].network->latentGrid();
		if (ImGui::SliderInt("Ensemble##Network", &currentEnsemble_,
			grid->ensembleMin, grid->ensembleMaxInclusive()))
		{
			setTimeAndEnsemble(currentTimestep_, currentEnsemble_);
			changed = true;
		}
	}

	if (ImGui::Checkbox("Use only shared memory", &onlySharedMemory_))
	{
		changed = true;
	}

	const auto currentGradientMode = magic_enum::enum_name(gradientMode_);
	if (ImGui::BeginCombo("Gradient Mode##Network", currentGradientMode.data()))
	{
		for (int i=0; i<magic_enum::enum_count<SceneNetwork::GradientMode>(); ++i)
		{
			SceneNetwork::GradientMode m = magic_enum::enum_value<SceneNetwork::GradientMode>(i);
			auto name = magic_enum::enum_name(m);
			bool selected = m == gradientMode_;
			if (ImGui::Selectable(name.data(), selected))
			{
				gradientMode_ = m;
				changed = true;
			}
			if (selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}
	if (gradientMode_ == SceneNetwork::GradientMode::FINITE_DIFFERENCES)
	{
		float divStepsize = 1 / finiteDifferencesStepsize_;
		if (ImGui::SliderFloat("1 / FD Stepsize##Network", &divStepsize, 32, 1024, "%.1f", 2))
		{
			finiteDifferencesStepsize_ = 1 / divStepsize;
			changed = true;
		}
	}

	if (!networks_.empty())
	{
		auto net = networks_[selectedNetwork_].network;
		std::stringstream layers;
		for (int i = 0; i < net->numLayers(); ++i)
		{
			if (i == 0) layers << net->getHidden(i)->channelsIn;
			layers << ":" << net->getHidden(i)->channelsOut;
		}
		std::string layerStr = layers.str();
		bool isAdjoint = gradientMode_ == SceneNetwork::GradientMode::ADJOINT_METHOD;
		ImGui::Text("Input: direction=%d, #fourier=%d\nOutput: %s\nLayers: %s (%dB)\nWarps: %d / %d",
			net->input()->hasDirection ? 1 : 0,
			net->input()->numFourierFeatures,
			OutputParametrization::OutputModeNames[net->output()->outputMode].c_str(),
			layerStr.c_str(), net->numParameters()*2 /*sizeof(half)*/,
			isAdjoint ? networks_[selectedNetwork_].numWarpsMixedAdjoint : networks_[selectedNetwork_].numWarpsMixed,
			isAdjoint ? networks_[selectedNetwork_].numWarpsSharedOnlyAdjoint : networks_[selectedNetwork_].numWarpsSharedOnly);
		if (net->latentGrid())
		{
			ImGui::Text("Grid: res=%d^3, channels=%d",
				net->latentGrid()->getResolution(), net->latentGrid()->getTotalChannels());
			ImGui::Text("Time keyframes: %d, ensembles: %d",
				net->latentGrid()->timeNum, net->latentGrid()->ensembleNum);
		}
	}

	return changed;
}

void renderer::VolumeInterpolationNetwork::load(const nlohmann::json& json, const ILoadingContext* fetcher)
{
	gradientMode_ = magic_enum::enum_cast<SceneNetwork::GradientMode>(json.value("gradientMode", ""))
		.value_or(SceneNetwork::GradientMode::OFF_OR_DIRECT);
	finiteDifferencesStepsize_ = json.value("finiteDifferencesStepsize", 1 / 256.f);

	//load the networks
	//TODO
}

void renderer::VolumeInterpolationNetwork::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["gradientMode"] = magic_enum::enum_name(gradientMode_);
	json["finiteDifferencesStepsize"] = finiteDifferencesStepsize_;

	//save the networks
	//TODO
}

int renderer::VolumeInterpolationNetwork::getCurrentNumWarps() const
{
	if (networks_.empty())
		throw std::runtime_error("No network specified!");
	const auto& net = networks_[selectedNetwork_];
	bool isAdjoint = gradientMode_ == SceneNetwork::GradientMode::ADJOINT_METHOD;
	int warpsSharedOnly = isAdjoint ? net.numWarpsSharedOnlyAdjoint : net.numWarpsSharedOnly;
	int warpsMixed = isAdjoint ? net.numWarpsMixedAdjoint : net.numWarpsMixed;
	if (warpsMixed > 0 && warpsSharedOnly > 0)
		return onlySharedMemory_ ? warpsSharedOnly : warpsMixed;
	return warpsMixed;
}

void renderer::VolumeInterpolationNetwork::prepareRendering(GlobalSettings& s) const
{
	IVolumeInterpolation::prepareRendering(s);
	if (networks_.empty())
		throw std::runtime_error("No network specified!");

	s.synchronizedThreads = true;
	const auto& net = networks_[selectedNetwork_];

	bool isAdjoint = gradientMode_ == SceneNetwork::GradientMode::ADJOINT_METHOD;
	int warpsSharedOnly = isAdjoint ? net.numWarpsSharedOnlyAdjoint : net.numWarpsSharedOnly;
	int warpsMixed = isAdjoint ? net.numWarpsMixedAdjoint : net.numWarpsMixed;

	if (warpsMixed > 0 && warpsSharedOnly > 0) {
		currentNumWarps_ = onlySharedMemory_ ? warpsSharedOnly : warpsMixed;
		currentOnlyShared_ = onlySharedMemory_;
	}
	else
	{
		if (warpsMixed <= 0)
			throw std::runtime_error("Network is too large!");
		currentNumWarps_ = warpsMixed;
		currentOnlyShared_ = false;
	}
	currentNumWarps_ = min(currentNumWarps_, MAX_BLOCK_SIZE / 32);
	currentTargetBlockSize_ = currentNumWarps_ * 32;

	if (s.fixedBlockSize > 0 && s.fixedBlockSize != currentTargetBlockSize_)
		throw std::runtime_error("Another module already requested a different, fixed block size");
	s.fixedBlockSize = currentTargetBlockSize_;
}

renderer::IKernelModule::GlobalSettings::VolumeOutput renderer::VolumeInterpolationNetwork::outputType() const
{
	if (networks_.empty())
		throw std::runtime_error("No network specified!");
	const auto& net = networks_[selectedNetwork_];
	int c = net.network->output()->channelsOut();
	for (int i=0; i<3; ++i)
	{
		if (OutputType2ChannelCount[i] == c)
			return static_cast<GlobalSettings::VolumeOutput>(i);
	}
	throw std::runtime_error("Unknown number of output channels, not compatible to one of the pre-defined output types");
}

std::optional<int> renderer::VolumeInterpolationNetwork::getBatches(const GlobalSettings& s) const
{
	return {};
}

std::string renderer::VolumeInterpolationNetwork::getDefines(const GlobalSettings& s) const
{
	if (networks_.empty())
		throw std::runtime_error("No network specified, can't render!");
	const auto& net = networks_[selectedNetwork_];
	
	if (s.volumeShouldProvideNormals) {
		switch (gradientMode_)
		{
		case SceneNetwork::GradientMode::OFF_OR_DIRECT:
			if (!net.network->supportsNormals())
				std::cerr << "WARNING: normals requested, but the SceneNetwork can't directly provide those, or FD/AD gradients are disabled" << std::endl;
			break;
		case SceneNetwork::GradientMode::FINITE_DIFFERENCES:
		case SceneNetwork::GradientMode::ADJOINT_METHOD:
			if (net.network->output()->isColor())
				std::cerr << "WARNING: normales requested, but color-predicting networks can't provide those" << std::endl;
			break;
		}
	}
	if (s.volumeShouldProvideCurvature)
	{
		if (!net.network->output()->isDensityCurvature())
		{
			throw std::runtime_error("curvature can only be provided for network output mode 'densitycurvature");
		}
		if (gradientMode_ != SceneNetwork::GradientMode::OFF_OR_DIRECT)
		{
			throw std::runtime_error("curvature can only be provided directly be the network. The Adjoint-method or finite differences is not supported yet.");
		}
	}

	return net.network->getDefines(s, currentNumWarps_, currentOnlyShared_, gradientMode_);
}

std::vector<std::string> renderer::VolumeInterpolationNetwork::getIncludeFileNames(const GlobalSettings& s) const
{
	if (networks_.empty())
		throw std::runtime_error("No network specified, can't render!");
	return networks_[selectedNetwork_].network->getIncludeFileNames(s);
}

std::string renderer::VolumeInterpolationNetwork::getConstantDeclarationName(const GlobalSettings& s) const
{
	if (networks_.empty())
		throw std::runtime_error("No network specified, can't render!");
	return networks_[selectedNetwork_].network->getConstantDeclarationName(s);
}

std::string renderer::VolumeInterpolationNetwork::getPerThreadType(const GlobalSettings& s) const
{
	if (networks_.empty())
		throw std::runtime_error("No network specified, can't render!");
	return networks_[selectedNetwork_].network->getPerThreadType(s);
}

void renderer::VolumeInterpolationNetwork::fillConstantMemory(const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
	if (networks_.empty()) 
		throw std::runtime_error("No network specified, can't render!");
	auto net = networks_[selectedNetwork_].network;
	float stepsize = finiteDifferencesStepsize_;
	if (gradientMode_ == SceneNetwork::GradientMode::ADJOINT_METHOD) {
		int gridResolution = net->latentGrid() ? net->latentGrid()->getResolution() : 1;
		stepsize = 1 / (static_cast<float>(gridResolution) * adjointLatentGridCentralDifferencesStepsizeScale_);
	}
	net->fillConstantMemory(
		s, stepsize, ptr, stream);
}

void renderer::VolumeInterpolationNetwork::registerPybindModule(pybind11::module& m)
{
	IVolumeInterpolation::registerPybindModule(m);

	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;

	namespace py = pybind11;

	py::class_<SceneNetwork, SceneNetwork_ptr> sn(m, "SceneNetwork");

	py::class_<InputParametrization, InputParametrization_ptr>(sn, "InputParametrization")
		.def_readwrite("has_time", &InputParametrization::hasTime)
		.def_readwrite("has_direction", &InputParametrization::hasDirection)
		.def("num_fourier_features", [](InputParametrization* p) {return p->numFourierFeatures; })
		.def("set_fourier_matrix_from_tensor", &InputParametrization::setFourierMatrixFromTensor)
		.def("disable_fourier_features", &InputParametrization::disableFourierFeatures)
		.def("channels_out", &InputParametrization::channelsOut)
		.def("valid", &InputParametrization::valid)
		;

	py::class_<OutputParametrization, OutputParametrization_ptr> om(sn, "OutputParametrization");
	py::enum_<OutputParametrization::OutputMode>(om, "OutputMode")
		.value("DENSITY", OutputParametrization::OutputMode::DENSITY)
		.value("DENSITY_DIRECT", OutputParametrization::OutputMode::DENSITY_DIRECT)
		.value("RGBO", OutputParametrization::OutputMode::RGBO)
		.value("RGBO_DIRECT", OutputParametrization::OutputMode::RGBO_DIRECT)
		.export_values()
		;
	om.def_readwrite("output_mode", &OutputParametrization::outputMode)
		.def_static("OutputModeFromString", &OutputParametrization::OutputModeFromString)
		.def("channels_in", &OutputParametrization::channelsIn)
		;

	py::class_<Layer, Layer_ptr> l(sn, "Layer");
	py::enum_<Layer::Activation>(l, "Activation")
		.value("ReLU", Layer::Activation::ReLU)
		.value("Sine", Layer::Activation::Sine)
		.value("Snake", Layer::Activation::Snake)
		.value("SnakeAlt", Layer::Activation::SnakeAlt)
		.value("Sigmoid", Layer::Activation::Sigmoid)
		.value("NONE", Layer::Activation::None)
		.export_values()
		;
	l.def_readwrite("activation", &Layer::activation)
		.def_static("ActivationFromString", &Layer::ActivationFromString)
		.def_readonly("channels_in", &Layer::channelsIn)
		.def_readonly("channels_out", &Layer::channelsOut)
		.def("valid", &Layer::valid)
		;

	py::class_<LatentGrid, LatentGrid_ptr> lg(sn, "LatentGrid");
	py::enum_<LatentGrid::Encoding>(lg, "Encoding")
		.value("Float", LatentGrid::Encoding::FLOAT)
		.value("ByteLinear", LatentGrid::Encoding::BYTE_LINEAR)
		.value("ByteGaussian", LatentGrid::Encoding::BYTE_GAUSSIAN)
		.export_values();
	lg.def(py::init<>())
		.def(py::init<const torch::Tensor&, LatentGrid::Encoding>())
		.def("is_valid", &LatentGrid::isValid)
		.def_readonly("grid_channels", &LatentGrid::gridChannels)
		.def_readonly("grid_size_z", &LatentGrid::gridSizeZ)
		.def_readonly("grid_size_y", &LatentGrid::gridSizeY)
		.def_readonly("grid_size_z", &LatentGrid::gridSizeX)
		.def_readonly("encoding", &LatentGrid::encoding)
		;

	py::class_<LatentGridTimeAndEnsemble, LatentGridTimeAndEnsemble_ptr>(sn, "LatentGridTimeAndEnsemble")
		.def(py::init<>())
		.def(py::init<int, int, int, int, int>(), py::doc(R"(
			Constructs a new container for time- and ensemble dependent Latent Grids.
		   
			Ranges for the time keyframes are specified as timeMin, timeNum, timeStep.
			The minimal timestep is 'timeMin', the maximal timestep is 'timeMin+(timeNum-1)*timeStep'.
			Allowed time values are 't in [timeMin, timeMin+(timeNum-1)*timeStep]',
			keyframes are where '(t-timeMin)%timeStep==0', linear interpolation in between.
			
			Ensembles are specified via ensembleMin, ensembleNum.
			)"), py::arg("time_min"), py::arg("time_num"), py::arg("time_step"), 
			py::arg("ensemble_min"), py::arg("ensemble_num"))
		.def_readonly("time_min", &LatentGridTimeAndEnsemble::timeMin)
		.def_readonly("time_num", &LatentGridTimeAndEnsemble::timeNum)
		.def_readonly("time_step", &LatentGridTimeAndEnsemble::timeStep)
		.def_readonly("ensemble_min", &LatentGridTimeAndEnsemble::ensembleMin)
		.def_readonly("ensemble_num", &LatentGridTimeAndEnsemble::ensembleNum)
		.def_property_readonly("time_max_inclusive", &LatentGridTimeAndEnsemble::timeMaxInclusive)
		.def_property_readonly("ensemble_max_inclusive", &LatentGridTimeAndEnsemble::ensembleMaxInclusive)
		.def("interpolate_time", &LatentGridTimeAndEnsemble::interpolateTime,
			py::doc("Computes the interpolation index into the list of time grids based on the given absolute time in [timeMin, timeMin+timeNum-1]"),
			py::arg("time"))
		.def("interpolate_ensemble", &LatentGridTimeAndEnsemble::interpolateEnsemble,
			py::doc("Computes the interpolation index into the list of ensemble grids based on the given absolute time in [ensembleMin, ensembleMin+ensembleNum-1]"),
			py::arg("time"))
		.def("get_time_grid", static_cast<LatentGrid_ptr(LatentGridTimeAndEnsemble::*)(int)>(&LatentGridTimeAndEnsemble::getTimeGrid),
			py::doc("Returns the time grid at the specified integer index"), py::arg("index"))
		.def("get_ensemble_grid", static_cast<LatentGrid_ptr(LatentGridTimeAndEnsemble::*)(int)>(&LatentGridTimeAndEnsemble::getEnsembleGrid),
			py::doc("Returns the ensemble grid at the specified integer index"), py::arg("index"))
		.def("set_time_grid_from_torch", &LatentGridTimeAndEnsemble::setTimeGridFromTorch,
			py::doc("Sets the grid at the given index from the pytorch tensor."),
			py::arg("index"), py::arg("tensor"), py::arg("encoding"))
		.def("set_ensemble_grid_from_torch", &LatentGridTimeAndEnsemble::setEnsembleGridFromTorch,
			py::doc("Sets the grid at the given index from the pytorch tensor."),
			py::arg("index"), py::arg("tensor"), py::arg("encoding"))
		.def("is_valid", &LatentGridTimeAndEnsemble::isValid)
		.def("common_encoding", &LatentGridTimeAndEnsemble::getCommonEncoding)
		.def("time_channels", &LatentGridTimeAndEnsemble::getTimeChannels)
		.def("ensemble_channels", &LatentGridTimeAndEnsemble::getEnsembleChannels)
	;

	sn.def(py::init<>())
		.def_property_readonly("input", [](SceneNetwork& n) {return n.input(); })
		.def_property_readonly("output", [](SceneNetwork& n) {return n.output(); })
		.def_property("latent_grid",
			static_cast<LatentGridTimeAndEnsemble_ptr(SceneNetwork::*)()>(&SceneNetwork::latentGrid),
			&SceneNetwork::setLatentGrid,
			py::doc("read or write the latent grid"))
		.def("add_layer", &SceneNetwork::addLayerFromTorch,
			py::arg("weights"), py::arg("bias"), py::arg("activation"), py::arg("activation_parameter")=1.0f)
		.def("num_layers", &SceneNetwork::numLayers)
		.def("get_layer", static_cast<Layer_ptr(SceneNetwork::*)(int index)>(&SceneNetwork::getHidden))
		.def_property("box_min", &SceneNetwork::boxMin, &SceneNetwork::setBoxMin)
		.def_property("box_size", &SceneNetwork::boxSize, &SceneNetwork::setBoxSize)
		.def("valid", &SceneNetwork::valid)
		.def("save", [](const SceneNetwork& n, const std::string& filename)
			{
				std::ofstream out(filename, std::ofstream::binary);
				n.save(out);
			})
		.def_static("load", [](const std::string& filename)
			{
				std::ifstream in(filename, std::ifstream::binary);
				return SceneNetwork::load(in);
			})
		.def("num_parameters", &SceneNetwork::numParameters)
		.def("compute_max_warps", &SceneNetwork::computeMaxWarps,
					py::doc(R"(
				 Computes the maximal number of warps that are possible
				 if this network is evaluated using tensor cores.
				 The number of warps is limited by the available shared memory.
				 This method returns a negative number if the network is too big.
			)"), py::arg("only_shared_memory"), py::arg("adjoint")=false)
		.def("clear_gpu_resources", &SceneNetwork::clearGPUResources, py::doc(R"(
				It is assumed that the network is fully configured before the first render.
				Before rendering it for the first time, GPU resources and settings are automatically
				allocated and cached.
				To clear those GPU resources (in case you do want to edit it afterwards), call
				this method.
			)"))
		.def("set_time_and_ensemble", &SceneNetwork::setTimeAndEnsemble, py::doc(R"(
	Sets the time and ensemble to be used for rendering.
	See time_min, time_max_inclusive, ensemble_min, ensemble_max_inclusive of 
	LatentGridTimeAndEnsemble for the allowed bounds)"),
			py::arg("time"), py::arg("ensemble"))
		;

		py::class_<VolumeInterpolationNetwork, IVolumeInterpolation, std::shared_ptr<VolumeInterpolationNetwork>> n(m, "VolumeInterpolationNetwork");
		py::enum_<SceneNetwork::GradientMode>(n, "GradientMode")
			.value("OFF_OR_DIRECT", SceneNetwork::GradientMode::OFF_OR_DIRECT, R"(
	OutputParametrization::DENSITY or ::COLOR -> disable gradient.
	OutputParametrization::DENSITY_GRADIENT -> use direct prediction gradient )")
			.value("FINITE_DIFFERENCES", SceneNetwork::GradientMode::FINITE_DIFFERENCES,
				"Use Finite Differences, see finite_differences_stepsize")
			.value("ADJOINT_METHOD", SceneNetwork::GradientMode::ADJOINT_METHOD,
				"Use the adjoint method")
		;
		n.def(py::init<>())
			.def("set_network", &VolumeInterpolationNetwork::setNetwork)
	        .def("current_network", &VolumeInterpolationNetwork::currentNetwork)
			.def_readwrite("only_shared_memory", &VolumeInterpolationNetwork::onlySharedMemory_)
			.def_property("gradient_mode", 
				&VolumeInterpolationNetwork::gradientMode,
				&VolumeInterpolationNetwork::setGradientMode)
			.def_property("finite_differences_stepsize",
				&VolumeInterpolationNetwork::finiteDifferencesStepsize,
				&VolumeInterpolationNetwork::setFiniteDifferencesStepsize,
				py::doc(R"(
	The stepsize for finite-difference gradient computation.
	Note: don't set it too low or you'll get massive cancellation errors due to the half-precision evaluation.)"))
	        .def_readwrite("adjoint_latent_grid_central_differences_stepsize_scale", &VolumeInterpolationNetwork::adjointLatentGridCentralDifferencesStepsizeScale_,
				py::doc(R"(
	In the adjoint method, the gradients through the latent grid sampling
	has still to be approximated via central differences.
	This approximation is perfect for an infinite small stepsize, if one ignores rounding errors.
	The final stepsize is set to
	 1/(latentGridResolution * adjointLatentGridCentralDifferencesStepsizeScale_)
            )"))
		;
	
}
