#include "volume.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cuMat/src/Errors.h>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <lz4cpp.hpp>
#include <string.h>
#include <json.hpp>
#include <tinyformat.h>
#include <magic_enum.hpp>

#include "imodule.h"
//#include "../third-party/pybind11/include/pybind11/pybind11.h"
#include "halton_sampler.h"
#include "errors.h"
#include "pytorch_utils.h"

BEGIN_RENDERER_NAMESPACE

static void printProgress(const std::string& prefix, float progress)
{
	static std::chrono::steady_clock::time_point lastTime;
	const auto currentTime = std::chrono::steady_clock::now();
	if (progress>=1 || std::chrono::duration_cast<std::chrono::milliseconds>(currentTime-lastTime).count()>100)
	{
		lastTime = currentTime;
		int barWidth = 50;
		std::cout << prefix << " [";
		int pos = static_cast<int>(barWidth * progress);
		for (int i = 0; i < barWidth; ++i) {
			if (i < pos) std::cout << "=";
			else if (i == pos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "] " << int(progress * 100.0) << " %\r";
		std::cout.flush();
		if (progress >= 1) std::cout << std::endl;
	}
}

const int Volume::BytesPerType[Volume::_TypeCount_] = {
	1, 2, 4 
};
static const cudaChannelFormatKind FormatPerType[Volume::_TypeCount_] = {
	cudaChannelFormatKindUnsigned,
	cudaChannelFormatKindUnsigned,
	cudaChannelFormatKindFloat
};

Volume::MipmapLevel::MipmapLevel(Feature* parent, uint64_t sizeX, uint64_t sizeY, uint64_t sizeZ)
	: channels_(parent->numChannels())
    , sizeX_(sizeX)
    , sizeY_(sizeY)
	, sizeZ_(sizeZ), dataCpu_(new char[parent->numChannels() * sizeX * sizeY * sizeZ * BytesPerType[parent->type()]]), dataGpu_(nullptr)
	, dataTexLinear_(0), dataTexNearest_(0)
	, cpuDataCounter_(0)
	, gpuDataCounter_(0)
	, parent_(parent)
{
}

Volume::MipmapLevel::~MipmapLevel()
{
	clearGpuResources();
	delete[] dataCpu_;
}



bool Volume::MipmapLevel::checkHasGpu() const
{
	if (!dataGpu_)
	{
		std::cerr << "GPU resources not initialized. Did you forget to call copyCpuToGpu()?" << std::endl;
		return false;
	}
	return true;
}

cudaArray_const_t Volume::MipmapLevel::dataGpu() const
{
	if (!checkHasGpu()) return nullptr;
    return dataGpu_;
}

cudaTextureObject_t Volume::MipmapLevel::dataTexGpuLinear() const
{
	if (!checkHasGpu()) return 0;
    return dataTexLinear_;
}

cudaTextureObject_t Volume::MipmapLevel::dataTexGpuNearest() const
{
	if (!checkHasGpu()) return 0;
    return dataTexNearest_;
}

void Volume::MipmapLevel::copyCpuToGpu()
{
	if (channels_ != 1 && channels_ != 2 && channels_ != 4)
		throw std::runtime_error("Only 1,2, or 4 channels are supported on the GPU");

	if (gpuDataCounter_ == cpuDataCounter_ && dataGpu_)
		return; //nothing changed
	gpuDataCounter_ = cpuDataCounter_;

	//create array
	cudaExtent extent = make_cudaExtent(sizeX_, sizeY_, sizeZ_);
	if (!dataGpu_) {
	    int bitsPerType = 8 * BytesPerType[parent_->type()];
	    auto format = FormatPerType[parent_->type()];
	    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(
		    bitsPerType,
		    (channels_ >= 2) ? bitsPerType : 0,
		    (channels_ >= 4) ? bitsPerType : 0,
		    (channels_ >= 4) ? bitsPerType : 0,
		    format);
		CUMAT_SAFE_CALL(cudaMalloc3DArray(&dataGpu_, &channelDesc, extent));
		//std::cout << "Cuda array allocated" << std::endl;
	}
	cudaMemcpy3DParms params = { 0 };
	params.srcPtr = make_cudaPitchedPtr(dataCpu_,
		BytesPerType[parent_->type()] * sizeX_ * channels_, sizeX_, sizeY_);
	params.dstArray = dataGpu_;
	params.extent = extent;
	params.kind = cudaMemcpyHostToDevice;
	CUMAT_SAFE_CALL(cudaMemcpy3D(&params));

	//create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(cudaResourceDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = dataGpu_;
	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = parent_->type() == TypeFloat ? cudaReadModeElementType : cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 0;
	if (dataTexLinear_ == 0)
		CUMAT_SAFE_CALL(cudaDestroyTextureObject(dataTexLinear_));
	dataTexLinear_ = 0;
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&dataTexLinear_, &resDesc, &texDesc, NULL));

	texDesc.filterMode = cudaFilterModePoint;
	if (dataTexNearest_ == 0)
		CUMAT_SAFE_CALL(cudaDestroyTextureObject(dataTexNearest_));
	dataTexNearest_ = 0;
	CUMAT_SAFE_CALL(cudaCreateTextureObject(&dataTexNearest_, &resDesc, &texDesc, NULL));
}

void Volume::MipmapLevel::clearGpuResources()
{

	if (dataTexLinear_ != 0) {
		CUMAT_SAFE_CALL_NO_THROW(cudaDestroyTextureObject(dataTexLinear_));
		dataTexLinear_ = 0;
	}
	if (dataTexNearest_ != 0) {
		CUMAT_SAFE_CALL_NO_THROW(cudaDestroyTextureObject(dataTexNearest_));
		dataTexNearest_ = 0;
	}
	if (dataGpu_ != nullptr) {
		CUMAT_SAFE_CALL_NO_THROW(cudaFreeArray(dataGpu_));
		dataGpu_ = nullptr;
	}
}

template<typename T>
torch::Tensor ToTensor(const Volume::MipmapLevel* l, float scale)
{
	const T* data = l->dataCpu<T>();
	const int C = l->channels();
	const int X = l->sizeX();
	const int Y = l->sizeY();
	const int Z = l->sizeZ();
	torch::Tensor t = torch::zeros({ C, X, Y, Z }, 
		at::TensorOptions().dtype(c10::kFloat));
	auto acc = t.accessor<float, 4>();
	for (int z = 0; z < Z; ++z) for (int y = 0; y < Y; ++y) for (int x = 0; x < X; ++x) for (int c = 0; c < C; ++c)
		acc[c][x][y][z] = static_cast<float>(data[l->idx(x, y, z, c)]) * scale;
	return t;
}

torch::Tensor Volume::MipmapLevel::toTensor() const
{
	std::cout << "toTensor() with type()=" << magic_enum::enum_name(type()) << std::endl;
	switch (type())
	{
	case TypeUChar:
		return ToTensor<unsigned char>(this, 1.0f / 0xff);
	case TypeUShort:
		return ToTensor<unsigned short>(this, 1.0f /0xffff);
	case TypeFloat:
		return ToTensor<float>(this, 1);
	default:
		throw std::runtime_error("Unknown enum constant");
	}
}

template<typename T>
void FromTensor(Volume::MipmapLevel* l, const torch::Tensor& t, float scale)
{
	T* data = l->dataCpu<T>();
	const int C = l->channels();
	const int X = l->sizeX();
	const int Y = l->sizeY();
	const int Z = l->sizeZ();
	const auto acc = t.packed_accessor64<float, 4>();
	for (int z = 0; z < Z; ++z) for (int y = 0; y < Y; ++y) for (int x = 0; x < X; ++x) for (int c = 0; c < C; ++c)
		data[l->idx(x, y, z, c)] = static_cast<T>(acc[c][x][y][z] * scale);
}

void Volume::MipmapLevel::fromTensor(const torch::Tensor& t)
{
	TORCH_CHECK(t.device()==c10::kCPU, "The input tensor must reside on the CPU");
	TORCH_CHECK(t.dtype() == c10::kFloat, "The input tensor must be of type 'float'");
	CHECK_DIM(t, 4);
	CHECK_SIZE(t, 0, channels());
	CHECK_SIZE(t, 1, sizeX());
	CHECK_SIZE(t, 2, sizeY());
	CHECK_SIZE(t, 3, sizeZ());

	switch (type())
	{
	case TypeUChar:
		FromTensor<unsigned char>(this, t, 0xff);
		break;
	case TypeUShort:
		FromTensor<unsigned short>(this, t, 0xffff);
		break;
	case TypeFloat:
		FromTensor<float>(this, t, 1);
		break;
	default:
		throw std::runtime_error("Unknown enum constant");
	}
}

Volume::Feature::Feature(Volume* parent, const std::string& name, DataType type, int numChannels, uint64_t sizeX,
                         uint64_t sizeY, uint64_t sizeZ) :
    name_(name), type_(type), numChannels_(numChannels), parent_(parent)
{
	levels_.push_back(std::make_unique<MipmapLevel>(this, sizeX, sizeY, sizeZ));
}

std::shared_ptr<Volume::Feature> Volume::Feature::load(
	Volume* parent,
	std::ifstream& s, LZ4Decompressor* compressor,
	const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging,
	const VolumeErrorCallback_t& error)
{
	int lenName;
	std::string name;
	uint64_t sizeX, sizeY, sizeZ;
	int channels;
	int type;

	s.read(reinterpret_cast<char*>(&lenName), 4);
	name.resize(lenName);
	s.read(name.data(), lenName);
	s.read(reinterpret_cast<char*>(&sizeX), 8);
	s.read(reinterpret_cast<char*>(&sizeY), 8);
	s.read(reinterpret_cast<char*>(&sizeZ), 8);
	s.read(reinterpret_cast<char*>(&channels), 4);
	s.read(reinterpret_cast<char*>(&type), 4);

	Feature_ptr f = std::make_shared<Feature>(parent, name, static_cast<DataType>(type), channels, sizeX, sizeY, sizeZ);
	auto data = f->getLevel(0);
	bool useCompression = compressor != nullptr;

	//body
	progress(0.0f);
	if (useCompression) {
		size_t dataToRead = BytesPerType[type] * sizeX * sizeY * sizeZ * channels;
		for (size_t offset = 0; offset < dataToRead;)
		{
			char* mem = data->dataCpu<char>() + offset;
			const int len = std::min(
				static_cast<int>(dataToRead - offset),
				std::numeric_limits<int>::max());
			int chunkSize = compressor->decompress(mem, len, s);
			progress(offset / float(dataToRead));
			offset += chunkSize;
		}
	}
	else
	{
		size_t dataToRead = BytesPerType[type] * sizeX * sizeY * channels;
		for (int z = 0; z < sizeZ; ++z)
		{
			s.read(data->dataCpu<char>() + z * dataToRead, dataToRead);
			if (z % 10 == 0)
				progress(z / float(sizeZ));
		}
	}
	progress(1.0f);

	return f;
}

void Volume::Feature::save(
	std::ofstream& s, LZ4Compressor* compressor,
	const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging,
	const VolumeErrorCallback_t& error)
{
	bool useCompression = compressor != nullptr;
	const auto data = getLevel(0);

	//header
	int lenName = name_.size();
	uint64_t sizeX = getLevel(0)->sizeX();
	uint64_t sizeY = getLevel(0)->sizeY();
	uint64_t sizeZ = getLevel(0)->sizeZ();
	int numChannels = numChannels_;
	int type = static_cast<int>(type_);

	s.write(reinterpret_cast<const char*>(&lenName), 4);
	s.write(name_.c_str(), lenName);
	s.write(reinterpret_cast<const char*>(&sizeX), 8);
	s.write(reinterpret_cast<const char*>(&sizeY), 8);
	s.write(reinterpret_cast<const char*>(&sizeZ), 8);
	s.write(reinterpret_cast<const char*>(&numChannels), 4);
	s.write(reinterpret_cast<const char*>(&type), 4);

	//body
	progress(0.0f);
	if (useCompression)
	{
		size_t dataToWrite = BytesPerType[type_] * data->sizeX_ * data->sizeY_ * data->sizeZ_ * numChannels_;
		int chunkSize = LZ4Compressor::MAX_CHUNK_SIZE;
		for (size_t offset = 0; offset < dataToWrite; offset += chunkSize)
		{
			const char* mem = data->dataCpu_ + offset;
			const int len = std::min(static_cast<int>(dataToWrite - offset), chunkSize);
			compressor->compress(s, mem, len);
			progress(offset / float(dataToWrite));
		}
	}
	else
	{
		size_t dataToWrite = BytesPerType[type_] * data->sizeX_ * data->sizeY_ * numChannels_;
		for (int z = 0; z < data->sizeZ_; ++z)
		{
			s.write(data->dataCpu_ + z * dataToWrite, dataToWrite);
			if (z % 10 == 0)
				progress(z / float(data->sizeZ_));
		}
	}
	progress(1.0f);
}

namespace
{
	//copied and adapted from Pytorch: ATen/native/AdaptiveAveragePooling3d.cpp

	inline int start_index(int a, int b, int c) {
		return (int)floor((float)(a * c) / b);
	}

	inline int end_index(int a, int b, int c) {
		return (int)ceil((float)((a + 1) * c) / b);
	}

	template<typename T>
	void adaptive_avg_pool3d(const Volume::MipmapLevel* in, Volume::MipmapLevel* out)
	{
		const T* dataIn = in->dataCpu<T>();
		T* dataOut = out->dataCpu<T>();
		//fetch sizes
		const int inSizeX = static_cast<int>(in->sizeX());
		const int inSizeY = static_cast<int>(in->sizeY());
		const int inSizeZ = static_cast<int>(in->sizeZ());
		const int outSizeX = static_cast<int>(out->sizeX());
		const int outSizeY = static_cast<int>(out->sizeY());
		const int outSizeZ = static_cast<int>(out->sizeZ());
		//loop over output
		const int numChannels = in->channels();
#pragma omp parallel for collapse(2)
		for (int c=0; c<numChannels; ++c)
		for (int oz = 0; oz < outSizeZ; ++oz)
		{
			const int iStartZ = start_index(oz, outSizeZ, inSizeZ);
			const int iEndZ = end_index(oz, outSizeZ, inSizeZ);
			const int kZ = iEndZ - iStartZ;
			for (int oy = 0; oy < outSizeY; ++oy)
			{
				const int iStartY = start_index(oy, outSizeY, inSizeY);
				const int iEndY = end_index(oy, outSizeY, inSizeY);
				const int kY = iEndY - iStartY;
				for (int ox = 0; ox < outSizeX; ++ox)
				{
					const int iStartX = start_index(ox, outSizeX, inSizeX);
					const int iEndX = end_index(ox, outSizeX, inSizeX);
					const int kX = iEndX - iStartX;

					//compute local average
					float sum = 0;
					for (int iz = iStartZ; iz < iEndZ; ++iz)
						for (int iy = iStartY; iy < iEndY; ++iy)
							for (int ix = iStartX; ix < iEndX; ++ix)
								sum += static_cast<float>(dataIn[in->idx(ix, iy, iz, c)]);
					dataOut[out->idx(ox, oy, oz, c)] = static_cast<T>(sum / (kX * kY * kZ));
					//std::cout << "[" << ox << "," << oy << "," << oz << "]: " << dataOut[out->idx(ox, oy, oz)] << std::endl;
					//if (dataOut[out->idx(ox, oy, oz)] < 0)
					//{
					//	std::cout << "NEGATIVE at [" << ox << "," << oy << "," << oz << "]: " <<
					//		"sum=" << sum << ", kX=" << kX << ", kY=" << kY << ", kZ=" << kZ <<
					//		", startX=" << iStartX << ", startY=" << iStartY << ", startZ" <<
					//		iStartZ << std::endl;
					//}
				}
			}
		}
	}

	//Halton-sampling the pixels to use.
	//It uses base 3, 5, 7 for the x,y,z axis
	template<typename T>
	void adaptive_halton_pool3d(const Volume::MipmapLevel* in, Volume::MipmapLevel* out)
	{
		const T* dataIn = in->dataCpu<T>();
		T* dataOut = out->dataCpu<T>();
		//fetch sizes
		const int inSizeX = static_cast<int>(in->sizeX());
		const int inSizeY = static_cast<int>(in->sizeY());
		const int inSizeZ = static_cast<int>(in->sizeZ());
		const int outSizeX = static_cast<int>(out->sizeX());
		const int outSizeY = static_cast<int>(out->sizeY());
		const int outSizeZ = static_cast<int>(out->sizeZ());
		//loop over output
		const int numChannels = in->channels();
#pragma omp parallel for collapse(2)
		for (int c = 0; c < numChannels; ++c)
		for (int oz = 0; oz < outSizeZ; ++oz)
		{
			const int iStartZ = start_index(oz, outSizeZ, inSizeZ);
			const int iEndZ = end_index(oz, outSizeZ, inSizeZ);
			const int kZ = iEndZ - iStartZ;
			for (int oy = 0; oy < outSizeY; ++oy)
			{
				const int iStartY = start_index(oy, outSizeY, inSizeY);
				const int iEndY = end_index(oy, outSizeY, inSizeY);
				const int kY = iEndY - iStartY;
				for (int ox = 0; ox < outSizeX; ++ox)
				{
					const int iStartX = start_index(ox, outSizeX, inSizeX);
					const int iEndX = end_index(ox, outSizeX, inSizeX);
					const int kX = iEndX - iStartX;

					//get sample index
					const uint64_t sampleIdx = uint64_t(out->idx(ox, oy, oz));
					const int ix = iStartX + int(kX * HaltonSampler::Sample<3, float>(sampleIdx));
					const int iy = iStartY + int(kY * HaltonSampler::Sample<5, float>(sampleIdx));
					const int iz = iStartZ + int(kZ * HaltonSampler::Sample<7, float>(sampleIdx));
					dataOut[out->idx(ox, oy, oz, c)] = dataIn[in->idx(ix, iy, iz, c)];
				}
			}
		}
	}
}

void Volume::Feature::createMipmapLevel(int level, MipmapFilterMode filter)
{
	switch (filter)
	{
	case MipmapFilterMode::AVERAGE:
		createMipmapLevelAverage(level);
		break;
	case MipmapFilterMode::HALTON:
		createMipmapLevelHalton(level);
		break;
	default:
		throw std::runtime_error("Unknown enum constant");
	}
}

bool Volume::Feature::mipmapCheckOrCreate(int level)
{
	CHECK_ERROR(level >= 0, "level has to be non-zero, but is ", level);
	if (level < levels_.size() && levels_[level]) return false; //already available

	//create storage
	if (level >= levels_.size()) levels_.resize(level + 1);
	size_t newSizeX = std::max(size_t(1), levels_[0]->sizeX_ / (level + 1));
	size_t newSizeY = std::max(size_t(1), levels_[0]->sizeY_ / (level + 1));
	size_t newSizeZ = std::max(size_t(1), levels_[0]->sizeZ_ / (level + 1));
	levels_[level] = std::unique_ptr<MipmapLevel>(new MipmapLevel(this, newSizeX, newSizeY, newSizeZ));
	return true;
}

static void fillMipmapLevelAverage(
	const Volume::MipmapLevel* in, Volume::MipmapLevel* out, Volume::DataType type)
{
	std::cout << "fillMipmapLevelAverage: from (" << in->sizeX() << "," << in->sizeY() <<
		"," << in->sizeZ() << ") to (" << out->sizeX() << "," << out->sizeY() <<
		"," << out->sizeZ() << ") with dtype " << magic_enum::enum_name(type) << std::endl;
	switch (type)
	{
	case Volume::TypeUChar:
		adaptive_avg_pool3d<unsigned char>(in, out);
		break;
	case Volume::TypeUShort:
		adaptive_avg_pool3d<unsigned short>(in, out);
		break;
	case Volume::TypeFloat:
		adaptive_avg_pool3d<float>(in, out);
		break;
	default:
		throw std::runtime_error("Unknown enum constant");
	}
}

void Volume::Feature::createMipmapLevelAverage(int level)
{
	if (!mipmapCheckOrCreate(level)) return; //already available
	auto data = levels_[level].get();

	//perform area filtering
	fillMipmapLevelAverage(levels_[0].get(), data, type_);
}

void Volume::Feature::createMipmapLevelHalton(int level)
{
	if (!mipmapCheckOrCreate(level)) return; //already available
	auto data = levels_[level].get();

	//perform area filtering
	switch (type_)
	{
	case TypeUChar:
		adaptive_halton_pool3d<unsigned char>(levels_[0].get(), data);
		break;
	case TypeUShort:
		adaptive_halton_pool3d<unsigned short>(levels_[0].get(), data);
		break;
	case TypeFloat:
		adaptive_halton_pool3d<float>(levels_[0].get(), data);
		break;
	default:
		throw std::runtime_error("Unknown enum constant");
	}
}

void Volume::Feature::deleteAllMipmapLevels()
{
	levels_.resize(1); //just keep the first level = original data
}

void Volume::Feature::clearGpuResources()
{
	for (const auto& l : levels_)
	{
		if (l) l->clearGpuResources();
	}
}

std::shared_ptr<const Volume::MipmapLevel> Volume::Feature::getLevel(int level) const
{
	CHECK_ERROR(level >= 0, "level has to be non-zero, but is ", level);
	if (level >= levels_.size()) return nullptr;
	return levels_[level];
}

Volume::MipmapLevel_ptr Volume::Feature::getLevel(int level)
{
	CHECK_ERROR(level >= 0, "level has to be non-zero, but is ", level);
	if (level >= levels_.size()) return nullptr;
	return levels_[level];
}

size_t Volume::Feature::estimateMemory() const
{
	int3 r = baseResolution();
	return static_cast<size_t>(BytesPerType[type()]) * r.x * r.y * r.z;
}

namespace {
	template<typename T>
	float density_cast(const T& value);

	template<>
	float density_cast<float>(const float& value) { return value; }

	template<>
	float density_cast<unsigned char>(const unsigned char& value) { return value / 255.0f; }

	template<>
	float density_cast<unsigned short>(const unsigned short& value) { return value / 65535.0f; }

	template<typename T>
	void fillHistogram(const Volume::MipmapLevel* level, Volume::Histogram* histogram)
	{
		// 1. extract min and max density
		float minDensity = histogram->minDensity;
		float maxDensity = histogram->maxDensity;
		int numNonZeroVoxels = 0;
		size_t sliceSize = level->channels() * level->sizeX() * level->sizeY();
#pragma omp parallel for
		for (int z = 0; z < static_cast<int>(level->sizeZ()); ++z)
		{
			float myMinDensity = FLT_MAX;
			float myMaxDensity = -FLT_MAX;
			int myNumNonZeroVoxels = 0;
			size_t sliceStart = z * sliceSize;
			for (size_t i = sliceStart; i < sliceStart + sliceSize; ++i)
			{
				auto raw = level->dataCpu<T>()[i];
				float density = density_cast<T>(raw);
				if (density != 0) { //ignore exact zeros (created when fitting the object in a cube for example)
					myMinDensity = std::min(myMinDensity, density);
					myMaxDensity = std::max(myMaxDensity, density);
					myNumNonZeroVoxels++;
				}
			}
#pragma omp critical
			{
				minDensity = std::min(myMinDensity, minDensity);
				maxDensity = std::max(myMaxDensity, maxDensity);
				numNonZeroVoxels += myNumNonZeroVoxels;
			}
		}
		histogram->maxDensity = maxDensity;
		histogram->minDensity = minDensity;
		histogram->numOfNonzeroVoxels = numNonZeroVoxels;

		// 2. fill histogram
		const int numBinsMinus1 = histogram->getNumOfBins() - 1;
		const float increment = 1.0f / numNonZeroVoxels;
#pragma omp parallel for
		for (int z = 0; z < static_cast<int>(level->sizeZ()); ++z)
		{
			Volume::Histogram myHistogram;
			size_t sliceStart = z * sliceSize;
			for (size_t i = sliceStart; i < sliceStart + sliceSize; ++i)
			{
				auto raw = level->dataCpu<T>()[i];
				float density = density_cast<T>(raw);
				if (density <= 0) continue;
				int binIdx = static_cast<int>(numBinsMinus1 * (density - minDensity) / (maxDensity - minDensity));
				binIdx = std::max(0, std::min(numBinsMinus1, binIdx));
				myHistogram.bins[binIdx] += increment;
			}
#pragma omp critical
			{
				for (int i = 0; i <= numBinsMinus1; ++i)
					histogram->bins[i] += myHistogram.bins[i];
			}
		}

		// 3. normalize
		histogram->maxFractionVal = *std::max_element(std::begin(histogram->bins), std::end(histogram->bins));
	}
}

Volume::Histogram_ptr Volume::Feature::extractHistogram() const
{
	Volume::Histogram_ptr histogram = std::make_shared<Histogram>();
	const auto level = getLevel(0);

	switch (type())
	{
	case TypeUChar:
		fillHistogram<unsigned char>(level.get(), histogram.get());
		break;
	case TypeUShort:
		fillHistogram<unsigned short>(level.get(), histogram.get());
		break;
	case TypeFloat:
		fillHistogram<float>(level.get(), histogram.get());
		break;
	default:
		throw std::runtime_error("Unknown data type");
	}

	return histogram;
}




Volume::Volume()
	: worldSizeX_(1), worldSizeY_(1), worldSizeZ_(1)
{
}

static const char MAGIC[] = "CVOL";
static const char MAGIC_OLD[] = "cvol";

static const int VERSION = 1;

void Volume::save(const std::string& filename,
	const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging,
	const VolumeErrorCallback_t& error,
	int compression) const
{
	static_assert(sizeof(size_t) == 8, "Size test failed, what compiler did you smoke?");
	static_assert(sizeof(double) == 8, "Size test failed, what compiler did you smoke?");
	static_assert(sizeof(float) == 4, "Size test failed, what compiler did you smoke?");
	std::ofstream s(filename, std::fstream::binary);

	if (compression < 0 || compression > MAX_COMPRESSION)
		throw std::runtime_error("Illegal compression factor");

	//header
	s.write(MAGIC, 4);
	s.write(reinterpret_cast<const char*>(&VERSION), 4);
	s.write(reinterpret_cast<const char*>(&worldSizeX_), 4);
	s.write(reinterpret_cast<const char*>(&worldSizeY_), 4);
	s.write(reinterpret_cast<const char*>(&worldSizeZ_), 4);
	int numFeatures = this->numFeatures();
	s.write(reinterpret_cast<const char*>(&numFeatures), 4);
	int flags = 0;
	char useCompression = compression > 0 ? 1 : 0;
	if (useCompression) flags = flags | Flag_Compressed;
	s.write(reinterpret_cast<const char*>(&flags), 4);
	char padding[4] = { 0 };
	s.write(padding, 4);

	LZ4Compressor c(compression >= LZ4Compressor::MIN_COMPRESSION ? compression : LZ4Compressor::FAST_COMPRESSION);
	LZ4Compressor* cPtr = useCompression > 0 ? &c : nullptr;

	//content
	const float progressStep = 1.0f / numFeatures;
	for (int i = 0; i < numFeatures; ++i)
	{
		const auto f = features_[i];
		logging("Save feature " + f->name());
		const float progressOffset = i * progressStep;
		const auto progress2 = [&](float p)
		{
			progress(progressOffset + p * progressStep);
		};

		f->save(s, cPtr, progress2, logging, error);
	}
	progress(1.0f);
}

void Volume::save(const std::string& filename, int compression) const
{
	save(filename,
		[](float v) {printProgress("Save", v); },
		[](const std::string& msg) {std::cout << msg << std::endl; },
		[](const std::string& msg, int code)
		{
			std::cerr << msg << std::endl;
			throw std::runtime_error(msg.c_str());
		},
		compression);
}

Volume::Volume(const std::string& filename,
               const VolumeProgressCallback_t& progress,
               const VolumeLoggingCallback_t& logging,
               const VolumeErrorCallback_t& error)
	: Volume()
{
	assert(sizeof(size_t) == 8);
	assert(sizeof(double) == 8);
	std::ifstream s(filename, std::fstream::binary);
	if (!s.is_open())
	{
		error(tinyformat::format("Unable to open file %s", filename), -2);
	}

	//header
	char magic[4];
	s.read(magic, 4);

	if (memcmp(MAGIC, magic, 4) == 0)
	{
	    //load new version
		int version;
		int numFeatures;
		int flags;
		s.read(reinterpret_cast<char*>(&version), 4);
        if (version != VERSION)
        {
			error("Unknown file version!", -3);
        }
		s.read(reinterpret_cast<char*>(&worldSizeX_), 4);
		s.read(reinterpret_cast<char*>(&worldSizeY_), 4);
		s.read(reinterpret_cast<char*>(&worldSizeZ_), 4);
		s.read(reinterpret_cast<char*>(&numFeatures), 4);
		s.read(reinterpret_cast<char*>(&flags), 4);
		s.ignore(4);
		bool useCompression = (flags & Flag_Compressed) > 0;

		LZ4Decompressor d;
		LZ4Decompressor* dPtr = useCompression ? &d : nullptr;

		//load features
		features_.resize(numFeatures);
		const float progressStep = 1.0f / numFeatures;
		for (int i = 0; i < numFeatures; ++i)
		{
			logging("Load feature " + std::to_string(i));
			const float progressOffset = i * progressStep;
			const auto progress2 = [&](float p)
			{
				progress(progressOffset + i * progressStep);
			};
			const auto f = Feature::load(this, s, dPtr, progress2, logging, error);
			features_[i] = f;
		}
		progress(1.0f);
	}
	else if (memcmp(MAGIC_OLD, magic, 4) == 0)
	{
		//old version, only density

		size_t sizeX, sizeY, sizeZ;
		double voxelSizeX, voxelSizeY, voxelSizeZ;
		char useCompression;
		s.read(reinterpret_cast<char*>(&sizeX), 8);
		s.read(reinterpret_cast<char*>(&sizeY), 8);
		s.read(reinterpret_cast<char*>(&sizeZ), 8);
		s.read(reinterpret_cast<char*>(&voxelSizeX), 8);
		s.read(reinterpret_cast<char*>(&voxelSizeY), 8);
		s.read(reinterpret_cast<char*>(&voxelSizeZ), 8);
		unsigned int type;
		s.read(reinterpret_cast<char*>(&type), 4);
		s.read(&useCompression, 1);
		s.ignore(7);

		//create feature and level
		auto feature = addFeature("density", static_cast<DataType>(type), 1, sizeX, sizeY, sizeZ);
		MipmapLevel_ptr data = feature->getLevel(0);
		worldSizeX_ = voxelSizeX * sizeX;
		worldSizeY_ = voxelSizeY * sizeY;
		worldSizeZ_ = voxelSizeZ * sizeZ;

		//body
		progress(0.0f);
		if (useCompression) {
			LZ4Decompressor d;
			size_t dataToRead = BytesPerType[type] * sizeX * sizeY * sizeZ;
			for (size_t offset = 0; offset < dataToRead;)
			{
				char* mem = data->dataCpu<char>() + offset;
				const int len = std::min(
					static_cast<int>(dataToRead - offset),
					std::numeric_limits<int>::max());
				int chunkSize = d.decompress(mem, len, s);
				progress(offset / float(dataToRead));
				offset += chunkSize;
			}
		}
		else
		{
			size_t dataToRead = BytesPerType[type] * sizeX * sizeY;
			for (int z = 0; z < sizeZ; ++z)
			{
				s.read(data->dataCpu<char>() + z * dataToRead, dataToRead);
				if (z % 10 == 0)
					progress(z / float(sizeZ));
			}
		}
		progress(1.0f);
	}
	else
	{
		error("Illegal magic number", -1);
	}
}

Volume::Volume(const std::string& filename)
	: Volume(filename,
		[](float v) {printProgress("Load", v); },
		[](const std::string& msg) {std::cout << msg << std::endl; },
		[](const std::string& msg, int code)
		{
			std::cerr << msg << std::endl;
			throw std::runtime_error(msg.c_str());
		})
{
}

Volume::Feature_ptr Volume::getFeature(int index) const
{
	TORCH_CHECK(index >= 0 && index < numFeatures(), "feature index is out of bounds!");
	return features_[index];
}

Volume::Feature_ptr Volume::getFeature(const std::string& name) const
{
	for (const auto& f : features_)
	{
		if (f->name() == name) return f;
	}
	return nullptr;
}

Volume::Feature_ptr Volume::addFeature(const std::string& name, DataType type, int numChannels, uint64_t sizeX,
    uint64_t sizeY, uint64_t sizeZ)
{
	Feature_ptr f = std::make_shared<Feature>(this, name, type, numChannels, sizeX, sizeY, sizeZ);
	features_.push_back(f);
	return f;
}

Volume::Feature_ptr Volume::addFeatureFromBuffer(const std::string& name, const float* buffer, long long sizes[4],
	long long strides[4])
{
	std::cout << "Add feature " << name << " from buffer with Channels=" << sizes[0] <<
		" (stride=" << strides[0] << "), X=" << sizes[1] << " (stride=" << strides[1] <<
		"), Y=" << sizes[2] << " (stride=" << strides[2] << "), Z=" << sizes[3] <<
		" (stride=" << strides[3] << ")" << std::endl;
	Feature_ptr f = addFeature(name, TypeFloat, static_cast<int>(sizes[0]), sizes[1], sizes[2], sizes[3]);

	auto level = f->getLevel(0);
	auto data = level->dataCpu<float>();
#pragma omp parallel for
	for (int x = 0; x < sizes[1]; ++x)
		for (int y = 0; y < sizes[2]; ++y)
			for (int z = 0; z < sizes[3]; ++z)
				for (int c = 0; c < sizes[0]; ++c)
					data[level->idx(x, y, z, c)] = buffer[c * strides[0] + x * strides[1] + y * strides[2] + z * strides[3]];

	return f;
}

void Volume::clearGpuResources()
{
	for (const auto& f : features_)
		f->clearGpuResources();
}

size_t Volume::estimateMemory() const
{
	size_t estimate = 0;
	for (const auto& f : features_)
		estimate += f->estimateMemory();
	return estimate;
}

std::unique_ptr<Volume> Volume::createSyntheticDataset(int resolution, float boxMin, float boxMax,
	const ImplicitFunction_t& f)
{
	auto vol = std::make_unique<Volume>();
	auto density = vol->addFeature(
		"density", DataType::TypeFloat, 1,
		resolution, resolution, resolution);
	auto level = density->getLevel(0);
	auto data = level->dataCpu<float>();
	float scale = (boxMax - boxMin) / (resolution - 1);
#pragma omp parallel for
	for (int x = 0; x < resolution; ++x)
		for (int y = 0; y < resolution; ++y)
			for (int z = 0; z < resolution; ++z)
			{
				float3 xyz = make_float3(
					boxMin + x * scale, boxMin + y * scale, boxMin + z * scale);
				float v = f(xyz);
				data[level->idx(x, y, z)] = v;
			}
	return vol;
}


std::shared_ptr<Volume> Volume::loadVolumeFromRaw(
	const std::string& filename, const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging, const VolumeErrorCallback_t& error, 
	const std::optional<int>& ensemble)
{
	auto filename_path = std::filesystem::path(filename);
	//read descriptor file
	if (filename_path.extension() == "dat")
	{
		error("Unrecognized extension, .dat expected : ." + filename_path.extension().string(), -1);
		return nullptr;
	}
	std::ifstream file(filename);
	if (!file.is_open())
	{
		error("Unable to open file " + filename, -1);
		return nullptr;
	}
	std::string line;
	std::string objectFileName = "";
	size_t resolutionX = 0;
	size_t resolutionY = 0;
	size_t resolutionZ = 0;
	double sliceThicknessX = 1;
	double sliceThicknessY = 1;
	double sliceThicknessZ = 1;
	std::string datatype = "";
	const std::string DATATYPE_UCHAR = "UCHAR";
	const std::string DATATYPE_USHORT = "USHORT";
	const std::string DATATYPE_BYTE = "BYTE";
	const std::string DATATYPE_FLOAT = "FLOAT";
	while (std::getline(file, line))
	{
		if (line.empty()) continue;
		std::istringstream iss(line);
		std::string token;
		iss >> token;
		if (!iss) continue; //no token in the current line
		if (token == "ObjectFileName:")
			iss >> objectFileName;
		else if (token == "Resolution:")
			iss >> resolutionX >> resolutionY >> resolutionZ;
		else if (token == "SliceThickness:")
			iss >> sliceThicknessX >> sliceThicknessY >> sliceThicknessZ;
		else if (token == "Format:")
			iss >> datatype;
		if (!iss)
		{
			error("Unable to parse line with token " + token, -2);
			return nullptr;
		}
	}
	file.close();
	if (objectFileName.empty() || resolutionX == 0 || datatype.empty())
	{
		error("Descriptor file does not contain ObjectFileName, Resolution and Format", -3);
		return nullptr;
	}
	if (!(datatype == DATATYPE_UCHAR || datatype == DATATYPE_USHORT || datatype == DATATYPE_BYTE || datatype == DATATYPE_FLOAT))
	{
		error("Unknown format " + datatype, -4);
		return nullptr;
	}

	if (ensemble.has_value())
	{
		char buff[256];
		snprintf(buff, sizeof(buff), objectFileName.c_str(), ensemble.value());
		objectFileName = buff;
	}
	
	logging(std::string("Descriptor file read")
		+ "\nObjectFileName: " + objectFileName
		+ "\nResolution: " + std::to_string(resolutionX) + ", " + std::to_string(resolutionY) + ", " + std::to_string(resolutionZ)
		+ "\nFormat: " + datatype);

	// open volume
	size_t bytesPerEntry = 0;
	if (datatype == DATATYPE_UCHAR) bytesPerEntry = 1;
	if (datatype == DATATYPE_BYTE) bytesPerEntry = 1;
	if (datatype == DATATYPE_USHORT) bytesPerEntry = 2;
	if (datatype == DATATYPE_FLOAT) bytesPerEntry = 4;
	size_t bytesToRead = resolutionX * resolutionY * resolutionZ * bytesPerEntry;
	std::string bfilename = filename_path.replace_filename(objectFileName).generic_string();

	if (bytesToRead > 1024ll * 1024 * 1024 * 16)
	{
		error("Files is too large", -10);
		return nullptr;
	}

	std::cout << "Load " << bytesToRead << " bytes from " << bfilename << std::endl;
	std::ifstream bfile(bfilename, std::ifstream::binary | std::ifstream::ate);
	if (!bfile.is_open())
	{
		error("Unable to open file " + bfilename, -5);
		return nullptr;
	}
	size_t filesize = bfile.tellg();
	int headersize = static_cast<int>(filesize - static_cast<long long>(bytesToRead));
	if (headersize < 0)
	{
		error("File is too small, " + std::to_string(-headersize) + " bytes missing", -6);
		return nullptr;
	}
	std::cout << "Skipping header of length " << headersize << std::endl;
	bfile.seekg(std::ifstream::pos_type(headersize));

	// create output volume and read the data
	bytesToRead = resolutionX * resolutionY * bytesPerEntry;
	std::vector<char> data(bytesToRead);

	std::unique_ptr<Volume> vol;
	if (datatype == DATATYPE_UCHAR || datatype == DATATYPE_BYTE) {
		vol = std::make_unique<Volume>();
		auto feature = vol->addFeature(
			"density", Volume::TypeUChar, 1, resolutionX, resolutionY, resolutionZ);
		MipmapLevel_ptr level = feature->getLevel(0);
		unsigned char* volumeData = level->dataCpu<unsigned char>();
		const unsigned char* raw = reinterpret_cast<unsigned char*>(data.data());
		for (int z = 0; z < resolutionZ; ++z)
		{
			bfile.read(&data[0], bytesToRead);
			if (!bfile)
			{
				error("Loading data file failed", -7);
				return nullptr;
			}
			if (z % 10 == 0)
				progress(z / float(resolutionZ));
#pragma omp parallel for
			for (int y = 0; y < resolutionY; ++y)
				for (int x = 0; x < resolutionX; ++x)
				{
					unsigned char val = raw[x + resolutionX * y];
					volumeData[level->idx(x, y, z)] = val;
				}
		}
	}
//	else if (datatype == DATATYPE_BYTE) {
//		logging("signed BYTE format not supported, convert to FLOAT");
//		vol = std::make_unique<Volume>(
//			Volume::TypeFloat, resolutionX, resolutionY, resolutionZ);
//		float* volumeData = vol->dataCpu<float>();
//		const float* raw = reinterpret_cast<float*>(data.data());
//		for (int z = 0; z < resolutionZ; ++z)
//		{
//			bfile.read(&data[0], bytesToRead);
//			if (!bfile)
//			{
//				error("Loading data file failed", -7);
//				return nullptr;
//			}
//			if (z % 10 == 0)
//				progress(z / float(resolutionZ));
//#pragma omp parallel for
//			for (int y = 0; y < resolutionY; ++y)
//				for (int x = 0; x < resolutionX; ++x)
//				{
//					float val = raw[x + resolutionX * y] / 255.0f;
//					volumeData[vol->idx(x, y, z)] = val;
//				}
//		}
//	}
	else if (datatype == DATATYPE_USHORT) {
		vol = std::make_unique<Volume>();
		auto feature = vol->addFeature(
			"density", Volume::TypeUShort, 1, resolutionX, resolutionY, resolutionZ);
		MipmapLevel_ptr level = feature->getLevel(0);
		unsigned short* volumeData = level->dataCpu<unsigned short>();
		const unsigned short* raw = reinterpret_cast<unsigned short*>(data.data());
		for (int z = 0; z < resolutionZ; ++z)
		{
			bfile.read(&data[0], bytesToRead);
			if (!bfile)
			{
				error("Loading data file failed", -7);
				return nullptr;
			}
			if (z % 10 == 0)
				progress(z / float(resolutionZ));
#pragma omp parallel for
			for (int y = 0; y < resolutionY; ++y)
				for (int x = 0; x < resolutionX; ++x)
				{
					unsigned short val = raw[x + resolutionX * y];
					volumeData[level->idx(x, y, z)] = val;
				}
		}
	}
	else if (datatype == DATATYPE_FLOAT) {
		vol = std::make_unique<Volume>();
		auto feature = vol->addFeature(
			"density", Volume::TypeFloat, 1, resolutionX, resolutionY, resolutionZ);
		MipmapLevel_ptr level = feature->getLevel(0);
		float* volumeData = level->dataCpu<float>();
		const float* raw = reinterpret_cast<float*>(data.data());
		for (int z = 0; z < resolutionZ; ++z)
		{
			bfile.read(&data[0], bytesToRead);
			if (!bfile)
			{
				error("Loading data file failed", -7);
				return nullptr;
			}
			if (z % 10 == 0)
				progress(z / float(resolutionZ));
#pragma omp parallel for
			for (int y = 0; y < resolutionY; ++y)
				for (int x = 0; x < resolutionX; ++x)
				{
					float val = raw[x + resolutionX * y];
					volumeData[level->idx(x, y, z)] = val;
				}
		}
	}
	progress(1.0f);

	// set voxel size, scale so that a box of at most 1x1x1 is occupied
	double maxSize = std::max({
		sliceThicknessX * resolutionX,
		sliceThicknessY * resolutionY,
		sliceThicknessZ * resolutionZ
	});
	vol->setWorldSizeX(sliceThicknessX / maxSize * resolutionX);
	vol->setWorldSizeY(sliceThicknessY / maxSize * resolutionY);
	vol->setWorldSizeZ(sliceThicknessZ / maxSize * resolutionZ);

	//done
	std::stringstream s;
	s << "Reading done, resolution=(" << resolutionX <<
		"," << resolutionY << "," << resolutionZ <<
		"), size=(" << vol->worldSizeX() <<
		"," << vol->worldSizeY() << "," << vol->worldSizeZ() <<
		")";
	logging(s.str());

	return vol;
}

std::shared_ptr<Volume> Volume::loadVolumeFromRaw(const std::string& file, const std::optional<int>& ensemble)
{
	return loadVolumeFromRaw(
		file,
		[](float v) {printProgress("Load", v); },
		[](const std::string& msg) {std::cout << msg << std::endl; },
		[](const std::string& msg, int code)
		{
			std::cerr << msg << std::endl;
			throw std::runtime_error(msg.c_str());
		},
		ensemble);
}


std::shared_ptr<Volume> Volume::loadVolumeFromXYZ(
	const std::string& filename, const VolumeProgressCallback_t& progress,
	const VolumeLoggingCallback_t& logging, const VolumeErrorCallback_t& error)
{
	std::ifstream in(filename, std::ifstream::in | std::ifstream::binary);
	unsigned int sizeX, sizeY, sizeZ;
	double voxelSizeX, voxelSizeY, voxelSizeZ;
	in.read(reinterpret_cast<char*>(&sizeX), sizeof(unsigned int));
	in.read(reinterpret_cast<char*>(&sizeY), sizeof(unsigned int));
	in.read(reinterpret_cast<char*>(&sizeZ), sizeof(unsigned int));
	in.read(reinterpret_cast<char*>(&voxelSizeX), sizeof(double));
	in.read(reinterpret_cast<char*>(&voxelSizeY), sizeof(double));
	in.read(reinterpret_cast<char*>(&voxelSizeZ), sizeof(double));
	unsigned int maxSize = std::max({ sizeX, sizeY, sizeZ });
	voxelSizeX = 1.0 / maxSize;
	voxelSizeY = 1.0 / maxSize;
	voxelSizeZ = 1.0 / maxSize;

	std::unique_ptr<Volume> vol = std::make_unique<Volume>();
	auto feature = vol->addFeature(
		"density", Volume::TypeFloat, 1, sizeX, sizeY, sizeZ);
	MipmapLevel_ptr level = feature->getLevel(0);
	vol->setWorldSizeX(voxelSizeX * sizeX);
	vol->setWorldSizeY(voxelSizeY * sizeY);
	vol->setWorldSizeZ(voxelSizeZ * sizeZ);
	float* volumeData = level->dataCpu<float>();

	size_t floatsToRead = sizeZ * sizeY;
	std::vector<float> data(floatsToRead);
	for (unsigned int x = 0; x < sizeX; ++x)
	{
		in.read(reinterpret_cast<char*>(&data[0]), sizeof(float)*floatsToRead);
		if (!in)
		{
			error("Loading data file failed", -7);
			return nullptr;
		}
		if (x % 10 == 0)
			progress(x / float(sizeX));

#pragma omp parallel for
		for (int y = 0; y < int(sizeY); ++y)
			for (int z = 0; z < int(sizeZ); ++z)
				volumeData[level->idx(x, y, z)] = data[z + sizeZ * y];
	}
	progress(1.0f);

	//done
	std::stringstream s;
	s << "Reading done, resolution=(" << sizeX <<
		"," << sizeY << "," << sizeZ <<
		"), size=(" << vol->worldSizeX() <<
		"," << vol->worldSizeY() << "," << vol->worldSizeZ() <<
		")" << std::endl;
	logging(s.str());

	return vol;
}

std::shared_ptr<Volume> Volume::loadVolumeFromXYZ(const std::string& file)
{
	return loadVolumeFromXYZ(file,
		[](float v) {printProgress("Load", v); },
		[](const std::string& msg) {std::cout << msg << std::endl; },
		[](const std::string& msg, int code)
		{
			std::cerr << msg << std::endl;
			throw std::runtime_error(msg.c_str());
		});
}

std::shared_ptr<Volume> Volume::createScaled(int X, int Y, int Z) const
{
	auto newVolume = std::make_shared<Volume>();
	newVolume->setWorldSizeX(worldSizeX());
	newVolume->setWorldSizeY(worldSizeY());
	newVolume->setWorldSizeZ(worldSizeZ());

	for (int i=0; i<numFeatures(); ++i)
	{
		const auto oldFeature = getFeature(i);
		auto newFeature = newVolume->addFeature(
			oldFeature->name(), oldFeature->type(), oldFeature->numChannels(),
			X, Y, Z);
		const auto oldData = oldFeature->getLevel(0);
		auto newData = newFeature->getLevel(0);
		fillMipmapLevelAverage(oldData.get(), newData.get(), oldFeature->type());
	}

	return newVolume;
}

void Volume::registerPybindModules(pybind11::module& m)
{
	namespace py = pybind11;
	py::class_<Volume, Volume_ptr> v(m, "Volume", py::buffer_protocol());

	py::class_<Histogram, Histogram_ptr>(v, "Histogram")
		.def_property_readonly("num_bins", &Histogram::getNumOfBins)
		.def_readonly("min_density", &Histogram::minDensity)
		.def_readonly("max_density", &Histogram::maxDensity)
		.def_readonly("max_fractional_value", &Histogram::maxFractionVal)
		.def_readonly("num_nonzero_voxels", &Histogram::numOfNonzeroVoxels)
		.def("bins", [](Histogram* self)
			{
				return py::memoryview(py::buffer_info(self->bins, self->getNumOfBins()));
			});

	py::enum_<DataType>(v, "DataType")
		.value("TypeUChar", DataType::TypeUChar)
		.value("TypeUShort", DataType::TypeUShort)
		.value("TypeFloat", DataType::TypeFloat)
		.export_values();
	v.def_static("bytes_per_type", [](DataType t) {return BytesPerType[t]; },
		py::doc("Returns the number of bytes per entry of the given DataType"));

	py::enum_<MipmapFilterMode>(v, "MipmapFilterMode")
		.value("AVERAGE", MipmapFilterMode::AVERAGE)
		.value("HALTON", MipmapFilterMode::HALTON)
		.export_values();

	py::class_<MipmapLevel, MipmapLevel_ptr>(v, "MipmapLevel")
	    .def("channels", &MipmapLevel::channels, py::doc("The number of channels"))
	    .def("sizeX", &MipmapLevel::sizeX, py::doc("The voxel resolution along X"))
		.def("sizeY", &MipmapLevel::sizeY, py::doc("The voxel resolution along Y"))
		.def("sizeZ", &MipmapLevel::sizeZ, py::doc("The voxel resolution along Z"))
		.def("sizes", &MipmapLevel::size, py::doc("The voxel resolution as int3"))
	    .def("idx", &MipmapLevel::idx, py::doc("Computes the linear index"),
			py::arg("x"), py::arg("y"), py::arg("z"), py::arg("channel"))
		.def("type", &MipmapLevel::type, py::doc("The data type"))
#ifdef _MSC_VER
        //Does only work on Windows for some reasons
	    .def("data", [](MipmapLevel* self)
	    {
	        switch (self->type())
	        {
			case TypeUChar: return py::memoryview(py::buffer_info(
				self->dataCpu<unsigned char>(), 
				{self->channels(), self->sizeX(), self->sizeY(), self->sizeZ()},
				{1ll, self->channels(), self->channels()*self->sizeX(), self->channels()*self->sizeX()*self->sizeY()}));
			case TypeUShort: return py::memoryview(py::buffer_info(
				self->dataCpu<unsigned short>(),
				{ self->channels(), self->sizeX(), self->sizeY(), self->sizeZ() },
				{ 1ll, self->channels(), self->channels() * self->sizeX(), self->channels() * self->sizeX() * self->sizeY() }));
	        case TypeFloat: return py::memoryview(py::buffer_info(
				self->dataCpu<float>(),
				{ self->channels(), self->sizeX(), self->sizeY(), self->sizeZ() },
				{ 1ll, self->channels(), self->channels() * self->sizeX(), self->channels() * self->sizeX() * self->sizeY() }));
			default: throw std::runtime_error("Unknown datatype");
	        }
			}, py::doc("Returns a view of the data as buffer of shape C*X*Y*Z"))
#endif
	    .def("copy_cpu_to_gpu", &MipmapLevel::copyCpuToGpu, py::doc("Copies the CPU data to the GPU"))
	    .def("clear_gpu_resources", &MipmapLevel::clearGpuResources, py::doc("Clears the GPU resources"))
	    .def("to_tensor", &MipmapLevel::toTensor, py::doc("Returns a copy oft his mipmap's data as a CPU float tensor. Shape: C*X*Y*Z"))
	    .def("from_tensor", &MipmapLevel::fromTensor, py::doc("Sets this mipmap's data from the given float tensor. The shape of C * X * Y * Z must match"))
	    ;

	py::class_<Feature, Feature_ptr>(v, "Feature")
	    .def("name", &Feature::name)
	    .def("type", &Feature::type)
	    .def("channels", &Feature::numChannels)
	    .def("base_resolution", &Feature::baseResolution, py::doc("The resolution of mipmap level 0"))
	    .def("create_mipmap_level", &Feature::createMipmapLevel, py::doc(R"(
            Creates the mipmap level specified by the given index.
            The level zero is always the original data.
            Level 1 is 2x downsampling, level 2 is 3x downsampling, level n is (n+1)x downsampling.
            This function does nothing if that level is already created.
            )"))
	    .def("delete_all_mipmaps", &Feature::deleteAllMipmapLevels,
	        py::doc("Deletes all mipmaps, except the base level 0."))
	    .def("clear_gpu_resources", &Feature::clearGpuResources,
	        py::doc("Clears all GPU resources of all mipmaps"))
	    .def("get_level", static_cast<MipmapLevel_ptr(Feature::*)(int level)>(&Feature::getLevel), py::doc("Returns the mipmap level"))
	    .def("extract_histogram", &Feature::extractHistogram)
	    ;

	//main volume
	v.def(py::init<>(), py::doc("Creates a new, empty volume"))
		.def(py::init<const std::string&>(), py::doc("Loads the volume from the given .cvol file"))
		.def("save", static_cast<void(Volume::*)(const std::string & filename, int compression) const>(&Volume::save),
			py::doc("Saves the volume to the .cvol file. The compression must be in [0,9] where 0 is no compression and 9 is max compression."),
			py::arg("filename"), py::arg("compression") = NO_COMPRESSION)
		.def_static("load_from_raw",
			static_cast<std::shared_ptr<Volume>(*)(const std::string & file, const std::optional<int>&ensemble)>(&Volume::loadVolumeFromRaw),
			py::doc("Loads the volume from the RAW file (specified typically by a .dat file)."),
			py::arg("filename"), py::arg("ensemble") = std::optional<int>())
		.def_static("load_from_xyz",
			static_cast<std::shared_ptr<Volume>(*)(const std::string & file)>(&Volume::loadVolumeFromXYZ),
			py::doc("Loads the volume from the RAW file (specified typically by a .dat file)."),
			py::arg("filename"))
		.def_property("worldX", &Volume::worldSizeX, &Volume::setWorldSizeX,
			"The size of the bounding box in world space along X")
		.def_property("worldY", &Volume::worldSizeY, &Volume::setWorldSizeY,
			"The size of the bounding box in world space along Y")
		.def_property("worldZ", &Volume::worldSizeZ, &Volume::setWorldSizeZ,
			"The size of the bounding box in world space along Z")
        .def("num_features", &Volume::numFeatures, 
			py::doc("Returns the number of Feature channels"))
	    .def("get_feature", static_cast<Feature_ptr(Volume::*)(int index) const>(&Volume::getFeature),
			py::doc("Returns the feature channel at the given index. If the index is out of bounds, an exception is thrown."),
			py::arg("index"))
		.def("get_feature", static_cast<Feature_ptr(Volume::*)(const std::string& name) const>(&Volume::getFeature),
			py::doc("Searches and returns the feature channel with the given name. If the feature was not found, None is returned."),
			py::arg("name"))
	    .def("add_feature", &Volume::addFeature,
			py::doc("Adds a new feature channel"),
			py::arg("name"),  py::arg("type"), py::arg("num_channels"), 
			py::arg("sizeX"), py::arg("sizeY"), py::arg("sizeZ"))
#ifdef _MSC_VER
        //does only work on Windows (error in pybind on Unix)
	    .def("add_feature_from_buffer", [](Volume* self, const std::string& name, py::buffer b)
	    {
				//Note: This function crashes for some reasons. Pybind11-error?
				/* Request a buffer descriptor from Python */
				const py::buffer_info info = b.request();

				/* Some sanity checks ... */
				if (info.format != py::format_descriptor<float>::format())
					throw std::runtime_error("Incompatible format: expected a float array!");

				if (info.ndim != 4)
					throw std::runtime_error("Incompatible buffer dimension, expected a 4D array!");

				long long sizes[] = { info.shape[0], info.shape[1], info.shape[2], info.shape[3] };
				long long strides[] = { info.strides[0], info.strides[1], info.strides[2], info.strides[3] };
				return self->addFeatureFromBuffer(name, reinterpret_cast<const float*>(info.ptr), sizes, strides);
	    }, py::doc("Adds a new feature with the given name from the raw buffer, a 4D array with the dimensions Channel,X,Y,Z."),
			py::arg("name"), py::arg("buffer"))
#endif
	    .def("add_feature_from_tensor", [](Volume* self, const std::string& name, const torch::Tensor& t)
	    {
	        if (!(t.dtype() == c10::kFloat))
				throw std::runtime_error("Incompatible format: expected a float array!");
			if (t.device() != c10::kCPU)
				throw std::runtime_error("Incompatible format, expected the tensor to reside in CPU memory");
			if (t.dim() != 4)
				throw std::runtime_error("Incompatible buffer dimension, expected a 4D array!");

			long long sizes[] = { t.size(0), t.size(1), t.size(2), t.size(3) };
			long long strides[] = { t.stride(0), t.stride(1), t.stride(2), t.stride(3) };
			const float* buffer = t.data_ptr<float>();
			return self->addFeatureFromBuffer(name, buffer, sizes, strides);
	    }, py::doc("Adds a new feature with the given name from the given CPU-float tensor, a 4D array with the dimensions Channel,X,Y,Z."),
			py::arg("name"), py::arg("buffer"))
	    .def("create_scaled", &Volume::createScaled,
			py::doc(" Creates a scaled version of this volume with the same features and world resolution, but new voxel resolution"),
			py::arg("X"), py::arg("Y"), py::arg("Z"))
		;
	    
}

VolumeEnsembleFactory::VolumeEnsembleFactory(const std::string& filename)
{
	nlohmann::json file;
	{
		std::ifstream in(filename);
		if (!in.is_open())
			throw std::runtime_error(tinyformat::format("Unable to open file %s", filename));
        try {
            in >> file;
        } catch (const std::exception& ex)
        {
            std::cout << "Unable to load json with the ensemble specification (" << filename << "): " << ex.what() << std::endl;
            throw;
        }
	}
	formatString_ = file.at("format");
	startEnsemble_ = file.value("start_ensemble", 0);
	stepEnsemble_ = file.value("step_ensembles", 1);
	numEnsembles_ = file.value("num_ensembles", 1);
	startTimestep_ = file.value("start_timestep", 0);
	stepTimestep_ = file.value("step_timestep", 1);
	numTimesteps_ = file.value("num_timesteps", 1);
	root_ = std::filesystem::absolute(std::filesystem::path(filename)).parent_path().string();
}

void VolumeEnsembleFactory::save(const std::string& filename)
{
	nlohmann::json file;
	file["format"] = formatString_;
	file["start_ensemble"] = startEnsemble_;
	file["step_ensembles"] = stepEnsemble_;
	file["num_ensembles"] = numEnsembles_;
	file["start_timestep"] = startTimestep_;
	file["step_timestep"] = stepTimestep_;
	file["num_timesteps"] = numTimesteps_;
	std::ofstream o(filename);
	if (!o.is_open())
		throw std::runtime_error(tinyformat::format("Unable to open file %s for writing", filename));
	o << file;
}

Volume_ptr VolumeEnsembleFactory::loadVolume(int ensemble, int time)
{
	static constexpr size_t MAX_CACHE_MEMORY = 1024ull * 1024 * 1024 * 4; //4 GB

	EnsembleTime_t key{ ensemble, time };
	std::string filename = getVolumeFilename(ensemble, time);

	Volume_ptr v = nullptr;
	if (!cache_)
	{
		//no cache created yet, this is the first volume
		try {
			v = std::make_shared<Volume>(filename);
		} catch (std::exception& w)
		{
			std::cerr << "Unable to load volume: " << w.what() << std::endl;
			return nullptr;
		}

		//estimate filesize
		size_t estimate = v->estimateMemory();
		int fit = std::max(size_t(1), MAX_CACHE_MEMORY / estimate);
		std::cerr << "Volume requires " << estimate << " bytes to store (estimate) -> fit " << fit << " volumes in the LRU-cache" << std::endl;
		cache_ = std::make_unique<Cache_t>(fit);
		cache_->put(key, v);
	}
	else
	{
		//cache exists, try to load from cache
		if (cache_->exist(key))
		{
			v = cache_->get(key);
		} else
		{
			try {
				v = std::make_shared<Volume>(filename);
			}
			catch (std::exception& w)
			{
				std::cerr << "Unable to load volume: " << w.what() << std::endl;
				return nullptr;
			}
			cache_->put(key, v);
		}
	}
	return v;
}

std::string VolumeEnsembleFactory::getVolumeFilename(int ensemble, int time)
{
	if (ensemble < 0 || ensemble >= numEnsembles())
		throw std::runtime_error("ensemble out of bounds");
	if (time < 0 || time >= numTimesteps())
		throw std::runtime_error("timestep out of bounds");

	ensemble = startEnsemble() + stepEnsemble() * ensemble;
	time = startTimestep() + stepTimestep() * time;

	auto filename = tinyformat::format(formatString_.c_str(), ensemble, time);
	auto filenamePath = std::filesystem::path(filename);
	if (filenamePath.is_relative())
		filename = absolute(std::filesystem::path(root_) / filename).string();
	return filename;
}

void VolumeEnsembleFactory::registerPybindModules(pybind11::module& m)
{
	namespace py = pybind11;
	py::class_<VolumeEnsembleFactory, VolumeEnsembleFactory_ptr>(m, "VolumeEnsembleFactory",
		R"(
        Factory class for \ref Volume instances that are part of a timeseries or ensemble.
        
        The ensemble has two parameters, "ensemble" and "timestep".
        The mapping to the filename of the actual volume file is done via the format string,
        it is called via positional arguments:
        <code>tinyformat::format(formatString, ensemble, timestep)</code>
        Examples:
         - both ensemble and timestep: "files/ensemble%1$03d/time%2$03d.cvol"
        	- only ensemble: "files/ensemble%1$03d.cvol"
        	- only timestep: "files/time%2$03d.cvol"
        See also: https://github.com/c42f/tinyformat
        
        The ensemble configuration is saved in a json file.
		)")
		.def(py::init<>())
		.def(py::init<const std::string&>(),
			py::doc("Loads the ensemble factory settings from the json file specified by the given filename."))
		.def(py::init< const std::string&, int, int, int, int>(),
			py::doc("Creates the ensemble factory from the given parameters"),
			py::arg("format_string"),
			py::arg("start_ensemble"),
			py::arg("num_ensembles"),
			py::arg("start_timestep"),
			py::arg("num_timesteps"))
		.def("save", &VolumeEnsembleFactory::save,
			py::doc(" Saves the ensemble factory settings to the file specified by the given filename."))
		.def("load_volume", &VolumeEnsembleFactory::loadVolume,
			py::doc(" Loads the volume at the given ensemble and time index. If the volume could not be found(i.e.illegal ensemble or time index, or the dataset is missing), an empty pointer is returned."),
			py::arg("ensemble"), py::arg("time"))
		.def_property("format_string", &VolumeEnsembleFactory::formatString, &VolumeEnsembleFactory::setFormatString)
		.def_property("start_timestep", &VolumeEnsembleFactory::startTimestep, &VolumeEnsembleFactory::setStartTimestep)
		.def_property("step_timestep", &VolumeEnsembleFactory::stepTimestep, &VolumeEnsembleFactory::setStepTimestep)
		.def_property("num_timesteps", &VolumeEnsembleFactory::numTimesteps, &VolumeEnsembleFactory::setNumTimesteps)
		.def_property("start_ensemble", &VolumeEnsembleFactory::startEnsemble, &VolumeEnsembleFactory::setStartEnsemble)
		.def_property("step_ensemble", &VolumeEnsembleFactory::stepEnsemble, &VolumeEnsembleFactory::setStepEnsemble)
		.def_property("num_ensembles", &VolumeEnsembleFactory::numEnsembles, &VolumeEnsembleFactory::setNumEnsembles)
	;
}

END_RENDERER_NAMESPACE
