#include "memtrace.h"
#include "compression.h"

#include <chrono>
#include <streambuf>
#include <sstream>
#include <iostream>

#include <tthresh/compress.h>
#include <tthresh/decompress.h>
#include <cudaCompress/CompressVolume.h>
#include <cudaCompress/Timing.h>

#include "memtrace_api.h"
#include "cudaCompress/InstanceImpl.h"

namespace {
    struct membuf : std::streambuf {
        membuf(char const* base, size_t size) {
            char* p(const_cast<char*>(base));
            this->setg(p, p, p + size);
        }
    };
    struct imemstream : virtual membuf, std::istream {
        imemstream(char const* base, size_t size)
            : membuf(base, size)
            , std::istream(static_cast<std::streambuf*>(this)) {
        }
    };
}

#define STATISTICS_START                                                                                \
    const auto timestart = std::chrono::steady_clock::now();                                            \
    const auto initialTotalMemCpu = compression::memory::totalMemAllocated(memory::Device::CPU);        \
    const auto initialCurrentMemCpu = compression::memory::currentMemAllocated(memory::Device::CPU);    \
    compression::memory::resetPeakMemory(memory::Device::CPU);                                          \
    const auto initialTotalMemGpu = compression::memory::totalMemAllocated(memory::Device::GPU);        \
    const auto initialCurrentMemGpu = compression::memory::currentMemAllocated(memory::Device::GPU);    \
    compression::memory::resetPeakMemory(memory::Device::GPU)

#define STATISTICS_END                                                                                             \
    const auto timeend = std::chrono::steady_clock::now();                                                         \
    const auto totalMemoryCpu = compression::memory::totalMemAllocated(memory::Device::CPU) - initialTotalMemCpu;  \
    const auto peakMemoryCpu = compression::memory::peakMemAllocated(memory::Device::CPU) - initialCurrentMemCpu;  \
    const auto totalMemoryGpu = compression::memory::totalMemAllocated(memory::Device::GPU) - initialTotalMemGpu;  \
    const auto peakMemoryGpu = compression::memory::peakMemAllocated(memory::Device::GPU) - initialCurrentMemGpu;  \
    const auto timeSeconds = std::chrono::duration<double>(timeend - timestart).count();                           \
    const compression::Statistics_t stats{                                                                         \
        std::make_pair<>("time_ms", timeSeconds*1000),                                       \
        std::make_pair<>("total_memory_cpu", totalMemoryCpu),                                \
        std::make_pair<>("peak_memory_cpu", peakMemoryCpu),                                  \
        std::make_pair<>("total_memory_gpu", totalMemoryGpu),                                \
        std::make_pair<>("peak_memory_gpu", peakMemoryGpu)                                   \
    }


std::tuple<compression::CompressedVolume_ptr, compression::Statistics_t> compression::compressTThresh(
    RawVolumeFortranStyle_ptr<double> volume, TThreshTarget target, double targetValue, bool verbose)
{
    //compress
    STATISTICS_START;
    std::stringstream ss;
    tthresh::compress(ss, volume->data(), cast<uint32_t>(volume->dimensions()), target, targetValue, verbose);
    STATISTICS_END;
    //collect memory
    if (ss.fail()) throw std::logic_error("Error while compressing volume, output stream is corrupt");
    auto pos = ss.tellp();
    CompressedVolume_ptr c = std::make_shared<CompressedVolume>(pos);
    ss.read(static_cast<char*>(c->data()), pos);
    return std::make_tuple(c, stats);
}

std::tuple<compression::RawVolumeFortranStyle_ptr<double>, compression::Statistics_t> compression::decompressTThresh(
    CompressedVolume_ptr v, bool verbose)
{
    imemstream in(static_cast<const char*>(v->data()), v->size());
    STATISTICS_START;
    std::vector<uint32_t> dimensions;
    auto d = tthresh::decompress(in, dimensions, verbose, verbose);
    STATISTICS_END;
    auto rv = std::make_shared<RawVolume<double, true>>(std::move(d), cast<size_t>(dimensions));
    return std::make_tuple(rv, stats);
}

std::tuple<compression::CompressedVolume_ptr, compression::Statistics_t> compression::compressTThreshChunked(
    const std::vector<RawVolumeFortranStyle_ptr<double>>& volumes, TThreshTarget target, double targetValue, bool verbose)
{
    const int numChunks = static_cast<int>(volumes.size());
    //compress
    STATISTICS_START;
    std::stringstream ss;
    ss.write(reinterpret_cast<const char*>(&numChunks), sizeof(int));
    for (size_t i = 0; i < volumes.size(); ++i) {
        tthresh::compress(ss, volumes[i]->data(), cast<uint32_t>(volumes[i]->dimensions()), target, targetValue, verbose);
    }
    STATISTICS_END;
    //collect memory
    if (ss.fail()) throw std::logic_error("Error while compressing volume, output stream is corrupt");
    auto pos = ss.tellp();
    CompressedVolume_ptr c = std::make_shared<CompressedVolume>(pos);
    ss.read(static_cast<char*>(c->data()), pos);
    return std::make_tuple(c, stats);
}

std::tuple<std::vector<compression::RawVolumeFortranStyle_ptr<double>>, compression::Statistics_t> compression::decompressTThreshChunked(
    CompressedVolume_ptr v, bool verbose)
{
    imemstream in(static_cast<const char*>(v->data()), v->size());
    int numChunks;
    std::vector<compression::RawVolumeFortranStyle_ptr<double>> outputChunks;
    STATISTICS_START;
    in.read(reinterpret_cast<char*>(&numChunks), sizeof(int));
    outputChunks.resize(numChunks);
    for (int i = 0; i < numChunks; ++i) {
        std::vector<uint32_t> dimensions;
        auto d = tthresh::decompress(in, dimensions, verbose, verbose);
        auto rv = std::make_shared<RawVolume<double, true>>(std::move(d), cast<size_t>(dimensions));
        outputChunks[i] = rv;
    }
    STATISTICS_END;
    return std::make_tuple(outputChunks, stats);
}

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckMsg(msg) __cudaCheckMsg(msg, __FILE__, __LINE__)

#ifdef _DEBUG
#define CHECK_ERROR(err) (cudaSuccess != err || cudaSuccess != (err = cudaDeviceSynchronize()))
#else
#define CHECK_ERROR(err) (cudaSuccess != err)
#endif
static inline void __cudaSafeCall(cudaError err, const char* file, const int line)
{
    if (CHECK_ERROR(err)) {
        fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error : %s.\n", file, line, cudaGetErrorString(err));
#ifdef _DEBUG
        __debugbreak();
#endif
    }
}
static inline void __cudaCheckMsg(const char* errorMessage, const char* file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (CHECK_ERROR(err)) {
        fprintf(stderr, "%s(%i) : cudaCheckMsg() CUDA error : %s : (%d) %s.\n", file, line, errorMessage, (int)err, cudaGetErrorString(err));
#ifdef _DEBUG
        __debugbreak();
#endif
    }
}

struct CudaCompressHeader
{
    size_t width;
    size_t height;
    size_t depth;
    size_t bitstreamSize;
    float quantStep;
    int numLevels;
    int numChunks;
};

std::tuple<compression::CompressedVolume_ptr, compression::Statistics_t> compression::compressCUDA(
    RawVolumeCStyle_ptr<float> volume, int numLevels,
    float quantizationStep, bool verbose, int numChunks)
{
    /*
     * if numChunks>1, chunk over the slowest moving dimension, aka 'width'.
     * This is super simple, but sufficient for now
     */

    if (volume->dimensions().size() != 3)
        throw std::logic_error("CudaCompress only works with 3D volumes");
    size_t original_width = volume->dimensions()[0];
    size_t width;
    size_t height = volume->dimensions()[1];
    size_t depth = volume->dimensions()[2];
    if (numChunks > 1)
    {
        if (original_width % numChunks != 0)
            throw std::logic_error("'width' is not divisble by 'numChunks', can't perform chunking");
        width = original_width / numChunks;
    } else if (numChunks <= 0)
    {
        throw std::logic_error("'numChunks' must be positive");
    } else
    { //numChunks==1
        width = original_width;
    }

    STATISTICS_START;

    using namespace cudaCompress;
    const bool doRLEOnlyOnLvl0 = true;
    const size_t elemCountTotal = width * height * depth;
    const size_t channelCount = 1;
    std::vector<float*> data(channelCount);
    const size_t chunkStride = width * volume->strides()[0];

    // allocate GPU arrays and upload data
    std::vector<float*> dpImages(channelCount);
    for (size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaMalloc(&dpImages[c], elemCountTotal * sizeof(float)));
    }

    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    float time = 0.0f;

    uint huffmanBits = 0;
    GPUResources::Config config = CompressVolumeResources::getRequiredResources(width, height, depth, (uint)channelCount, huffmanBits);
    GPUResources shared;
    shared.create(config);
    CompressVolumeResources res;
    res.create(shared.getConfig());
    setTimingDetail(shared.m_pCuCompInstance, verbose ? TIMING_DETAIL_HIGH : TIMING_DETAIL_NONE);

    std::vector<uint> bitStreamGlobal;
    std::vector<size_t> localBitstreamSizes;
    std::vector<std::vector<uint>> bitStreamsLocal(channelCount);

    for (int chunk = 0; chunk < numChunks; ++chunk)
    {
        data[0] = volume->data() + chunk*chunkStride;
        // upload data
        for (size_t c = 0; c < channelCount; c++) {
            cudaSafeCall(cudaMemcpy(dpImages[c], data[c], elemCountTotal * sizeof(float), cudaMemcpyHostToDevice));
        }

        //cudaProfilerStart();
        cudaSafeCall(cudaEventRecord(start));

        for (size_t c = 0; c < channelCount; c++) {
            bitStreamsLocal[c].clear();
            compressVolumeFloatQuantFirst(shared, res, dpImages[c], width, height, depth, numLevels, bitStreamsLocal[c], quantizationStep, doRLEOnlyOnLvl0);
        }

        cudaSafeCall(cudaEventRecord(end));
        //cudaProfilerStop();

        cudaSafeCall(cudaEventSynchronize(end));
        cudaSafeCall(cudaEventElapsedTime(&time, start, end));
        float throughput = float(elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
        if (verbose) {
            printf("Compress chunk %d:   %6.2f ms  (%7.2f MPix/s  %7.2f Mfloat/s)\n", chunk, time, throughput, throughput * float(channelCount));
            printf("Detailed Timings:\n");
            printTimings(shared.m_pCuCompInstance);
            resetTimings(shared.m_pCuCompInstance);
            printf("\n");
        }

        //insert into global bitstream
        localBitstreamSizes.push_back(bitStreamsLocal[0].size());
        bitStreamGlobal.insert(bitStreamGlobal.end(), bitStreamsLocal[0].begin(), bitStreamsLocal[0].end());
    }
    //cleanup
    res.destroy();
    shared.destroy();
    for (size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaFree(dpImages[c]));
    }

    //collect memory
    CudaCompressHeader header;
    header.width = width;
    header.height = height;
    header.depth = depth;
    header.bitstreamSize = bitStreamGlobal.size();
    header.quantStep = quantizationStep;
    header.numLevels = numLevels;
    header.numChunks = numChunks;
    size_t outputSize = sizeof(CudaCompressHeader) + sizeof(size_t) * numChunks + sizeof(uint) * bitStreamGlobal.size();
    CompressedVolume_ptr c = std::make_shared<CompressedVolume>(outputSize);
    memcpy(c->data(), &header, sizeof(CudaCompressHeader));
    memcpy(static_cast<uint8_t*>(c->data()) + sizeof(CudaCompressHeader),
        localBitstreamSizes.data(), sizeof(size_t) * localBitstreamSizes.size());
    memcpy(static_cast<uint8_t*>(c->data()) + sizeof(CudaCompressHeader) + sizeof(size_t) * numChunks,
        bitStreamGlobal.data(), sizeof(uint) * bitStreamGlobal.size());

    STATISTICS_END;

    return std::make_tuple(c, stats);
}

std::tuple<compression::RawVolumeCStyle_ptr<float>, compression::Statistics_t> compression::decompressCUDA(
    CompressedVolume_ptr v, bool verbose)
{
    STATISTICS_START;

    //read dimensions
    CudaCompressHeader header;
    memcpy(&header, v->data(), sizeof(CudaCompressHeader));
    size_t original_width = header.width * header.numChunks;

    using namespace cudaCompress;
    const bool doRLEOnlyOnLvl0 = true;
    const size_t elemCountTotal = header.width * header.height * header.depth;
    constexpr const size_t channelCount = 1;
    const size_t* localBitstreamSizes = reinterpret_cast<const size_t*>(static_cast<const uint8_t*>(v->data()) + sizeof(CudaCompressHeader));
    const uint* bitStreamGlobal = reinterpret_cast<const uint*>(
        static_cast<const uint8_t*>(v->data()) + sizeof(CudaCompressHeader) + sizeof(size_t)*header.numChunks);
    std::vector<const uint*> bitStreams(channelCount);
    size_t globalBitstreamOffset = 0;
    //bitStreams[0] = reinterpret_cast<const uint*>(static_cast<const uint8_t*>(v->data()) + sizeof(CudaCompressHeader));

    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    float time = 0.0f;

    // allocate GPU arrays
    std::vector<float*> dpImages(channelCount);
    for (size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaMalloc(&dpImages[c], elemCountTotal * sizeof(float)));
        cudaSafeCall(cudaMemset(dpImages[c], 0, elemCountTotal * sizeof(float)));
    }

    uint huffmanBits = 0;
    GPUResources::Config config = CompressVolumeResources::getRequiredResources(header.width, header.height, header.depth, (uint)channelCount, huffmanBits);
    GPUResources shared;
    shared.create(config);
    CompressVolumeResources res;
    res.create(shared.getConfig());
    setTimingDetail(shared.m_pCuCompInstance, verbose ? TIMING_DETAIL_HIGH : TIMING_DETAIL_NONE);

    assert(channelCount == 1);
    RawVolumeCStyle_ptr<float> vol = std::make_shared<RawVolume<float, false>>(
        std::vector<size_t>{ original_width, header.height, header.depth });
    const size_t chunkStride = header.width * vol->strides()[0];

    for (int chunk = 0; chunk < header.numChunks; ++chunk)
    {
        bitStreams[0] = bitStreamGlobal + globalBitstreamOffset;
        globalBitstreamOffset += localBitstreamSizes[chunk];
        cudaSafeCall(cudaHostRegister(const_cast<uint*>(bitStreams[0]), localBitstreamSizes[chunk] * sizeof(uint), cudaHostRegisterDefault));

        cudaSafeCall(cudaEventRecord(start));

        std::vector<VolumeChannel> channels(channelCount);
        for (size_t c = 0; c < channelCount; c++) {
            channels[c].dpImage = dpImages[c];
            channels[c].pBits = bitStreams[c];
            channels[c].bitCount = localBitstreamSizes[chunk] * sizeof(uint) * 8;
            channels[c].quantizationStepLevel0 = header.quantStep;
        }
        if (channelCount == 1) {
            uint bitCount = localBitstreamSizes[chunk] * sizeof(uint) * 8;
            decompressVolumeFloatQuantFirst(shared, res, dpImages.front(), header.width, header.height, header.depth, header.numLevels, bitStreams.front(), bitCount, header.quantStep, doRLEOnlyOnLvl0);
        }
        else {
            decompressVolumeFloatQuantFirstMultiChannel(shared, res, channels.data(), (uint)channels.size(), header.width, header.height, header.depth, header.numLevels, doRLEOnlyOnLvl0);
        }

        cudaSafeCall(cudaEventRecord(end));
        //cudaProfilerStop();

        for (size_t c = 0; c < channelCount; c++) {
            cudaSafeCall(cudaHostUnregister(const_cast<uint*>(bitStreams[c])));
        }

        cudaSafeCall(cudaEventSynchronize(end));
        cudaSafeCall(cudaEventElapsedTime(&time, start, end));
        float throughput = float(elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
        if (verbose) {
            printf("Decompress chunk %d: %6.2f ms  (%7.2f MPix/s  %7.2f Mfloat/s)\n", chunk, time, throughput, throughput * float(channelCount));
            printf("Detailed Timings:\n");
            printTimings(shared.m_pCuCompInstance);
            resetTimings(shared.m_pCuCompInstance);
            printf("\n");
        }

        //copy back to host
        cudaSafeCall(cudaMemcpy(vol->data() + chunk*chunkStride, dpImages[0], elemCountTotal * sizeof(float), cudaMemcpyDeviceToHost));
    }
    STATISTICS_END;

    //cleanup
    res.destroy();
    shared.destroy();
    for (size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaFree(dpImages[c]));
    }

    return std::make_tuple(vol, stats);
}


std::tuple<compression::CompressedVolume_ptr, compression::Statistics_t> compression::compressCUDAChunked(
    const std::vector<RawVolumeCStyle_ptr<float>>& volumes, int numLevels,
    float quantizationStep, bool verbose)
{
    size_t width, height, depth;
    /**
     * Check that all chunks have the same shape.
     */
    if (volumes.empty()) throw std::logic_error("Attempt to compress empty list of chunks");
    if (volumes[0]->dimensions().size() != 3)
        throw std::logic_error("CudaCompress only works with 3D volumes");
    width = volumes[0]->dimensions()[0];
    height = volumes[0]->dimensions()[1];
    depth = volumes[0]->dimensions()[2];
    for (size_t i=1; i<volumes.size(); ++i)
    {
        if (volumes[i]->dimensions().size() != 3)
            throw std::logic_error("CudaCompress only works with 3D volumes");
        if (volumes[i]->dimensions()[0] != width)
            throw std::logic_error("All chunks must have the same size");
        if (volumes[i]->dimensions()[1] != height)
            throw std::logic_error("All chunks must have the same size");
        if (volumes[i]->dimensions()[2] != depth)
            throw std::logic_error("All chunks must have the same size");
    }
    const int numChunks = static_cast<int>(volumes.size());

    STATISTICS_START;

    using namespace cudaCompress;
    const bool doRLEOnlyOnLvl0 = true;
    const size_t elemCountTotal = width * height * depth;
    const size_t channelCount = 1;
    std::vector<float*> data(channelCount);

    // allocate GPU arrays and upload data
    std::vector<float*> dpImages(channelCount);
    for (size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaMalloc(&dpImages[c], elemCountTotal * sizeof(float)));
    }

    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    float time = 0.0f;

    uint huffmanBits = 0;
    GPUResources::Config config = CompressVolumeResources::getRequiredResources(width, height, depth, (uint)channelCount, huffmanBits);
    GPUResources shared;
    shared.create(config);
    CompressVolumeResources res;
    res.create(shared.getConfig());
    setTimingDetail(shared.m_pCuCompInstance, verbose ? TIMING_DETAIL_HIGH : TIMING_DETAIL_NONE);

    std::vector<uint> bitStreamGlobal;
    std::vector<size_t> localBitstreamSizes;
    std::vector<std::vector<uint>> bitStreamsLocal(channelCount);

    for (int chunk = 0; chunk < numChunks; ++chunk)
    {
        data[0] = volumes[chunk]->data();
        // upload data
        for (size_t c = 0; c < channelCount; c++) {
            cudaSafeCall(cudaMemcpy(dpImages[c], data[c], elemCountTotal * sizeof(float), cudaMemcpyHostToDevice));
        }

        //cudaProfilerStart();
        cudaSafeCall(cudaEventRecord(start));

        for (size_t c = 0; c < channelCount; c++) {
            bitStreamsLocal[c].clear();
            compressVolumeFloatQuantFirst(shared, res, dpImages[c], width, height, depth, numLevels, bitStreamsLocal[c], quantizationStep, doRLEOnlyOnLvl0);
        }

        cudaSafeCall(cudaEventRecord(end));
        //cudaProfilerStop();

        cudaSafeCall(cudaEventSynchronize(end));
        cudaSafeCall(cudaEventElapsedTime(&time, start, end));
        float throughput = float(elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
        if (verbose) {
            printf("Compress chunk %d:   %6.2f ms  (%7.2f MPix/s  %7.2f Mfloat/s)\n", chunk, time, throughput, throughput * float(channelCount));
            printf("Detailed Timings:\n");
            printTimings(shared.m_pCuCompInstance);
            resetTimings(shared.m_pCuCompInstance);
            printf("\n");
        }

        //insert into global bitstream
        localBitstreamSizes.push_back(bitStreamsLocal[0].size());
        bitStreamGlobal.insert(bitStreamGlobal.end(), bitStreamsLocal[0].begin(), bitStreamsLocal[0].end());
    }
    //cleanup
    res.destroy();
    shared.destroy();
    for (size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaFree(dpImages[c]));
    }

    //collect memory
    CudaCompressHeader header;
    header.width = width;
    header.height = height;
    header.depth = depth;
    header.bitstreamSize = bitStreamGlobal.size();
    header.quantStep = quantizationStep;
    header.numLevels = numLevels;
    header.numChunks = numChunks;
    size_t outputSize = sizeof(CudaCompressHeader) + sizeof(size_t) * numChunks + sizeof(uint) * bitStreamGlobal.size();
    CompressedVolume_ptr c = std::make_shared<CompressedVolume>(outputSize);
    memcpy(c->data(), &header, sizeof(CudaCompressHeader));
    memcpy(static_cast<uint8_t*>(c->data()) + sizeof(CudaCompressHeader),
        localBitstreamSizes.data(), sizeof(size_t) * localBitstreamSizes.size());
    memcpy(static_cast<uint8_t*>(c->data()) + sizeof(CudaCompressHeader) + sizeof(size_t) * numChunks,
        bitStreamGlobal.data(), sizeof(uint) * bitStreamGlobal.size());

    STATISTICS_END;

    return std::make_tuple(c, stats);
}

std::tuple<std::vector<compression::RawVolumeCStyle_ptr<float>>, compression::Statistics_t>
    compression::decompressCUDAChunked(
    CompressedVolume_ptr v, bool verbose)
{
    STATISTICS_START;

    //read dimensions
    CudaCompressHeader header;
    memcpy(&header, v->data(), sizeof(CudaCompressHeader));

    using namespace cudaCompress;
    const bool doRLEOnlyOnLvl0 = true;
    const size_t elemCountTotal = header.width * header.height * header.depth;
    constexpr const size_t channelCount = 1;
    const size_t* localBitstreamSizes = reinterpret_cast<const size_t*>(static_cast<const uint8_t*>(v->data()) + sizeof(CudaCompressHeader));
    const uint* bitStreamGlobal = reinterpret_cast<const uint*>(
        static_cast<const uint8_t*>(v->data()) + sizeof(CudaCompressHeader) + sizeof(size_t) * header.numChunks);
    std::vector<const uint*> bitStreams(channelCount);
    size_t globalBitstreamOffset = 0;
    //bitStreams[0] = reinterpret_cast<const uint*>(static_cast<const uint8_t*>(v->data()) + sizeof(CudaCompressHeader));

    cudaEvent_t start, end;
    cudaSafeCall(cudaEventCreate(&start));
    cudaSafeCall(cudaEventCreate(&end));
    float time = 0.0f;

    // allocate GPU arrays
    std::vector<float*> dpImages(channelCount);
    for (size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaMalloc(&dpImages[c], elemCountTotal * sizeof(float)));
        cudaSafeCall(cudaMemset(dpImages[c], 0, elemCountTotal * sizeof(float)));
    }

    uint huffmanBits = 0;
    GPUResources::Config config = CompressVolumeResources::getRequiredResources(header.width, header.height, header.depth, (uint)channelCount, huffmanBits);
    GPUResources shared;
    shared.create(config);
    CompressVolumeResources res;
    res.create(shared.getConfig());
    setTimingDetail(shared.m_pCuCompInstance, verbose ? TIMING_DETAIL_HIGH : TIMING_DETAIL_NONE);

    assert(channelCount == 1);
    std::vector<RawVolumeCStyle_ptr<float>> outputVolumes;
   
    for (int chunk = 0; chunk < header.numChunks; ++chunk)
    {
        bitStreams[0] = bitStreamGlobal + globalBitstreamOffset;
        globalBitstreamOffset += localBitstreamSizes[chunk];
        cudaSafeCall(cudaHostRegister(const_cast<uint*>(bitStreams[0]), localBitstreamSizes[chunk] * sizeof(uint), cudaHostRegisterDefault));

        cudaSafeCall(cudaEventRecord(start));

        std::vector<VolumeChannel> channels(channelCount);
        for (size_t c = 0; c < channelCount; c++) {
            channels[c].dpImage = dpImages[c];
            channels[c].pBits = bitStreams[c];
            channels[c].bitCount = localBitstreamSizes[chunk] * sizeof(uint) * 8;
            channels[c].quantizationStepLevel0 = header.quantStep;
        }
        if (channelCount == 1) {
            uint bitCount = localBitstreamSizes[chunk] * sizeof(uint) * 8;
            decompressVolumeFloatQuantFirst(shared, res, dpImages.front(), header.width, header.height, header.depth, header.numLevels, bitStreams.front(), bitCount, header.quantStep, doRLEOnlyOnLvl0);
        }
        else {
            decompressVolumeFloatQuantFirstMultiChannel(shared, res, channels.data(), (uint)channels.size(), header.width, header.height, header.depth, header.numLevels, doRLEOnlyOnLvl0);
        }

        cudaSafeCall(cudaEventRecord(end));
        //cudaProfilerStop();

        for (size_t c = 0; c < channelCount; c++) {
            cudaSafeCall(cudaHostUnregister(const_cast<uint*>(bitStreams[c])));
        }

        cudaSafeCall(cudaEventSynchronize(end));
        cudaSafeCall(cudaEventElapsedTime(&time, start, end));
        float throughput = float(elemCountTotal) * 1000.0f / (time * 1024.0f * 1024.0f);
        if (verbose) {
            printf("Decompress chunk %d: %6.2f ms  (%7.2f MPix/s  %7.2f Mfloat/s)\n", chunk, time, throughput, throughput * float(channelCount));
            printf("Detailed Timings:\n");
            printTimings(shared.m_pCuCompInstance);
            resetTimings(shared.m_pCuCompInstance);
            printf("\n");
        }

        //copy back to host
        RawVolumeCStyle_ptr<float> vol = std::make_shared<RawVolume<float, false>>(
            std::vector<size_t>{ header.width, header.height, header.depth });
        cudaSafeCall(cudaMemcpy(vol->data(), dpImages[0], elemCountTotal * sizeof(float), cudaMemcpyDeviceToHost));
        outputVolumes.push_back(vol);
    }
    STATISTICS_END;

    //cleanup
    res.destroy();
    shared.destroy();
    for (size_t c = 0; c < channelCount; c++) {
        cudaSafeCall(cudaFree(dpImages[c]));
    }

    return std::make_tuple(outputVolumes, stats);
}

namespace compression
{
    //defined in compression_cu.cu
    void fillChunkFloat(
        cudaSurfaceObject_t dst,
        const float* src,
        int sizeX, int sizeY, int sizeZ,
        int strideX, int strideY, int strideZ,
        cudaStream_t stream);
    void fillChunkUChar(
        cudaSurfaceObject_t dst,
        const float* src,
        int sizeX, int sizeY, int sizeZ,
        int strideX, int strideY, int strideZ,
        cudaStream_t stream);
    void fillChunkUShort(
        cudaSurfaceObject_t dst,
        const float* src,
        int sizeX, int sizeY, int sizeZ,
        int strideX, int strideY, int strideZ,
        cudaStream_t stream);
}
struct compression::CudaCompressInteractiveDecompression::impl
{
    static constexpr const bool doRLEOnlyOnLvl0 = true;
    static constexpr const size_t channelCount = 1;

    CompressedVolume_ptr v_;
    CudaCompressHeader header_;
    size_t elemCountTotal_; //per chunk
    const size_t* localBitstreamSizes_;
    const cudaCompress::uint* bitStreamGlobal_;

    std::vector<const cudaCompress::uint*> bitStreams_;
    std::vector<float*> dpImages_;
    cudaCompress::GPUResources shared_;
    cudaCompress::CompressVolumeResources res_;

    std::chrono::time_point<std::chrono::steady_clock> timestart_;
    ptrdiff_t initialTotalMemCpu_;
    ptrdiff_t initialCurrentMemCpu_;
    ptrdiff_t initialTotalMemGpu_;
    ptrdiff_t initialCurrentMemGpu_;

    impl(CompressedVolume_ptr v, bool verbose=false)
        : v_(v)
    {
        //start statistics
        timestart_ = std::chrono::steady_clock::now();
        initialTotalMemCpu_ = compression::memory::totalMemAllocated(memory::Device::CPU);
        initialCurrentMemCpu_ = compression::memory::currentMemAllocated(memory::Device::CPU);
        compression::memory::resetPeakMemory(memory::Device::CPU);
        initialTotalMemGpu_ = compression::memory::totalMemAllocated(memory::Device::GPU);
        initialCurrentMemGpu_ = compression::memory::currentMemAllocated(memory::Device::GPU);
        compression::memory::resetPeakMemory(memory::Device::GPU);

        using namespace cudaCompress;

        memcpy(&header_, v->data(), sizeof(CudaCompressHeader));
        elemCountTotal_ = header_.width * header_.height * header_.depth;
        localBitstreamSizes_ = reinterpret_cast<const size_t*>(static_cast<const uint8_t*>(v->data()) + sizeof(CudaCompressHeader));
        bitStreamGlobal_ = reinterpret_cast<const cudaCompress::uint*>(
            static_cast<const uint8_t*>(v->data()) + sizeof(CudaCompressHeader) + sizeof(size_t) * header_.numChunks);

        bitStreams_.resize(channelCount);
        dpImages_.resize(channelCount);
        for (size_t c = 0; c < channelCount; c++) {
            cudaSafeCall(cudaMalloc(&dpImages_[c], elemCountTotal_ * sizeof(float)));
        }

        uint huffmanBits = 0;
        GPUResources::Config config = CompressVolumeResources::getRequiredResources(
            header_.width, header_.height, header_.depth, (uint)channelCount, huffmanBits);
        shared_.create(config);
        res_.create(shared_.getConfig());
        setTimingDetail(shared_.m_pCuCompInstance, verbose ? TIMING_DETAIL_HIGH : TIMING_DETAIL_NONE);
    }

    ~impl()
    {
        res_.destroy();
        shared_.destroy();
        for (size_t c = 0; c < channelCount; c++) {
            cudaSafeCall(cudaFree(dpImages_[c]));
        }
    }

    compression::Statistics_t globalStatistics() const
    {
        const auto timeend = std::chrono::steady_clock::now();
        const auto totalMemoryCpu = compression::memory::totalMemAllocated(memory::Device::CPU) - initialTotalMemCpu_;
        const auto peakMemoryCpu = compression::memory::peakMemAllocated(memory::Device::CPU) - initialCurrentMemCpu_;
        const auto totalMemoryGpu = compression::memory::totalMemAllocated(memory::Device::GPU) - initialTotalMemGpu_;
        const auto peakMemoryGpu = compression::memory::peakMemAllocated(memory::Device::GPU) - initialCurrentMemGpu_;
        const auto timeSeconds = std::chrono::duration<double>(timeend - timestart_).count();
        return compression::Statistics_t{
            std::make_pair<>("time_ms", timeSeconds * 1000), std::make_pair<>("total_memory_cpu", totalMemoryCpu),
            std::make_pair<>("peak_memory_cpu", peakMemoryCpu), std::make_pair<>("total_memory_gpu", totalMemoryGpu),
            std::make_pair<>("peak_memory_gpu", peakMemoryGpu)
        };
    }

    compression::Statistics_t decompress(int chunk, cudaSurfaceObject_t target, DataType targetDtype)
    {
        using namespace cudaCompress;
        STATISTICS_START;

        // decompress

        size_t offset = 0;
        for (int i = 0; i < chunk; ++i) offset += localBitstreamSizes_[i];
        bitStreams_[0] = bitStreamGlobal_ + offset;
        cudaSafeCall(cudaHostRegister(const_cast<uint*>(bitStreams_[0]), localBitstreamSizes_[chunk] * sizeof(uint), cudaHostRegisterDefault));

        for (size_t c = 0; c < channelCount; c++) {
            cudaSafeCall(cudaMemset(dpImages_[c], 0, elemCountTotal_ * sizeof(float)));
        }
        std::vector<VolumeChannel> channels(channelCount);
        for (size_t c = 0; c < channelCount; c++) {
            channels[c].dpImage = dpImages_[c];
            channels[c].pBits = bitStreams_[c];
            channels[c].bitCount = localBitstreamSizes_[chunk] * sizeof(uint) * 8;
            channels[c].quantizationStepLevel0 = header_.quantStep;
        }
        if (channelCount == 1) {
            uint bitCount = localBitstreamSizes_[chunk] * sizeof(uint) * 8;
            decompressVolumeFloatQuantFirst(shared_, res_, dpImages_.front(), 
                header_.width, header_.height, header_.depth,
                header_.numLevels, bitStreams_.front(), bitCount, 
                header_.quantStep, doRLEOnlyOnLvl0);
        }
        else {
            decompressVolumeFloatQuantFirstMultiChannel(shared_, res_,
                channels.data(), (uint)channels.size(),
                header_.width, header_.height, header_.depth,
                header_.numLevels, doRLEOnlyOnLvl0);
        }

        cudaSafeCall(cudaHostUnregister(const_cast<uint*>(bitStreams_[0])));

        // copy dpImages[0] to target with dtype conversion
        cudaStream_t stream = shared_.m_pCuCompInstance->m_stream;
        switch (targetDtype)
        {
        case DataType::TypeFloat:
            fillChunkFloat(target, dpImages_[0],
                header_.width, header_.height, header_.depth,
                header_.height * header_.depth, header_.depth, 1,
                stream);
            break;
        case DataType::TypeUChar:
            fillChunkUChar(target, dpImages_[0],
                header_.width, header_.height, header_.depth,
                header_.height * header_.depth, header_.depth, 1,
                stream);
            break;
        case DataType::TypeUShort:
            fillChunkUShort(target, dpImages_[0],
                header_.width, header_.height, header_.depth,
                header_.height * header_.depth, header_.depth, 1,
                stream);
            break;
        default:
            throw std::runtime_error("Unknown dtype");
        }
        cudaSafeCall(cudaStreamSynchronize(stream));

        STATISTICS_END;
        return stats;
    }
};

compression::CudaCompressInteractiveDecompression::CudaCompressInteractiveDecompression(CompressedVolume_ptr v)
    : pImpl(std::make_unique<impl>(v))
{
}

compression::CudaCompressInteractiveDecompression::~CudaCompressInteractiveDecompression() = default;

int compression::CudaCompressInteractiveDecompression::chunkWidth() const
{
    return pImpl->header_.width;
}

int compression::CudaCompressInteractiveDecompression::chunkHeight() const
{
    return pImpl->header_.height;
}

int compression::CudaCompressInteractiveDecompression::chunkDepth() const
{
    return pImpl->header_.depth;
}

int compression::CudaCompressInteractiveDecompression::numChunks() const
{
    return pImpl->header_.numChunks;
}

compression::Statistics_t compression::CudaCompressInteractiveDecompression::decompress(int chunk, cudaSurfaceObject_t target,
    DataType targetDtype)
{
    return pImpl->decompress(chunk, target, targetDtype);
}

compression::Statistics_t compression::CudaCompressInteractiveDecompression::globalStatistics()
{
    return pImpl->globalStatistics();
}
