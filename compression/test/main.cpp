#include <iostream>
#include <filesystem>

#include <compression.h>
#include <memtrace_api.h>
#include "rawfile.h"

struct Stats
{
    float Range;
    float MaxE;
    float RMSE;
    float PSNR;
    float SNR;
};

int benchmarkCompression(
    const std::string& filenameOrig,
    uint width, uint height, uint depth,
    uint cudaCompressNumLevels, float cudaCompressQuantStep,
    compression::TThreshTarget tthreshTarget, double tthreshTargetValue,
    const std::string& filenameOutCudaCompress,
    const std::string& filenameOutTThresh)
{
    using namespace compression;
    //read file
    size_t elemCountTotal = size_t(width) * height * depth;
    RawVolumeCStyle_ptr<float> dataFloat = std::make_shared<RawVolume<float, false>>(
        std::vector<size_t>{width, height, depth});
    if (!readFloatRaw(filenameOrig, elemCountTotal, dataFloat->data())) {
        printf("Failed opening file %s\n", filenameOrig.c_str());
        return -1;
    }
    //downsample for testing. Also make non-cubic to test C vs. Fortran order
    const int downscale = 1;
    //dataFloat = dataFloat->slice({
    //    Slice(0, int(width*0.5), downscale), Slice(0, int(height*0.8), downscale), Slice(0, int(depth*0.6), downscale)
    //    });
    elemCountTotal = dataFloat->numel();
    
    std::chrono::high_resolution_clock::time_point timenow;
    ptrdiff_t initialTotalMemCpu, initialCurrentMemCpu;
    ptrdiff_t initialTotalMemGpu, initialCurrentMemGpu;

    // cudaCompress with chunks
    std::cout << "\n====================================\n CUDA COMPRESS with chunks \n====================================\n" << std::endl;
    const int numChunks = 8;
    timenow = std::chrono::high_resolution_clock::now();
    auto [compressedVolumeCuda2, stats5] = compressCUDA(
        dataFloat, cudaCompressNumLevels, cudaCompressQuantStep, true, numChunks);
    double timeCudaCompressionSeconds2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count() / 1000000.;

    initialTotalMemCpu = compression::memory::totalMemAllocated(memory::Device::CPU);
    initialCurrentMemCpu = compression::memory::currentMemAllocated(memory::Device::CPU);
    compression::memory::resetPeakMemory(memory::Device::CPU);
    initialTotalMemGpu = compression::memory::totalMemAllocated(memory::Device::GPU);
    initialCurrentMemGpu = compression::memory::currentMemAllocated(memory::Device::GPU);
    compression::memory::resetPeakMemory(memory::Device::GPU);

    timenow = std::chrono::high_resolution_clock::now();
    auto [decompressedVolumeCuda2, stats6] = decompressCUDA(compressedVolumeCuda2, true);
    //auto decompressedVolumeCuda = dataFloat;
    double timeCudaDecompressionSeconds2 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count() / 1000000.;

    auto totalMemoryCpuCuda2 = compression::memory::totalMemAllocated(memory::Device::CPU) - initialTotalMemCpu;
    auto peakMemoryCpuCuda2 = compression::memory::peakMemAllocated(memory::Device::CPU) - initialCurrentMemCpu;
    auto totalMemoryGpuCuda2 = compression::memory::totalMemAllocated(memory::Device::GPU) - initialTotalMemGpu;
    auto peakMemoryGpuCuda2 = compression::memory::peakMemAllocated(memory::Device::GPU) - initialCurrentMemGpu;


    // cudaCompress
    std::cout << "\n====================================\n CUDA COMPRESS \n====================================\n" << std::endl;
    timenow = std::chrono::high_resolution_clock::now();
    auto [compressedVolumeCuda1, stats3] = compressCUDA(
        dataFloat, cudaCompressNumLevels, cudaCompressQuantStep, true);
    double timeCudaCompressionSeconds1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count() / 1000000.;

    initialTotalMemCpu = compression::memory::totalMemAllocated(memory::Device::CPU);
    initialCurrentMemCpu = compression::memory::currentMemAllocated(memory::Device::CPU);
    compression::memory::resetPeakMemory(memory::Device::CPU);
    initialTotalMemGpu = compression::memory::totalMemAllocated(memory::Device::GPU);
    initialCurrentMemGpu = compression::memory::currentMemAllocated(memory::Device::GPU);
    compression::memory::resetPeakMemory(memory::Device::GPU);

    timenow = std::chrono::high_resolution_clock::now();
    auto [decompressedVolumeCuda1, stats4] = decompressCUDA(compressedVolumeCuda1, true);
    //auto decompressedVolumeCuda = dataFloat;
    double timeCudaDecompressionSeconds1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count() / 1000000.;

    auto totalMemoryCpuCuda1 = compression::memory::totalMemAllocated(memory::Device::CPU) - initialTotalMemCpu;
    auto peakMemoryCpuCuda1 = compression::memory::peakMemAllocated(memory::Device::CPU) - initialCurrentMemCpu;
    auto totalMemoryGpuCuda1 = compression::memory::totalMemAllocated(memory::Device::GPU) - initialTotalMemGpu;
    auto peakMemoryGpuCuda1 = compression::memory::peakMemAllocated(memory::Device::GPU) - initialCurrentMemGpu;

    // TThresh
    std::cout << "\n====================================\n TTHRESH       \n====================================\n" << std::endl;

    //cast to doubles for tthresh
    RawVolumeFortranStyle_ptr<double> dataDouble = dataFloat->cast<double>()->toFortranStyle();

    timenow = std::chrono::high_resolution_clock::now();
    auto [compressedVolumeTThresh, stats1] = compressTThresh(
        dataDouble, tthreshTarget, tthreshTargetValue, true);
    double timeTThreshCompressionSeconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count() / 1000000.;

    initialTotalMemCpu = compression::memory::totalMemAllocated(memory::Device::CPU);
    initialCurrentMemCpu = compression::memory::currentMemAllocated(memory::Device::CPU);
    compression::memory::resetPeakMemory(memory::Device::CPU);
    initialTotalMemGpu = compression::memory::totalMemAllocated(memory::Device::GPU);
    initialCurrentMemGpu = compression::memory::currentMemAllocated(memory::Device::GPU);
    compression::memory::resetPeakMemory(memory::Device::GPU);

    timenow = std::chrono::high_resolution_clock::now();
    auto [decompressedVolumeTThresh_d, stats2] = decompressTThresh(compressedVolumeTThresh, true);
    double timeTThreshDecompressionSeconds = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count() / 1000000.;
    auto decompressedVolumeTThresh = decompressedVolumeTThresh_d->cast<float>()->toCStyle();

    auto totalMemoryCpuTThresh = compression::memory::totalMemAllocated(memory::Device::CPU) - initialTotalMemCpu;
    auto peakMemoryCpuTThresh = compression::memory::peakMemAllocated(memory::Device::CPU) - initialCurrentMemCpu;
    auto totalMemoryGpuTThresh = compression::memory::totalMemAllocated(memory::Device::GPU) - initialTotalMemGpu;
    auto peakMemoryGpuTThresh = compression::memory::peakMemAllocated(memory::Device::GPU) - initialCurrentMemGpu;


    // compute stats
    Stats statsCuda1, statsCuda2, statsTThresh;
    std::cout << "\n====================================\n COMPUTE STATS \n====================================\n" << std::endl;
    computeStatsFloatArrays(dataFloat->data(), decompressedVolumeCuda1->data(), elemCountTotal,
        &statsCuda1.Range, &statsCuda1.MaxE, &statsCuda1.RMSE, &statsCuda1.PSNR, &statsCuda1.SNR);
    computeStatsFloatArrays(dataFloat->data(), decompressedVolumeCuda2->data(), elemCountTotal,
        &statsCuda2.Range, &statsCuda2.MaxE, &statsCuda2.RMSE, &statsCuda2.PSNR, &statsCuda2.SNR);
    computeStatsFloatArrays(dataFloat->data(), decompressedVolumeTThresh->data(), elemCountTotal,
        &statsTThresh.Range, &statsTThresh.MaxE, &statsTThresh.RMSE, &statsTThresh.PSNR, &statsTThresh.SNR);
    double compressionFactorCuda1 = sizeof(float)* float(elemCountTotal) / float(compressedVolumeCuda1->size());
    double compressionFactorCuda2 = sizeof(float) * float(elemCountTotal) / float(compressedVolumeCuda2->size());
    double compressionFactorTThresh = sizeof(float)* float(elemCountTotal) / float(compressedVolumeTThresh->size());

    // and print
    printf(" Stat              | cudaCompress | cuda (chunk) |   TThresh \n");
    printf("-------------------------------------------------------------\n");
    printf("Time Compr. (sec)  | %12.5f | %12.5f | %12.5f \n", timeCudaCompressionSeconds1, timeCudaCompressionSeconds2, timeTThreshCompressionSeconds);
    printf("Time Decmpr. (sec) | %12.5f | %12.5f | %12.5f \n", timeCudaDecompressionSeconds1, timeCudaDecompressionSeconds2, timeTThreshDecompressionSeconds);
    printf("Compression Rate   | %12.5f | %12.5f | %12.5f \n", compressionFactorCuda1, compressionFactorCuda2, compressionFactorTThresh);
    printf("Range              | %12.5f | %12.5f | %12.5f \n", statsCuda1.Range, statsCuda2.Range, statsTThresh.Range);
    printf("MaxE               | %12.5f | %12.5f | %12.5f \n", statsCuda1.MaxE, statsCuda2.MaxE, statsTThresh.MaxE);
    printf("RMSE               | %12.5f | %12.5f | %12.5f \n", statsCuda1.RMSE, statsCuda2.RMSE, statsTThresh.RMSE);
    printf("PSNR               | %12.5f | %12.5f | %12.5f \n", statsCuda1.RMSE, statsCuda2.RMSE, statsTThresh.PSNR);
    printf("SNR                | %12.5f | %12.5f | %12.5f \n", statsCuda1.RMSE, statsCuda2.RMSE, statsTThresh.SNR);
    printf("Total Memory CPU   | %12lld | %12lld | %12lld \n", totalMemoryCpuCuda1, totalMemoryCpuCuda2, totalMemoryCpuTThresh);
    printf("Peak Memory CPU    | %12lld | %12lld | %12lld \n", peakMemoryCpuCuda1, peakMemoryCpuCuda2, peakMemoryCpuTThresh);
    printf("Total Memory GPU   | %12lld | %12lld | %12lld \n", totalMemoryGpuCuda1, totalMemoryGpuCuda2, totalMemoryGpuTThresh);
    printf("Peak Memory GPU    | %12lld | %12lld | %12lld \n", peakMemoryGpuCuda1, peakMemoryGpuCuda2, peakMemoryGpuTThresh);

    if (!filenameOutCudaCompress.empty())
    {
        std::cout << "Save cudaCompress result to " << filenameOutCudaCompress<< std::endl;
        writeFloatRaw(filenameOutCudaCompress, elemCountTotal, decompressedVolumeCuda1->data());
    }
    if (!filenameOutTThresh.empty())
    {
        std::cout << "Save TThresh result to " << filenameOutTThresh << std::endl;
        writeFloatRaw(filenameOutTThresh, elemCountTotal, decompressedVolumeTThresh->data());
    }

    return 0;
}

void testMemtraceFunction();
void testMemtrace()
{
    std::cout << "TEST MEMTRACE" << std::endl;

    auto initialTotalAllocatedCpu = totalMemAllocated(compression::memory::Device::CPU);
    auto initialTotalFreedCpu = totalMemAllocated(compression::memory::Device::CPU);
    auto initialCurrentMemCpu = currentMemAllocated(compression::memory::Device::CPU);
    resetPeakMemory(compression::memory::Device::CPU);

    testMemtraceFunction();

    auto finalTotalAllocatedCpu = totalMemAllocated(compression::memory::Device::CPU);
    auto finalTotalFreedCpu = totalMemAllocated(compression::memory::Device::CPU);
    auto finalCurrentMemCpu = currentMemAllocated(compression::memory::Device::CPU);
    auto finalPeakMemCpu = peakMemAllocated(compression::memory::Device::CPU);

    std::cout << "Mem allocated: " << (finalTotalAllocatedCpu - initialTotalAllocatedCpu) << std::endl;
    std::cout << "Mem freed: " << (finalTotalFreedCpu - initialTotalFreedCpu) << std::endl;
    std::cout << "Initial current: " << initialCurrentMemCpu << ", final: " << finalCurrentMemCpu << std::endl;
    std::cout << "Peak memory: " << (finalPeakMemCpu - initialCurrentMemCpu) << std::endl;
}

int main(int argc, char** argv)
{
    // enable run-time memory check for debug builds
#if defined( _WIN32 ) && ( defined( DEBUG ) || defined( _DEBUG ) )
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    //test memtrace
    testMemtrace();

    // Run benchmark
#if 1
    auto path = std::filesystem::path(__FILE__).parent_path() / "data/Iso_128_128_128_t4000_VelocityX.raw";
    uint width = 128;
    uint height = 128;
    uint depth = 128;
#else
    auto path = std::filesystem::path(__FILE__).parent_path() / "data/ejecta1024_1024_1024_1024.raw";
    uint width = 1024;
    uint height = 1024;
    uint depth = 1024;
#endif
    uint numLevels = 2;
    float quantStep = 0.001f;
    compression::TThreshTarget target = compression::PSNR;
    double targetValue = 40;
    benchmarkCompression(path.string(), width, height, depth, numLevels, quantStep, target, targetValue, "", "");
}