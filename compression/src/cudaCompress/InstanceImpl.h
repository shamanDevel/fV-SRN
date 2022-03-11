#ifndef __TUM3D_CUDACOMPRESS__INSTANCE_IMPL_H__
#define __TUM3D_CUDACOMPRESS__INSTANCE_IMPL_H__


#include <cudaCompress/global.h>

#include <vector>

#include <cuda_runtime.h>

#include <cudaCompress/util/CudaTimer.h>

#include <cudaCompress/HuffmanTable.h>
#include <cudaCompress/Timing.h>

#include <cudaCompress/profiler/profiler.hpp>


namespace cudaCompress
{

class ReducePlan;
class ScanPlan;

struct HuffmanGPUStreamInfo;

class Instance
{
public:
    Instance();
    ~Instance();

    // pass -1 as the cudaDevice to use the current one
    bool create(int cudaDevice, uint streamCountMax, uint elemCountPerStreamMax, uint codingBlockSize, uint log2HuffmanDistinctSymbolCountMax);
    void destroy();

    void setTimingDetail(ETimingDetail detail);
    ETimingDetail getTimingDetail() const { return m_timingDetail; }
    void getTimings(std::vector<std::string>& names, std::vector<float>& times);
    void printTimings();
    void resetTimings();

    // get a buffer of the specified size in GPU memory
    byte* getByteBuffer(size_t bytes);
    template<typename T>
    T* getBuffer(size_t count) { return (T*)getByteBuffer(count * sizeof(T)); }
    // release the last buffer(s) returned from getBuffer
    void releaseBuffer();
    void releaseBuffers(uint bufferCount);


    int m_cudaDevice;

    uint m_streamCountMax;
    uint m_elemCountPerStreamMax;
    uint m_codingBlockSize;
    uint m_log2HuffmanDistinctSymbolCountMax;


    ReducePlan* m_pReducePlan;
    ScanPlan*   m_pScanPlan;


    cudaStream_t m_stream;


    // TIER 1
    struct EncodeResources
    {
        // encode*:
        // used for downloads
        uint* pCodewordBuffer;
        uint* pOffsetBuffer;
        uint* pEncodeCodewords;
        uint* pEncodeCodewordLengths;

        HuffmanGPUStreamInfo* pEncodeSymbolStreamInfos;

        std::vector<HuffmanEncodeTable> symbolEncodeTables;
        std::vector<HuffmanEncodeTable> zeroCountEncodeTables;

        cudaEvent_t encodeFinishedEvent;

        // decode*:
        // decode resources are multi-buffered to avoid having to sync too often
        struct DecodeResources
        {
            cudaEvent_t syncEvent;

            std::vector<HuffmanDecodeTable> symbolDecodeTables;
            std::vector<HuffmanDecodeTable> zeroCountDecodeTables;
            byte* pSymbolDecodeTablesBuffer;
            byte* pZeroCountDecodeTablesBuffer;

            uint*                 pCodewordStreams;
            uint*                 pSymbolOffsets;
            uint*                 pZeroCountOffsets;

            HuffmanGPUStreamInfo* pSymbolStreamInfos;
            HuffmanGPUStreamInfo* pZeroCountStreamInfos;

            DecodeResources()
                : syncEvent(0)
                , pSymbolDecodeTablesBuffer(nullptr), pZeroCountDecodeTablesBuffer(nullptr)
                , pCodewordStreams(nullptr), pSymbolOffsets(nullptr), pZeroCountOffsets(nullptr)
                , pSymbolStreamInfos(nullptr), pZeroCountStreamInfos(nullptr) {}
        };
        const static int ms_decodeResourcesCount = 8;
        DecodeResources Decode[ms_decodeResourcesCount];
        int nextDecodeResources;
        DecodeResources& GetDecodeResources() {
            DecodeResources& result = Decode[nextDecodeResources];
            nextDecodeResources = (nextDecodeResources + 1) % ms_decodeResourcesCount;
            return result;
        }


        util::CudaTimerResources timerEncodeLowDetail;
        util::CudaTimerResources timerEncodeHighDetail;
        util::CudaTimerResources timerDecodeLowDetail;
        util::CudaTimerResources timerDecodeHighDetail;

        EncodeResources()
            : pCodewordBuffer(nullptr), pOffsetBuffer(nullptr)
            , pEncodeCodewords(nullptr), pEncodeCodewordLengths(nullptr)
            , pEncodeSymbolStreamInfos(nullptr)
            , encodeFinishedEvent(0)
            , nextDecodeResources(0) {}
    } Encode;


    // TIER 0
    struct HistogramResources
    {
        byte* pUpload;
        cudaEvent_t syncEvent;

        HistogramResources()
            : pUpload(nullptr), syncEvent(0) {}
    } Histogram;

    struct HuffmanResources
    {
        uint* pReadback;
        cudaEvent_t syncEventReadback;

        util::CudaTimerResources timerEncode;
        util::CudaTimerResources timerDecode;

        HuffmanResources()
            : pReadback(nullptr), syncEventReadback(0) {}
    } Huffman;

    struct HuffmanTableResources
    {
        uint* pReadback;

        HuffmanTableResources()
            : pReadback(nullptr) {}
    } HuffmanTable;

    struct RunLengthResources
    {
        uint* pReadback;
        std::vector<cudaEvent_t> syncEventsReadback;

        byte* pUpload;
        cudaEvent_t syncEventUpload;

        util::CudaTimerResources timerEncode;
        util::CudaTimerResources timerDecode;

        RunLengthResources()
            : pReadback(nullptr), pUpload(nullptr), syncEventUpload(0) {}
    } RunLength;


    Profiler* m_pProfiler;

private:
    ETimingDetail m_timingDetail;

    byte* m_dpBuffer;
    size_t m_bufferSize;

    size_t m_bufferOffset;
    std::vector<size_t> m_allocatedSizes;
};

}


#endif
