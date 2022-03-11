#include "memtrace.h"
#include <cudaCompress/InstanceImpl.h>

#include <algorithm>
#include <cassert>

#include <cudaCompress/util.h>
#include <cudaCompress/cudaUtil.h>

#include <cudaCompress/reduce/reduce_plan.h>
#include <cudaCompress/scan/scan_plan.h>

#include <cudaCompress/Histogram.h>
#include <cudaCompress/Huffman.h>
#include <cudaCompress/HuffmanTable.h>
#include <cudaCompress/RunLength.h>
#include <cudaCompress/PackInc.h>
#include <cudaCompress/Encode.h>

#include <cudaCompress/profiler/profilerlogwriter.hpp>

#include <omp.h>


namespace cudaCompress
{

Instance::Instance()
    : m_cudaDevice(-1)
    , m_streamCountMax(0)
    , m_elemCountPerStreamMax(0)
    , m_codingBlockSize(0)
    , m_log2HuffmanDistinctSymbolCountMax(0)
    , m_pReducePlan(nullptr), m_pScanPlan(nullptr)
    , m_stream(0)
    , m_timingDetail(TIMING_DETAIL_NONE)
    , m_dpBuffer(nullptr), m_bufferSize(0), m_bufferOffset(0)
    , m_pProfiler(nullptr)
{
}

Instance::~Instance()
{
    assert(m_cudaDevice == -1);
}

bool Instance::create(int cudaDevice, uint streamCountMax, uint elemCountPerStreamMax, uint codingBlockSize, uint log2HuffmanDistinctSymbolCountMax)
{
    if(m_log2HuffmanDistinctSymbolCountMax > 24) {
        printf("WARNING: log2HuffmanDistinctSymbolCountMax must be <= 24 (provided: %u)\n", m_log2HuffmanDistinctSymbolCountMax);
        return false;
    }
    //TODO also check for valid offset interval and sane streamCountMax, elemCountPerStreamMax

    if(cudaDevice >= 0) {
        m_cudaDevice = cudaDevice;
        cudaSafeCall(cudaSetDevice(m_cudaDevice));
    } else {
        // value < 0 means "use current device"
        cudaSafeCall(cudaGetDevice(&m_cudaDevice));
    }

    m_streamCountMax = streamCountMax;
    m_elemCountPerStreamMax = elemCountPerStreamMax;

    m_codingBlockSize = codingBlockSize;
    if(m_codingBlockSize == 0) {
        // default to 128
        m_codingBlockSize = 128;
    }

    m_log2HuffmanDistinctSymbolCountMax = log2HuffmanDistinctSymbolCountMax;
    if(m_log2HuffmanDistinctSymbolCountMax == 0) {
        // default to 14 bits (which was used before this was configurable)
        m_log2HuffmanDistinctSymbolCountMax = 14;
    }


    uint offsetCountMax = (m_elemCountPerStreamMax + m_codingBlockSize - 1) / m_codingBlockSize;

    uint rowPitch = (uint)getAlignedSize(m_elemCountPerStreamMax + 1, 128 / sizeof(uint));
    m_pScanPlan = new ScanPlan(sizeof(uint), m_elemCountPerStreamMax + 1, m_streamCountMax, rowPitch); // "+ 1" for total
    m_pReducePlan = new ReducePlan(sizeof(uint), m_elemCountPerStreamMax);


    size_t sizeTier0 = 0;
    sizeTier0 = max(sizeTier0, runLengthGetRequiredMemory(this));
    sizeTier0 = max(sizeTier0, huffmanGetRequiredMemory(this));
    // HuffmanEncodeTable uses histogram...
    sizeTier0 = max(sizeTier0, HuffmanEncodeTable::getRequiredMemory(this) + histogramGetRequiredMemory(this));
    sizeTier0 = max(sizeTier0, packIncGetRequiredMemory(this));
    size_t sizeTier1 = 0;
    sizeTier1 = max(sizeTier1, encodeGetRequiredMemory(this));

    m_bufferSize = sizeTier0 + sizeTier1;
    //TODO don't use cudaSafeCall, but manually check for out of memory!
    cudaSafeCall(cudaMalloc(&m_dpBuffer, m_bufferSize));


    runLengthInit(this);
    huffmanInit(this);
    HuffmanEncodeTable::init(this);
    histogramInit(this);
    packIncInit(this);

    encodeInit(this);

    //m_pProfiler = new Profiler();
    //m_pProfiler->addOutputHandler(new ProfilerLogWriter("profiler.log")); //TODO add cudaDevice to filename if != -1
    //m_pProfiler->start();


    // set default number of omp threads
    omp_set_num_threads(omp_get_num_procs());


    return true;
}

void Instance::destroy()
{
    assert(m_bufferOffset == 0);
    assert(m_allocatedSizes.empty());


    if(m_pProfiler) {
        m_pProfiler->stop();
        m_pProfiler->output();
        delete m_pProfiler;
        m_pProfiler = nullptr;
    }


    cudaSafeCall(cudaFree(m_dpBuffer));
    m_dpBuffer = nullptr;
    m_bufferSize = 0;
    m_bufferOffset = 0;


    encodeShutdown(this);

    packIncShutdown(this);
    runLengthShutdown(this);
    HuffmanEncodeTable::shutdown(this);
    huffmanShutdown(this);
    histogramShutdown(this);


    delete m_pReducePlan;
    m_pReducePlan = nullptr;

    delete m_pScanPlan;
    m_pScanPlan = nullptr;


    m_log2HuffmanDistinctSymbolCountMax = 0;
    m_codingBlockSize = 0;
    m_elemCountPerStreamMax = 0;
    m_streamCountMax = 0;

    m_cudaDevice = -1;
}


void Instance::setTimingDetail(ETimingDetail detail)
{
    m_timingDetail = detail;

    bool enableTier2 = (m_timingDetail >= TIMING_DETAIL_LOW);
    bool enableTier1 = (m_timingDetail >= TIMING_DETAIL_MEDIUM);
    bool enableTier0 = (m_timingDetail >= TIMING_DETAIL_HIGH);

    Encode.timerEncodeLowDetail.setEnabled(enableTier2);
    Encode.timerDecodeLowDetail.setEnabled(enableTier2);

    Encode.timerEncodeHighDetail.setEnabled(enableTier1);
    Encode.timerDecodeHighDetail.setEnabled(enableTier1);

    Huffman.timerEncode.setEnabled(enableTier0);
    Huffman.timerDecode.setEnabled(enableTier0);
    RunLength.timerEncode.setEnabled(enableTier0);
    RunLength.timerDecode.setEnabled(enableTier0);
}

void Instance::getTimings(std::vector<std::string>& names, std::vector<float>& times)
{
    bool enableTier2 = (m_timingDetail >= TIMING_DETAIL_LOW);
    bool enableTier1 = (m_timingDetail >= TIMING_DETAIL_MEDIUM);
    bool enableTier0 = (m_timingDetail >= TIMING_DETAIL_HIGH);

    if(enableTier2)
    {
        Encode.timerEncodeLowDetail.getAccumulatedTimes(names, times, true);
        Encode.timerDecodeLowDetail.getAccumulatedTimes(names, times, true);
    }

    if(enableTier1)
    {
        Encode.timerEncodeHighDetail.getAccumulatedTimes(names, times, true);
        Encode.timerDecodeHighDetail.getAccumulatedTimes(names, times, true);
    }

    if(enableTier0)
    {
        Huffman.timerEncode.getAccumulatedTimes(names, times, true);
        Huffman.timerDecode.getAccumulatedTimes(names, times, true);
        RunLength.timerEncode.getAccumulatedTimes(names, times, true);
        RunLength.timerDecode.getAccumulatedTimes(names, times, true);
    }
}

void Instance::printTimings()
{
    std::vector<std::string> eventNames;
    std::vector<float>       eventTimes;

    getTimings(eventNames, eventTimes);

    size_t eventCount = min(eventNames.size(), eventTimes.size());

    uint nameLengthMax = 0;
    for(uint i = 0; i < eventCount; i++) {
        nameLengthMax = max(nameLengthMax, (uint)eventNames[i].length());
    }
    for(uint i = 0; i < eventCount; i++) {
        printf("%-*s: %7.2f ms\n", nameLengthMax+1, eventNames[i].c_str(), eventTimes[i]);
    }
}

void Instance::resetTimings()
{
    Encode.timerEncodeLowDetail.reset();
    Encode.timerDecodeLowDetail.reset();

    Encode.timerEncodeHighDetail.reset();
    Encode.timerDecodeHighDetail.reset();

    Huffman.timerEncode.reset();
    Huffman.timerDecode.reset();
    RunLength.timerEncode.reset();
    RunLength.timerDecode.reset();
}


byte* Instance::getByteBuffer(size_t bytes)
{
    assert(m_bufferOffset + bytes <= m_bufferSize);
    if(m_bufferOffset + bytes > m_bufferSize) {
        printf("ERROR: Instance::getByteBuffer: out of memory!\n");
        return nullptr;
    }

    byte* dpResult = m_dpBuffer + m_bufferOffset;
    m_allocatedSizes.push_back(bytes);
    m_bufferOffset += getAlignedSize(bytes, 128);

    return dpResult;
}

void Instance::releaseBuffer()
{
    assert(!m_allocatedSizes.empty());
    if(m_allocatedSizes.empty()) {
        printf("ERROR: Instance::releaseBuffer: no more buffers to release\n");
        return;
    }

    size_t lastSize = m_allocatedSizes.back();
    m_allocatedSizes.pop_back();

    m_bufferOffset -= getAlignedSize(lastSize, 128);
    assert(m_bufferOffset % 128 == 0);
}

void Instance::releaseBuffers(uint bufferCount)
{
    for(uint i = 0; i < bufferCount; i++) {
        releaseBuffer();
    }
}

}
