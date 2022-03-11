#include "memtrace.h"
#include <cudaCompress/CompressVolume.h>

#include <algorithm>

#include <cudaCompress/BitStream.h>
#include <cudaCompress/Encode.h>
#include <cudaCompress/EncodeCommon.h>
#include <cudaCompress/util/DWT.h>
#include <cudaCompress/util/Quantize.h>
using namespace cudaCompress;

#include "cudaUtil.h"

namespace cudaCompress
{
static constexpr uint g_blockCountPerChannel = 7;

static constexpr uint g_levelCountMax = 10;

GPUResources::Config::Config()
    : cudaDevice(-1), blockCountMax(0), elemCountPerBlockMax(0), codingBlockSize(0), log2HuffmanDistinctSymbolCountMax(0), bufferSize(0)
{
}

void GPUResources::Config::merge(const GPUResources::Config& other)
{
    if (cudaDevice == -1) {
        cudaDevice = other.cudaDevice;
    }

    if (blockCountMax == 0) {
        blockCountMax = other.blockCountMax;
    }
    else {
        blockCountMax = std::max(blockCountMax, other.blockCountMax);
    }

    if (elemCountPerBlockMax == 0) {
        elemCountPerBlockMax = other.elemCountPerBlockMax;
    }
    else {
        elemCountPerBlockMax = std::max(elemCountPerBlockMax, other.elemCountPerBlockMax);
    }

    if (codingBlockSize == 0) {
        codingBlockSize = other.codingBlockSize;
    }
    else {
        codingBlockSize = std::min(codingBlockSize, other.codingBlockSize);
    }

    if (log2HuffmanDistinctSymbolCountMax == 0) {
        log2HuffmanDistinctSymbolCountMax = other.log2HuffmanDistinctSymbolCountMax;
    }
    else {
        log2HuffmanDistinctSymbolCountMax = std::max(log2HuffmanDistinctSymbolCountMax, other.log2HuffmanDistinctSymbolCountMax);
    }

    if (bufferSize == 0) {
        bufferSize = other.bufferSize;
    }
    else {
        bufferSize = std::max(bufferSize, other.bufferSize);
    }
}


GPUResources::GPUResources()
    : m_pCuCompInstance(nullptr)
    , m_dpBuffer(nullptr)
    , m_bufferOffset(0)
{
}

GPUResources::~GPUResources()
{
    assert(m_pCuCompInstance == nullptr);
    assert(m_dpBuffer == nullptr);
}


bool GPUResources::create(const Config& config)
{
    m_config = config;

    assert(m_pCuCompInstance == nullptr);
    m_pCuCompInstance = cudaCompress::createInstance(m_config.cudaDevice, m_config.blockCountMax, m_config.elemCountPerBlockMax, m_config.codingBlockSize, m_config.log2HuffmanDistinctSymbolCountMax);
    if (!m_pCuCompInstance) {
        return false;
    }

    //TODO don't use cudaSafeCall, but manually check for out of memory?
    assert(m_dpBuffer == nullptr);
    cudaSafeCall(cudaMalloc(&m_dpBuffer, m_config.bufferSize));

    return true;
}

void GPUResources::destroy()
{
    cudaSafeCall(cudaFree(m_dpBuffer));
    m_dpBuffer = nullptr;

    cudaCompress::destroyInstance(m_pCuCompInstance);
    m_pCuCompInstance = nullptr;
}


byte* GPUResources::getByteBuffer(size_t bytes)
{
    assert(m_bufferOffset + bytes <= m_config.bufferSize);
    if (m_bufferOffset + bytes > m_config.bufferSize) {
        printf("ERROR: GPUResources::getByteBuffer: out of memory!\n");
        return nullptr;
    }

    byte* dpResult = m_dpBuffer + m_bufferOffset;
    m_allocatedSizes.push_back(bytes);
    m_bufferOffset += getAlignedSize(bytes, 128);

    return dpResult;
}

void GPUResources::releaseBuffer()
{
    assert(!m_allocatedSizes.empty());
    if (m_allocatedSizes.empty()) {
        printf("ERROR: GPUResources::releaseBuffer: no more buffers to release\n");
        return;
    }

    size_t lastSize = m_allocatedSizes.back();
    m_allocatedSizes.pop_back();

    m_bufferOffset -= getAlignedSize(lastSize, 128);
    assert(m_bufferOffset % 128 == 0);
}

void GPUResources::releaseBuffers(uint bufferCount)
{
    for (uint i = 0; i < bufferCount; i++) {
        releaseBuffer();
    }
}

GPUResources::Config CompressVolumeResources::getRequiredResources(uint sizeX, uint sizeY, uint sizeZ, uint channelCount, uint log2HuffmanDistinctSymbolCountMax)
{
    bool longSymbols = (log2HuffmanDistinctSymbolCountMax > 16);
    uint symbolSize = longSymbols ? sizeof(Symbol32) : sizeof(Symbol16);

    uint blockCount = g_blockCountPerChannel * channelCount;

    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;
    uint blockSizeZ = sizeZ / 2;

    uint elemCount = sizeX * sizeY * sizeZ;
    uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;

    // accumulate GPU buffer size
    size_t size = 0;

    // dpDWT
    size_t sizeDWTFloat = 2 * getAlignedSize(elemCount * sizeof(float));
    size_t sizeDWTInt   = 2 * getAlignedSize(elemCount * sizeof(short)) * channelCount;
    size += std::max(sizeDWTFloat, sizeDWTInt);

    // dpSymbolStreams
    size += getAlignedSize(blockCount * elemCountPerBlock * symbolSize);

    // dppSymbolStreams
    size += getAlignedSize(blockCount * g_levelCountMax * sizeof(Symbol16*));

    // build GPUResources config
    GPUResources::Config config;
    config.blockCountMax = blockCount;
    config.elemCountPerBlockMax = elemCountPerBlock;
    config.log2HuffmanDistinctSymbolCountMax = log2HuffmanDistinctSymbolCountMax;
    config.bufferSize = size;

    return config;
}

bool CompressVolumeResources::create(const GPUResources::Config& config)
{
    size_t uploadSize = config.blockCountMax * g_levelCountMax * sizeof(Symbol32*);
    cudaSafeCall(cudaMallocHost(&pUpload, uploadSize, cudaHostAllocWriteCombined));

    cudaSafeCall(cudaEventCreateWithFlags(&syncEventUpload, cudaEventDisableTiming));
    // immediately record to signal that buffers are ready to use (ie first cudaEventSynchronize works)
    cudaSafeCall(cudaEventRecord(syncEventUpload));

    return true;
}

void CompressVolumeResources::destroy()
{
    if(syncEventUpload) {
        cudaSafeCall(cudaEventDestroy(syncEventUpload));
        syncEventUpload = 0;
    }

    cudaSafeCall(cudaFreeHost(pUpload));
    pUpload = nullptr;
}



bool compressVolumeLosslessOneLevel(GPUResources& shared, CompressVolumeResources& resources, const short* dpImage, uint sizeX, uint sizeY, uint sizeZ, short* dpLowpass, std::vector<uint>& highpassBitStream)
{
    uint blockCount = g_blockCountPerChannel;

    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;
    uint blockSizeZ = sizeZ / 2;

    uint elemCount = sizeX * sizeY * sizeZ;
    uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;


    short* dpDWT0 = shared.getBuffer<short>(elemCount);
    short* dpDWT1 = shared.getBuffer<short>(elemCount);
    std::vector<Symbol16*> dpSymbolStreams(blockCount);
    for(uint i = 0; i < blockCount; i++) {
        dpSymbolStreams[i] = shared.getBuffer<Symbol16>(elemCountPerBlock);
    }

    util::CudaScopedTimer timer(resources.timerEncode);


    // perform DWT
    timer("DWT");
    util::dwtIntForward(dpDWT1, dpDWT0, dpImage, sizeX, sizeY, sizeZ);

    // copy lowpass band
    timer("Copy LP");
    cudaMemcpy3DParms params = { 0 };
    params.srcPtr.ptr = dpDWT1;
    params.srcPtr.pitch = sizeX * sizeof(short);
    params.srcPtr.xsize = sizeX * sizeof(short);
    params.srcPtr.ysize = sizeY;
    params.dstPtr.ptr = dpLowpass;
    params.dstPtr.pitch = blockSizeX * sizeof(short);
    params.dstPtr.xsize = blockSizeX * sizeof(short);
    params.dstPtr.ysize = blockSizeY;
    params.extent.width  = blockSizeX * sizeof(short);
    params.extent.height = blockSizeY;
    params.extent.depth  = blockSizeZ;
    params.kind = cudaMemcpyDeviceToDevice;
    cudaSafeCall(cudaMemcpy3DAsync(&params));

    // symbolize highpass bands
    timer("Symbolize HP");
    uint nextSymbolStream = 0;
    for(uint block = 1; block < 1 + blockCount; block++) {
        uint x = block % 2;
        uint y = block / 2 % 2;
        uint z = block / 4;
        uint offset = x * blockSizeX + y * blockSizeY * sizeX + z * blockSizeZ * sizeX * sizeY;

        // make (unsigned!) symbols
        util::symbolize(dpSymbolStreams[nextSymbolStream++], dpDWT1 + offset, blockSizeX, blockSizeY, blockSizeZ, sizeX, sizeX * sizeY);
    }

    BitStream bitStream(&highpassBitStream);

    // encode
    timer("Encode");
    bitStream.reserveBitSize(bitStream.getBitPosition() + sizeX * sizeY * sizeZ * 16);
    bool result = encodeRLHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCount, elemCountPerBlock);

    timer();

    shared.releaseBuffers(blockCount + 2);

    return result;
}

void decompressVolumeLosslessOneLevel(GPUResources& shared, CompressVolumeResources& resources, short* dpImage, uint sizeX, uint sizeY, uint sizeZ, const short* dpLowpass, const std::vector<uint>& highpassBitStream, uint& bitStreamOffset)
{
    uint blockCount = g_blockCountPerChannel;

    uint blockSizeX = sizeX / 2;
    uint blockSizeY = sizeY / 2;
    uint blockSizeZ = sizeZ / 2;

    uint elemCount = sizeX * sizeY * sizeZ;
    uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;


    short* dpDWT0 = shared.getBuffer<short>(elemCount);
    short* dpDWT1 = shared.getBuffer<short>(elemCount);
    Symbol16* dpSymbolStream = shared.getBuffer<Symbol16>(blockCount * elemCountPerBlock);

    cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, blockCount * elemCountPerBlock * sizeof(Symbol16)));

    std::vector<Symbol16*> dpSymbolStreams(blockCount);
    for(uint i = 0; i < blockCount; i++) {
        dpSymbolStreams[i] = dpSymbolStream + i * elemCountPerBlock;
    }

    util::CudaScopedTimer timer(resources.timerDecode);


    // copy lowpass band into larger buffer
    timer("Copy LP");
    cudaMemcpy3DParms params = { 0 };
    params.srcPtr.ptr = const_cast<short*>(dpLowpass);
    params.srcPtr.pitch = blockSizeX * sizeof(short);
    params.srcPtr.xsize = blockSizeX * sizeof(short);
    params.srcPtr.ysize = blockSizeY;
    params.dstPtr.ptr = dpDWT1;
    params.dstPtr.pitch = sizeX * sizeof(short);
    params.dstPtr.xsize = sizeX * sizeof(short);
    params.dstPtr.ysize = sizeY;
    params.extent.width  = blockSizeX * sizeof(short);
    params.extent.height = blockSizeY;
    params.extent.depth  = blockSizeZ;
    params.kind = cudaMemcpyDeviceToDevice;
    cudaSafeCall(cudaMemcpy3DAsync(&params));

    BitStreamReadOnly bitStream(highpassBitStream.data(), uint(highpassBitStream.size() * sizeof(uint) * 8));
    bitStream.setBitPosition(bitStreamOffset * 32);

    // decode
    timer("Decode");
    decodeRLHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCount, elemCountPerBlock);

    bitStream.align<uint>();
    bitStreamOffset = bitStream.getBitPosition() / 32;

    // unsymbolize highpass bands
    timer("Unsymbolize HP");
    uint nextSymbolStream = 0;
    for(uint block = 1; block < 1 + blockCount; block++) {
        uint x = block % 2;
        uint y = block / 2 % 2;
        uint z = block / 4;
        uint offset = x * blockSizeX + y * blockSizeY * sizeX + z * blockSizeZ * sizeX * sizeY;

        // get signed values back from unsigned symbols
        util::unsymbolize(dpDWT1 + offset, dpSymbolStreams[nextSymbolStream++], blockSizeX, blockSizeY, blockSizeZ, sizeX, sizeX * sizeY);
    }

    // perform IDWT
    timer("IDWT");
    util::dwtIntInverse(dpImage, dpDWT0, dpDWT1, sizeX, sizeY, sizeZ);

    shared.releaseBuffers(3);
}

void decompressVolumeLosslessOneLevel(GPUResources& shared, CompressVolumeResources& resources, short* dpImage, uint sizeX, uint sizeY, uint sizeZ, const short* dpLowpass, const std::vector<uint>& highpassBitStream)
{
    uint bitStreamOffset = 0;
    decompressVolumeLosslessOneLevel(shared, resources, dpImage, sizeX, sizeY, sizeZ, dpLowpass, highpassBitStream, bitStreamOffset);
}

bool compressVolumeLossless(GPUResources& shared, CompressVolumeResources& resources, const short* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, std::vector<uint>& bits)
{
    bits.clear(); //FIXME?

    uint elemCount = sizeX * sizeY * sizeZ;
    uint sizeBytes = elemCount * sizeof(short);

    short* dpBuffer0 = shared.getBuffer<short>(elemCount);
    short* dpBuffer1 = shared.getBuffer<short>(elemCount / 8);

    std::vector<std::vector<uint>> levelBitStreams(numLevels);

    uint factor = 1;
    for(uint level = 0; level < numLevels; level++) {
        const short* dpSource = (level == 0 ? dpImage : dpBuffer0);
        if(!compressVolumeLosslessOneLevel(shared, resources, dpSource, sizeX / factor, sizeY / factor, sizeZ / factor, dpBuffer1, levelBitStreams[level])) {
            shared.releaseBuffers(2);
            return false;
        }
        factor *= 2;
        // copy lowpass to buffer0
        cudaMemcpy(dpBuffer0, dpBuffer1, sizeBytes / (factor*factor*factor), cudaMemcpyDeviceToDevice);
    }

    util::symbolize((ushort*)dpBuffer1, dpBuffer0, sizeX / factor, sizeY / factor, sizeZ / factor, sizeX / factor, sizeX * sizeY / (factor*factor));

    BitStream bitStream(&bits);
    bool result = encodeHuff(shared.m_pCuCompInstance, bitStream, (Symbol16**)&dpBuffer1, 1, sizeX * sizeY * sizeZ / (factor * factor * factor));
    if(!result) {
        shared.releaseBuffers(2);
        return false;
    }

    // append level bit streams
    for(int level = numLevels - 1; level >= 0; level--) {
        size_t oldSize = bits.size();
        bits.resize(bits.size() + levelBitStreams[level].size());
        memcpy(bits.data() + oldSize, levelBitStreams[level].data(), levelBitStreams[level].size() * sizeof(uint));
    }

    shared.releaseBuffers(2);

    return true;
}

void decompressVolumeLossless(GPUResources& shared, CompressVolumeResources& resources, short* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const std::vector<uint>& bits)
{
    uint elemCount = sizeX * sizeY * sizeZ;
    uint sizeBytes = elemCount * sizeof(short);

    short* dpBuffer0 = shared.getBuffer<short>(elemCount);
    short* dpBuffer1 = shared.getBuffer<short>(elemCount / 8);

    uint factor = (1 << numLevels);

    // huffman decode lowpass band
    BitStreamReadOnly bitStream(bits.data(), uint(bits.size() * sizeof(uint) * 8));
    decodeHuff(shared.m_pCuCompInstance, bitStream, (Symbol16**)&dpBuffer1, 1, sizeX * sizeY * sizeZ / (factor * factor * factor));

    // convert (unsigned) symbols to signed values for IDWT
    util::unsymbolize(dpBuffer0, (ushort*)dpBuffer1, sizeX / factor, sizeY / factor, sizeZ / factor, sizeX / factor, sizeX * sizeY / (factor * factor));

    bitStream.align<uint>();
    uint bitStreamOffset = bitStream.getBitPosition() / 32;

    for(int level = numLevels - 1; level >= 0; level--) {
        // copy lowpass from previous level to buffer1
        cudaMemcpy(dpBuffer1, dpBuffer0, sizeBytes / (factor*factor*factor), cudaMemcpyDeviceToDevice);
        factor /= 2;
        // decode highpass bands and perform IDWT
        short* dpTarget = (level == 0 ? dpImage : dpBuffer0);
        decompressVolumeLosslessOneLevel(shared, resources, dpTarget, sizeX / factor, sizeY / factor, sizeZ / factor, dpBuffer1, bits, bitStreamOffset);
    }

    shared.releaseBuffers(2);
}


template<typename Symbol>
bool compressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, const float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, std::vector<uint>& bits, float quantizationStepLevel0, bool doRLEOnlyOnLvl0)
{
    uint blockCount = g_blockCountPerChannel;

    uint blockSizeMaxX = sizeX / 2;
    uint blockSizeMaxY = sizeY / 2;
    uint blockSizeMaxZ = sizeZ / 2;

    uint elemCount = sizeX * sizeY * sizeZ;
    uint elemCountPerBlockMax = blockSizeMaxX * blockSizeMaxY * blockSizeMaxZ;


    float* dpDWT[2];
    dpDWT[0] = shared.getBuffer<float>(elemCount);
    dpDWT[1] = shared.getBuffer<float>(elemCount);
    std::vector<Symbol*> dpSymbolStreams(blockCount);
    for(uint i = 0; i < blockCount; i++) {
        dpSymbolStreams[i] = shared.getBuffer<Symbol>(elemCountPerBlockMax);
    }

    util::CudaScopedTimer timer(resources.timerEncode);


    bits.clear(); //FIXME?

    // perform full dwt
    timer("DWT");
    int d = 0;
    util::dwtFloat3DForward(dpDWT[d], dpDWT[1-d], dpDWT[d], dpImage, sizeX, sizeY, sizeZ);
    uint factor = 2;
    for(uint level = 1; level < numLevels; level++) {
        util::dwtFloat3DForward(dpDWT[1-d], dpDWT[d], dpDWT[1-d], dpDWT[d], sizeX / factor, sizeY / factor, sizeZ / factor, 1, sizeX, sizeX * sizeY, sizeX, sizeX * sizeY);
        d = 1 - d;
        factor *= 2;
    }
    // note: the correct coefficients are now distributed over both DWT buffers:
    //       the coarsest level is in d, the next finest in 1-d, then d again etc.

    BitStream bitStream(&bits);

    // quantize and encode lowpass band
    timer("Quantize LP");
    float quantizationStep = quantizationStepLevel0 / float(factor/2);
    util::quantizeToSymbols(dpSymbolStreams[0], dpDWT[d], sizeX / factor, sizeY / factor, sizeZ / factor, quantizationStep, sizeX, sizeX * sizeY, util::QUANTIZE_DEADZONE);
    timer("Encode LP");
    bool result = encodeHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), 1, elemCount / (factor * factor * factor));
    if(!result) {
        shared.releaseBuffers(blockCount + 2);
        return false;
    }

    // highpass bands
    for(int level = numLevels - 1; level >= 0; level--) {
        quantizationStep = quantizationStepLevel0 / float(factor/2);

        uint blockSizeX = sizeX / factor;
        uint blockSizeY = sizeY / factor;
        uint blockSizeZ = sizeZ / factor;

        // symbolize highpass bands
        timer("Symbolize HP");
        uint nextSymbolStream = 0;
        for(uint block = 1; block < 1 + blockCount; block++) {
            uint x = block % 2;
            uint y = block / 2 % 2;
            uint z = block / 4;
            uint offset = x * blockSizeX + y * blockSizeY * sizeX + z * blockSizeZ * sizeX * sizeY;

            // make (unsigned!) symbols
            util::quantizeToSymbols(dpSymbolStreams[nextSymbolStream++], dpDWT[d] + offset, blockSizeX, blockSizeY, blockSizeZ, quantizationStep, sizeX, sizeX * sizeY, util::QUANTIZE_DEADZONE);
        }
        // switch DWT buffer for next level
        d = 1 - d;

        // encode
        timer("Encode HP");
        uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;
        bool result = false;
        if(level > 0 && doRLEOnlyOnLvl0) {
            result = encodeHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCount, elemCountPerBlock);
        } else {
            result = encodeRLHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCount, elemCountPerBlock);
        }
        if(!result) {
            shared.releaseBuffers(blockCount + 2);
            return false;
        }

        factor /= 2;
    }

    timer();

    shared.releaseBuffers(blockCount + 2);

    return true;
}

template<typename Symbol>
void decompressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const uint* pBits, uint bitCount, float quantizationStepLevel0, bool doRLEOnlyOnLvl0)
{
    uint blockCount = g_blockCountPerChannel;

    uint blockSizeMaxX = sizeX / 2;
    uint blockSizeMaxY = sizeY / 2;
    uint blockSizeMaxZ = sizeZ / 2;

    uint elemCount = sizeX * sizeY * sizeZ;
    uint elemCountPerBlockMax = blockSizeMaxX * blockSizeMaxY * blockSizeMaxZ;
    uint pad = WARP_SIZE * cudaCompress::getInstanceCodingBlockSize(shared.m_pCuCompInstance);
    uint elemCountPerBlockMaxPadded = (elemCountPerBlockMax + pad - 1) / pad * pad;


    float* dpDWT[2];
    dpDWT[0] = shared.getBuffer<float>(elemCount);
    dpDWT[1] = shared.getBuffer<float>(elemCount);
    Symbol* dpSymbolStream = shared.getBuffer<Symbol>(blockCount * elemCountPerBlockMaxPadded);
    Symbol** dppSymbolStreams = shared.getBuffer<Symbol*>(blockCount * numLevels);

    // build array of symbol stream pointers for each level
    std::vector<Symbol*> dpSymbolStreams(blockCount * numLevels);
    {
        int factor = (1 << numLevels);
        uint next = 0;
        for(int level = numLevels - 1; level >= 0; level--) {
            uint blockSizeX = sizeX / factor;
            uint blockSizeY = sizeY / factor;
            uint blockSizeZ = sizeZ / factor;
            uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;
            uint elemCountPerBlockPadded = (elemCountPerBlock + pad - 1) / pad * pad;

            for(uint i = 0; i < blockCount; i++) {
                dpSymbolStreams[next++] = dpSymbolStream + i * elemCountPerBlockPadded;
            }

            factor /= 2;
        }
    }
    // upload per-level symbol stream pointers
    cudaSafeCall(cudaEventSynchronize(resources.syncEventUpload));
    memcpy(resources.pUpload, dpSymbolStreams.data(), dpSymbolStreams.size() * sizeof(Symbol*));
    cudaSafeCall(cudaMemcpyAsync(dppSymbolStreams, resources.pUpload, dpSymbolStreams.size() * sizeof(Symbol*), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaEventRecord(resources.syncEventUpload));


    util::CudaScopedTimer timer(resources.timerDecode);


    BitStreamReadOnly bitStream(pBits, bitCount);

    int factor = (1 << numLevels);

    // lowpass band
    //TODO: if doRLEOnlyOnLvl0, merge LP with coarsest-level HP
    timer("Decode LP");
    uint elemCountLP = elemCount / (factor * factor * factor);
    cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, elemCountLP * sizeof(Symbol)));
    decodeHuff(shared.m_pCuCompInstance, bitStream, &dpSymbolStream, 1, elemCountLP);

    timer("Unquantize LP");
    float quantizationStep = quantizationStepLevel0 / float(factor/2);
    util::unquantizeFromSymbols(dpImage, dpSymbolStream, sizeX / factor, sizeY / factor, sizeZ / factor, quantizationStep, sizeX, sizeX * sizeY);

    // highpass bands
    Symbol** pdpSymbolStreamsNext = dpSymbolStreams.data();
    Symbol** dppSymbolStreamsNext = dppSymbolStreams;
    for(int level = numLevels - 1; level >= 0; level--) {
        uint blockSizeX = sizeX / factor;
        uint blockSizeY = sizeY / factor;
        uint blockSizeZ = sizeZ / factor;
        uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;

        timer("Decode HP");

        // clear output array
        uint elemCountPerBlockPadded = (elemCountPerBlock + pad - 1) / pad * pad;
        cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, blockCount * elemCountPerBlockPadded * sizeof(Symbol)));

        // decode
        if(level > 0 && doRLEOnlyOnLvl0) {
            decodeHuff(shared.m_pCuCompInstance, bitStream, pdpSymbolStreamsNext, blockCount, elemCountPerBlock);
        } else {
            decodeRLHuff(shared.m_pCuCompInstance, bitStream, pdpSymbolStreamsNext, blockCount, elemCountPerBlock);
        }

        factor /= 2;

        // unsymbolize and idwt
        timer("IDWT");
        float quantizationStep = quantizationStepLevel0 / float(factor);
        util::dwtFloat3DInverseFromSymbols(
            dpImage, dpDWT[1], dpDWT[0], dpImage,
            dppSymbolStreamsNext, quantizationStep,
            sizeX / factor, sizeY / factor, sizeZ / factor,
            1, sizeX, sizeX * sizeY, sizeX, sizeX * sizeY);

        pdpSymbolStreamsNext += blockCount;
        dppSymbolStreamsNext += blockCount;
    }

    timer();

    shared.releaseBuffers(4);
}

template<typename Symbol>
void decompressVolumeFloatMultiChannel(GPUResources& shared, CompressVolumeResources& resources, const VolumeChannel* pChannels, uint channelCount, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, bool doRLEOnlyOnLvl0)
{
    uint blockCountPerChannel = g_blockCountPerChannel;
    uint blockCountTotal = blockCountPerChannel * channelCount;

    uint blockSizeMaxX = sizeX / 2;
    uint blockSizeMaxY = sizeY / 2;
    uint blockSizeMaxZ = sizeZ / 2;

    uint elemCount = sizeX * sizeY * sizeZ;
    uint elemCountPerBlockMax = blockSizeMaxX * blockSizeMaxY * blockSizeMaxZ;
    uint pad = WARP_SIZE * cudaCompress::getInstanceCodingBlockSize(shared.m_pCuCompInstance);
    uint elemCountPerBlockMaxPadded = (elemCountPerBlockMax + pad - 1) / pad * pad;


    std::vector<float*> dpDWT(2);
    for(size_t i = 0; i < dpDWT.size(); i++) {
        dpDWT[i] = shared.getBuffer<float>(elemCount);
    }
    Symbol* dpSymbolStream = shared.getBuffer<Symbol>(blockCountTotal * elemCountPerBlockMaxPadded);
    Symbol** dppSymbolStreams = shared.getBuffer<Symbol*>(blockCountTotal * numLevels);

    // build array of symbol stream pointers for each level
    std::vector<Symbol*> dpSymbolStreams(blockCountTotal * numLevels);
    {
        int factor = (1 << numLevels);
        uint next = 0;
        for(int level = numLevels - 1; level >= 0; level--) {
            uint blockSizeX = sizeX / factor;
            uint blockSizeY = sizeY / factor;
            uint blockSizeZ = sizeZ / factor;
            uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;
            uint elemCountPerBlockPadded = (elemCountPerBlock + pad - 1) / pad * pad;

            for(uint i = 0; i < blockCountTotal; i++) {
                dpSymbolStreams[next++] = dpSymbolStream + i * elemCountPerBlockPadded;
            }

            factor /= 2;
        }
    }
    // upload per-level symbol stream pointers
    cudaSafeCall(cudaEventSynchronize(resources.syncEventUpload));
    memcpy(resources.pUpload, dpSymbolStreams.data(), dpSymbolStreams.size() * sizeof(Symbol*));
    cudaSafeCall(cudaMemcpyAsync(dppSymbolStreams, resources.pUpload, dpSymbolStreams.size() * sizeof(Symbol*), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaEventRecord(resources.syncEventUpload));


    util::CudaScopedTimer timer(resources.timerDecode);


    std::vector<BitStreamReadOnly> bitStreams;
    for(uint c = 0; c < channelCount; c++) {
        const VolumeChannel& channel = pChannels[c];
        bitStreams.push_back(BitStreamReadOnly(channel.pBits, channel.bitCount));
    }

    std::vector<BitStreamReadOnly*> pChannelBitStreams;
    for(uint c = 0; c < channelCount; c++) {
        pChannelBitStreams.push_back(&bitStreams[c]);
    }

    std::vector<BitStreamReadOnly*> pBlockBitStreams;
    for(uint c = 0; c < channelCount; c++) {
        for(uint block = 0; block < blockCountPerChannel; block++) {
            pBlockBitStreams.push_back(&bitStreams[c]);
        }
    }

    int factor = (1 << numLevels);

    // lowpass band
    timer("Decode LP");
    uint elemCountLP = elemCount / (factor * factor * factor);
    cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, channelCount * elemCountLP * sizeof(Symbol)));
    decodeHuff(shared.m_pCuCompInstance, pChannelBitStreams.data(), dpSymbolStreams.data(), channelCount, elemCountLP);

    timer("Unquantize LP");
    for(uint c = 0; c < channelCount; c++) {
        const VolumeChannel& channel = pChannels[c];
        float quantizationStep = channel.quantizationStepLevel0 / float(factor/2);
        util::unquantizeFromSymbols(channel.dpImage, dpSymbolStreams[c], sizeX / factor, sizeY / factor, sizeZ / factor, quantizationStep, sizeX, sizeX * sizeY);
    }

    // highpass bands
    Symbol** pdpSymbolStreamsNext = dpSymbolStreams.data();
    Symbol** dppSymbolStreamsNext = dppSymbolStreams;
    for(int level = numLevels - 1; level >= 0; level--) {
        uint blockSizeX = sizeX / factor;
        uint blockSizeY = sizeY / factor;
        uint blockSizeZ = sizeZ / factor;
        uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;

        timer("Decode HP");

        // clear output array
        uint elemCountPerBlockPadded = (elemCountPerBlock + pad - 1) / pad * pad;
        cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, blockCountTotal * elemCountPerBlockPadded * sizeof(Symbol)));

        // decode
        if(level > 0 && doRLEOnlyOnLvl0) {
            decodeHuff(shared.m_pCuCompInstance, pBlockBitStreams.data(), pdpSymbolStreamsNext, blockCountTotal, elemCountPerBlock);
        } else {
            decodeRLHuff(shared.m_pCuCompInstance, pBlockBitStreams.data(), pdpSymbolStreamsNext, blockCountTotal, elemCountPerBlock);
        }

        factor /= 2;

        // unsymbolize and idwt
        timer("IDWT");
        for(uint c = 0; c < channelCount; c++) {
            const VolumeChannel& channel = pChannels[c];

            float quantizationStep = channel.quantizationStepLevel0 / float(factor);

            util::dwtFloat3DInverseFromSymbols(
                channel.dpImage, dpDWT[1], dpDWT[0], channel.dpImage,
                dppSymbolStreamsNext + c * blockCountPerChannel, quantizationStep,
                sizeX / factor, sizeY / factor, sizeZ / factor,
                1, sizeX, sizeX * sizeY, sizeX, sizeX * sizeY);
        }

        pdpSymbolStreamsNext += blockCountTotal;
        dppSymbolStreamsNext += blockCountTotal;
    }

    timer();

    shared.releaseBuffers(uint(dpDWT.size()) + 2);
}


bool compressVolumeFloatQuantFirst(GPUResources& shared, CompressVolumeResources& resources, const float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, std::vector<uint>& bits, float quantizationStep, bool doRLEOnlyOnLvl0)
{
    uint blockCount = g_blockCountPerChannel;

    uint blockSizeMaxX = sizeX / 2;
    uint blockSizeMaxY = sizeY / 2;
    uint blockSizeMaxZ = sizeZ / 2;

    uint elemCount = sizeX * sizeY * sizeZ;
    uint elemCountPerBlockMax = blockSizeMaxX * blockSizeMaxY * blockSizeMaxZ;


    assert(shared.getConfig().log2HuffmanDistinctSymbolCountMax <= 16);


    std::vector<short*> dpDWT(2);
    for(size_t i = 0; i < dpDWT.size(); i++) {
        dpDWT[i] = shared.getBuffer<short>(elemCount);
    }
    std::vector<Symbol16*> dpSymbolStreams(blockCount);
    for(uint i = 0; i < dpSymbolStreams.size(); i++) {
        dpSymbolStreams[i] = shared.getBuffer<Symbol16>(elemCountPerBlockMax);
    }


    bits.clear(); //FIXME?


    util::CudaScopedTimer timer(resources.timerEncode);


    BitStream bitStream(&bits);


    // quantize image
    timer("Quantize");
    util::quantizeToShort(dpDWT[0], dpImage, sizeX, sizeY, sizeZ, quantizationStep, 0, 0, util::QUANTIZE_UNIFORM);


    // perform full dwt
    timer("DWT");
    int d = 0;
    uint factor = 1;
    for(uint level = 0; level < numLevels; level++) {
        // data path is src->dst->buf->dst, so src and buf can be the same buffer
        util::dwtIntForward(dpDWT[1-d], dpDWT[d], dpDWT[d], sizeX / factor, sizeY / factor, sizeZ / factor, sizeX, sizeX * sizeY, sizeX, sizeX * sizeY);
        d = 1 - d;
        factor *= 2;
    }
    // note: the correct coefficients are now distributed over both DWT buffers:
    //       the coarsest level is in d, the next finest in 1-d, then d again etc.


    // symbolize and encode lowpass band
    timer("Symbolize LP");
    util::symbolize(dpSymbolStreams[0], dpDWT[d], sizeX / factor, sizeY / factor, sizeZ / factor, sizeX, sizeX * sizeY);
    timer("Encode LP");
    bool result = encodeHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), 1, elemCount / (factor * factor * factor));
    if(!result) {
        shared.releaseBuffers(uint(dpDWT.size() + dpSymbolStreams.size()));
        return false;
    }

    // highpass bands
    for(int level = numLevels - 1; level >= 0; level--) {
        uint blockSizeX = sizeX / factor;
        uint blockSizeY = sizeY / factor;
        uint blockSizeZ = sizeZ / factor;

        // symbolize highpass bands
        timer("Symbolize HP");
        uint nextSymbolStream = 0;
        for(uint block = 1; block < 1 + blockCount; block++) {
            uint x = block % 2;
            uint y = block / 2 % 2;
            uint z = block / 4;
            uint offset = x * blockSizeX + y * blockSizeY * sizeX + z * blockSizeZ * sizeX * sizeY;

            // make (unsigned!) symbols
            util::symbolize(dpSymbolStreams[nextSymbolStream++], dpDWT[d] + offset, blockSizeX, blockSizeY, blockSizeZ, sizeX, sizeX * sizeY);
        }
        // switch DWT buffer for next level
        d = 1 - d;

        // encode
        timer("Encode HP");
        uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;
        bool result = false;
        if(level > 0 && doRLEOnlyOnLvl0) {
            result = encodeHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCount, elemCountPerBlock);
        } else {
            result = encodeRLHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCount, elemCountPerBlock);
        }
        if(!result) {
            shared.releaseBuffers(uint(dpDWT.size() + dpSymbolStreams.size()));
            return false;
        }

        factor /= 2;
    }

    timer();

    shared.releaseBuffers(uint(dpDWT.size() + dpSymbolStreams.size()));

    return true;
}

bool decompressVolumeFloatQuantFirst(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const uint* pBits, uint bitCount, float quantizationStep, bool doRLEOnlyOnLvl0)
{
    uint blockCount = g_blockCountPerChannel;

    uint blockSizeMaxX = sizeX / 2;
    uint blockSizeMaxY = sizeY / 2;
    uint blockSizeMaxZ = sizeZ / 2;

    uint elemCount = sizeX * sizeY * sizeZ;
    uint elemCountPerBlockMax = blockSizeMaxX * blockSizeMaxY * blockSizeMaxZ;
    uint pad = WARP_SIZE * cudaCompress::getInstanceCodingBlockSize(shared.m_pCuCompInstance);
    uint elemCountPerBlockMaxPadded = (elemCountPerBlockMax + pad - 1) / pad * pad;


    assert(shared.getConfig().log2HuffmanDistinctSymbolCountMax <= 16);

    std::vector<short*> dpDWT(2);
    for(size_t i = 0; i < dpDWT.size(); i++) {
        dpDWT[i] = shared.getBuffer<short>(elemCount);
    }
    Symbol16* dpSymbolStream = shared.getBuffer<Symbol16>(blockCount * elemCountPerBlockMaxPadded);

    std::vector<Symbol16*> dpSymbolStreams(blockCount);
    for(uint i = 0; i < blockCount; i++) {
        dpSymbolStreams[i] = dpSymbolStream + i * elemCountPerBlockMaxPadded;
    }

    util::CudaScopedTimer timer(resources.timerDecode);


    BitStreamReadOnly bitStream(pBits, bitCount);

    int d = 0;
    int factor = (1 << numLevels);

    // lowpass band
    timer("Decode LP");
    uint elemCountLP = elemCount / (factor * factor * factor);
    cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, elemCountPerBlockMaxPadded * sizeof(Symbol16)));
    decodeHuff(shared.m_pCuCompInstance, bitStream, &dpSymbolStream, 1, elemCountLP);

    timer("Unsymbolize LP");
    util::unsymbolize(dpDWT[d], dpSymbolStreams[0], sizeX / factor, sizeY / factor, sizeZ / factor, sizeX, sizeX * sizeY);

    // highpass bands
    for(int level = numLevels - 1; level >= 0; level--) {
        //TODO make per-level symbol stream pointers, and memset only elemCountPerBlock instead of elemCountPerBlockMax
        cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, blockCount * elemCountPerBlockMaxPadded * sizeof(Symbol16)));

        uint blockSizeX = sizeX / factor;
        uint blockSizeY = sizeY / factor;
        uint blockSizeZ = sizeZ / factor;

        // decode
        timer("Decode HP");
        uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;
        if(level > 0 && doRLEOnlyOnLvl0) {
            decodeHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCount, elemCountPerBlock);
        } else {
            decodeRLHuff(shared.m_pCuCompInstance, bitStream, dpSymbolStreams.data(), blockCount, elemCountPerBlock);
        }

        factor /= 2;

        // unsymbolize
        timer("Unsymbolize HP");
        uint nextSymbolStream = 0;
        for(uint block = 1; block < 1 + blockCount; block++) {
            uint x = block % 2;
            uint y = block / 2 % 2;
            uint z = block / 4;
            uint offset = x * blockSizeX + y * blockSizeY * sizeX + z * blockSizeZ * sizeX * sizeY;

            // make (unsigned!) symbols
            util::unsymbolize(dpDWT[d] + offset, dpSymbolStreams[nextSymbolStream++], blockSizeX, blockSizeY, blockSizeZ, sizeX, sizeX * sizeY);
        }
        // switch DWT buffer for next level
        d = 1 - d;
    }

    timer("IDWT");
    for(int level = numLevels - 1; level >= 0; level--) {
        factor = (1 << level);
        // data path is src->dst->buf->dst, so src and buf can be the same buffer
        util::dwtIntInverse(dpDWT[1-d], dpDWT[d], dpDWT[d], sizeX / factor, sizeY / factor, sizeZ / factor, sizeX, sizeX * sizeY, sizeX, sizeX * sizeY);
        d = 1 - d;
    }

    // unquantize image
    timer("Unquantize");
    util::unquantizeFromShort(dpImage, dpDWT[d], sizeX, sizeY, sizeZ, quantizationStep, 0, 0, util::QUANTIZE_UNIFORM);

    timer();

    shared.releaseBuffers(uint(dpDWT.size()) + 1);

    return true;
}

bool decompressVolumeFloatQuantFirst(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const std::vector<uint>& bits, float quantizationStep, bool doRLEOnlyOnLvl0)
{
    return decompressVolumeFloatQuantFirst(shared, resources, dpImage, sizeX, sizeY, sizeZ, numLevels, bits.data(), uint(bits.size() * sizeof(uint) * 8), quantizationStep, doRLEOnlyOnLvl0);
}

bool decompressVolumeFloatQuantFirstMultiChannel(GPUResources& shared, CompressVolumeResources& resources, const VolumeChannel* pChannels, uint channelCount, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, bool doRLEOnlyOnLvl0)
{
    uint blockCountPerChannel = g_blockCountPerChannel;
    uint blockCountTotal = blockCountPerChannel * channelCount;

    uint blockSizeMaxX = sizeX / 2;
    uint blockSizeMaxY = sizeY / 2;
    uint blockSizeMaxZ = sizeZ / 2;

    uint elemCount = sizeX * sizeY * sizeZ;
    uint elemCountPerBlockMax = blockSizeMaxX * blockSizeMaxY * blockSizeMaxZ;
    uint pad = WARP_SIZE * cudaCompress::getInstanceCodingBlockSize(shared.m_pCuCompInstance);
    uint elemCountPerBlockMaxPadded = (elemCountPerBlockMax + pad - 1) / pad * pad;


    assert(shared.getConfig().log2HuffmanDistinctSymbolCountMax <= 16);

    std::vector<short*> dpDWT(2 * channelCount);
    for(size_t i = 0; i < dpDWT.size(); i++) {
        dpDWT[i] = shared.getBuffer<short>(elemCount);
    }
    Symbol16* dpSymbolStream = shared.getBuffer<Symbol16>(blockCountTotal * elemCountPerBlockMaxPadded);

    std::vector<Symbol16*> dpSymbolStreams(blockCountTotal);
    for(uint i = 0; i < blockCountTotal; i++) {
        dpSymbolStreams[i] = dpSymbolStream + i * elemCountPerBlockMaxPadded;
    }

    util::CudaScopedTimer timer(resources.timerDecode);


    std::vector<BitStreamReadOnly> bitStreams;
    for(uint c = 0; c < channelCount; c++) {
        const VolumeChannel& channel = pChannels[c];
        bitStreams.push_back(BitStreamReadOnly(channel.pBits, channel.bitCount));
    }

    std::vector<BitStreamReadOnly*> pChannelBitStreams;
    for(uint c = 0; c < channelCount; c++) {
        pChannelBitStreams.push_back(&bitStreams[c]);
    }

    std::vector<BitStreamReadOnly*> pBlockBitStreams;
    for(uint c = 0; c < channelCount; c++) {
        for(uint block = 0; block < blockCountPerChannel; block++) {
            pBlockBitStreams.push_back(&bitStreams[c]);
        }
    }

    int d = 0;
    int factor = (1 << numLevels);

    // lowpass band
    timer("Decode LP");
    uint elemCountLP = elemCount / (factor * factor * factor);
    cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, channelCount * elemCountPerBlockMaxPadded * sizeof(Symbol16)));
    decodeHuff(shared.m_pCuCompInstance, pChannelBitStreams.data(), dpSymbolStreams.data(), channelCount, elemCountLP);

    timer("Unsymbolize LP");
    for(uint c = 0; c < channelCount; c++) {
        util::unsymbolize(dpDWT[2*c + d], dpSymbolStreams[c], sizeX / factor, sizeY / factor, sizeZ / factor, sizeX, sizeX * sizeY);
    }

    // highpass bands
    for(int level = numLevels - 1; level >= 0; level--) {
        //TODO make per-level symbol stream pointers, and memset only elemCountPerBlock instead of elemCountPerBlockMax
        cudaSafeCall(cudaMemsetAsync(dpSymbolStream, 0, blockCountTotal * elemCountPerBlockMaxPadded * sizeof(Symbol16)));

        uint blockSizeX = sizeX / factor;
        uint blockSizeY = sizeY / factor;
        uint blockSizeZ = sizeZ / factor;

        // decode
        timer("Decode HP");
        uint elemCountPerBlock = blockSizeX * blockSizeY * blockSizeZ;
        if(level > 0 && doRLEOnlyOnLvl0) {
            decodeHuff(shared.m_pCuCompInstance, pBlockBitStreams.data(), dpSymbolStreams.data(), blockCountTotal, elemCountPerBlock);
        } else {
            decodeRLHuff(shared.m_pCuCompInstance, pBlockBitStreams.data(), dpSymbolStreams.data(), blockCountTotal, elemCountPerBlock);
        }

        factor /= 2;

        // unsymbolize
        timer("Unsymbolize HP");
        uint nextSymbolStream = 0;
        for(uint c = 0; c < channelCount; c++) {
            for(uint block = 1; block < 1 + blockCountPerChannel; block++) {
                uint x = block % 2;
                uint y = block / 2 % 2;
                uint z = block / 4;
                uint offset = x * blockSizeX + y * blockSizeY * sizeX + z * blockSizeZ * sizeX * sizeY;

                // make (unsigned!) symbols
                util::unsymbolize(dpDWT[2*c + d] + offset, dpSymbolStreams[nextSymbolStream++], blockSizeX, blockSizeY, blockSizeZ, sizeX, sizeX * sizeY);
            }
        }
        // switch DWT buffer for next level
        d = 1 - d;
    }

    timer("IDWT");
    for(uint c = 0; c < channelCount; c++) {
        for(int level = numLevels - 1; level >= 0; level--) {
            factor = (1 << level);
            // data path is src->dst->buf->dst, so src and buf can be the same buffer
            util::dwtIntInverse(dpDWT[2*c + 1-d], dpDWT[2*c + d], dpDWT[2*c + d], sizeX / factor, sizeY / factor, sizeZ / factor, sizeX, sizeX * sizeY, sizeX, sizeX * sizeY);
            d = 1 - d;
        }
    }

    // unquantize image
    timer("Unquantize");
    for(uint c = 0; c < channelCount; c++) {
        util::unquantizeFromShort(pChannels[c].dpImage, dpDWT[2*c + d], sizeX, sizeY, sizeZ, pChannels[c].quantizationStepLevel0, 0, 0, util::QUANTIZE_UNIFORM);
    }

    timer();

    shared.releaseBuffers(uint(dpDWT.size()) + 1);

    return true;
}



bool compressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, const float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, std::vector<uint>& bits, float quantizationStepLevel0, bool doRLEOnlyOnLvl0)
{
    bool longSymbols = (shared.getConfig().log2HuffmanDistinctSymbolCountMax > 16);
    if(longSymbols) {
        return compressVolumeFloat<Symbol32>(shared, resources, dpImage, sizeX, sizeY, sizeZ, numLevels, bits, quantizationStepLevel0, doRLEOnlyOnLvl0);
    } else {
        return compressVolumeFloat<Symbol16>(shared, resources, dpImage, sizeX, sizeY, sizeZ, numLevels, bits, quantizationStepLevel0, doRLEOnlyOnLvl0);
    }
}

void decompressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const std::vector<uint>& bits, float quantizationStepLevel0, bool doRLEOnlyOnLvl0)
{
    decompressVolumeFloat(shared, resources, dpImage, sizeX, sizeY, sizeZ, numLevels, bits.data(), uint(bits.size() * sizeof(uint) * 8), quantizationStepLevel0, doRLEOnlyOnLvl0);
}

void decompressVolumeFloat(GPUResources& shared, CompressVolumeResources& resources, float* dpImage, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, const uint* pBits, uint bitCount, float quantizationStepLevel0, bool doRLEOnlyOnLvl0)
{
    bool longSymbols = (shared.getConfig().log2HuffmanDistinctSymbolCountMax > 16);
    if(longSymbols) {
        decompressVolumeFloat<Symbol32>(shared, resources, dpImage, sizeX, sizeY, sizeZ, numLevels, pBits, bitCount, quantizationStepLevel0, doRLEOnlyOnLvl0);
    } else {
        decompressVolumeFloat<Symbol16>(shared, resources, dpImage, sizeX, sizeY, sizeZ, numLevels, pBits, bitCount, quantizationStepLevel0, doRLEOnlyOnLvl0);
    }
}

void decompressVolumeFloatMultiChannel(GPUResources& shared, CompressVolumeResources& resources, const VolumeChannel* pChannels, uint channelCount, uint sizeX, uint sizeY, uint sizeZ, uint numLevels, bool doRLEOnlyOnLvl0)
{
    bool longSymbols = (shared.getConfig().log2HuffmanDistinctSymbolCountMax > 16);
    if(longSymbols) {
        decompressVolumeFloatMultiChannel<Symbol32>(shared, resources, pChannels, channelCount, sizeX, sizeY, sizeZ, numLevels, doRLEOnlyOnLvl0);
    } else {
        decompressVolumeFloatMultiChannel<Symbol16>(shared, resources, pChannels, channelCount, sizeX, sizeY, sizeZ, numLevels, doRLEOnlyOnLvl0);
    }
}

}
