#include "memtrace.h"
#include <cudaCompress/PackInc.h>

#include <cassert>

#include <cudaCompress/tools/Operator.h>

#include <cudaCompress/cudaUtil.h>
#include <cudaCompress/util.h>

#include <cudaCompress/InstanceImpl.h>


#include <cudaCompress/reduce/reduce_app.cui>
#include <cudaCompress/scan/scan_app.cui>


namespace cudaCompress {


static uint getValueCountMax(const Instance* pInstance)
{
    return (pInstance->m_elemCountPerStreamMax + pInstance->m_codingBlockSize - 1) / pInstance->m_codingBlockSize;
}


size_t packIncGetRequiredMemory(const Instance* pInstance)
{
    uint valueCountMax = getValueCountMax(pInstance);

    size_t size = 0;

    // encode and decode: dpValueIncrements
    size += getAlignedSize(valueCountMax * sizeof(uint), 128);
    // encode: dpReduceOut
    size += getAlignedSize(sizeof(uint), 128);

    return size;
}

bool packIncInit(Instance* pInstance)
{
    return true;
}

bool packIncShutdown(Instance* pInstance)
{
    return true;
}


template<typename TOut>
__global__ void computeIncrementsKernel(const uint* __restrict__ pValues, TOut* __restrict__ pValueIncrements, uint valueCount)
{
    for(uint index = blockIdx.x * blockDim.x + threadIdx.x; index < valueCount; index += gridDim.x * blockDim.x) {
        uint value = pValues[index];
        uint previous = (index == 0 ? 0 : pValues[index-1]);
        pValueIncrements[index] = TOut(value - previous);
    }
}

__global__ void packIncrementsKernel(const uint* __restrict__ pValueIncrements, uint* __restrict__ pPackedValueIncrements, uint valueCount, uint packedWordCount, uint bitsPerValue)
{
    for(uint packedIndex = blockIdx.x * blockDim.x + threadIdx.x; packedIndex < packedWordCount; packedIndex += gridDim.x * blockDim.x) {
        uint bitIndex = packedIndex * 32;
        uint valueIndex = bitIndex / bitsPerValue;
        uint leftOverhang = bitIndex % bitsPerValue;
        int bitShift = 32 - bitsPerValue + leftOverhang;

        uint count = min(valueCount - valueIndex, (32 + leftOverhang + bitsPerValue - 1) / bitsPerValue);
        
        uint result = 0;
        for(uint i = 0; i < count; i++) {
            uint value = pValueIncrements[valueIndex];
            if(bitShift > 0)
                value <<= bitShift;
            else
                value >>= -bitShift;

            result |= value;

            valueIndex++;
            bitShift -= bitsPerValue;
        }

        pPackedValueIncrements[packedIndex] = result;
    }
}

__global__ void unpackIncrementsKernel(uint* __restrict__ pValueIncrements, const uint* __restrict__ pPackedValueIncrements, uint valueCount, uint bitsPerValue)
{
    uint mask = (1 << bitsPerValue) - 1;
    for(uint index = blockIdx.x * blockDim.x + threadIdx.x; index < valueCount; index += gridDim.x * blockDim.x) {
        uint bitIndex = index * bitsPerValue;
        uint packedIndex = bitIndex / 32;

        uint leftBitOffset = bitIndex % 32;
        int bitShift = bitsPerValue + leftBitOffset - 32;

        uint value = pPackedValueIncrements[packedIndex];
        if(bitShift > 0)
            value <<= bitShift;
        else
            value >>= -bitShift;
        uint result = value & mask;

        if(bitShift > 0) {
            bitShift -= 32;
            value = pPackedValueIncrements[packedIndex + 1];
            value >>= -bitShift;
            result |= value & mask;
        }

        pValueIncrements[index] = result;
    }
}


bool packInc(Instance* pInstance, const uint* dpValues, uint* dpPackedValueIncrements, uint valueCount, uint& bitsPerValue)
{
    uint* dpValueIncrements = pInstance->getBuffer<uint>(valueCount);
    uint* dpReduceOut = pInstance->getBuffer<uint>(1);

    // build the increments
    uint blockSize = 256;
    uint blockCount = min((valueCount + blockSize - 1) / blockSize, 512u);

    computeIncrementsKernel<uint><<<blockCount, blockSize>>>(dpValues, dpValueIncrements, valueCount);
    cudaCheckMsg("computeIncrementsKernel execution failed");

    // find max value, compute packed size
    uint valueMax = 0;
    reduceArray<uint, OperatorMax<uint> >(dpReduceOut, dpValueIncrements, valueCount, pInstance->m_pReducePlan);
    cudaCheckMsg("packInc: error in reduceArray");
    cudaSafeCall(cudaMemcpy(&valueMax, dpReduceOut, sizeof(uint), cudaMemcpyDeviceToHost));

    bitsPerValue = getRequiredBits(valueMax);

    uint packedWordCount = (valueCount * bitsPerValue + 31) / 32;

    // pack
    blockSize = 256;
    blockCount = min((packedWordCount + blockSize - 1) / blockSize, 512u);

    packIncrementsKernel<<<blockCount, blockSize>>>(dpValueIncrements, dpPackedValueIncrements, valueCount, packedWordCount, bitsPerValue);
    cudaCheckMsg("packIncrementsKernel execution failed");

    pInstance->releaseBuffers(2);

    return true;
}

bool unpackInc(Instance* pInstance, uint* dpValues, const uint* dpPackedValueIncrements, uint valueCount, uint bitsPerValue)
{
    uint* dpValueIncrements = pInstance->getBuffer<uint>(valueCount);

    // unpack
    uint blockSize = 256;
    uint blockCount = min((valueCount + blockSize - 1) / blockSize, 512u);

    unpackIncrementsKernel<<<blockCount, blockSize>>>(dpValueIncrements, dpPackedValueIncrements, valueCount, bitsPerValue);
    cudaCheckMsg("unpackIncrementsKernel execution failed");

    // scan to build absolute offsets
    scanArray<uint, uint, false>(dpValues, dpValueIncrements, valueCount, pInstance->m_pScanPlan, pInstance->m_stream);
    cudaCheckMsg("unpackInc: error in scanArray");

    pInstance->releaseBuffer();

    return true;
}

bool packInc16(Instance* pInstance, const uint* dpValues, ushort* dpValueIncrements, uint valueCount)
{
    if(valueCount == 0)
        return true;

    uint* dpValueIncrementsTemp = pInstance->getBuffer<uint>(valueCount);

    // build the increments
    uint blockSize = 256;
    uint blockCount = min((valueCount + blockSize - 1) / blockSize, 512u);

    // don't write directly into dpValueIncrements because it may alias dpValues...
    computeIncrementsKernel<ushort><<<blockCount, blockSize>>>(dpValues, (ushort*)dpValueIncrementsTemp, valueCount);
    cudaCheckMsg("computeIncrementsKernel execution failed");

    // copy to output array
    cudaSafeCall(cudaMemcpyAsync(dpValueIncrements, dpValueIncrementsTemp, valueCount * sizeof(ushort), cudaMemcpyDeviceToDevice, pInstance->m_stream));

    pInstance->releaseBuffer();

    return true;
}

bool unpackInc16(Instance* pInstance, uint* dpValues, const ushort* dpValueIncrements, uint valueCount)
{
    if(valueCount == 0)
        return true;

    // scan values to build absolute offsets
    // FIXME This is *not* safe if dpValues and dpValueIncrements alias, and the scan needs > 1 block!
    //       (Would be okay if sizeof(*dpValues) == sizeof(*dpValueIncrements), which is not the case here!)
    scanArray<ushort, uint, false>(dpValues, dpValueIncrements, valueCount, pInstance->m_pScanPlan, pInstance->m_stream);
    cudaCheckMsg("unpackInc16: error in scanArray");

    return true;
}


void packInc16CPU(const uint* pValues, ushort* pValueIncrements, uint valueCount)
{
    // written to work correctly even when pValues and pValueIncrements alias
    uint prev = pValues[0];
    pValueIncrements[0] = prev;
    for(uint i = 1; i < valueCount; i++) {
        uint value = pValues[i];
        pValueIncrements[i] = pValues[i] - prev;
        prev = value;
    }
}

void unpackInc16CPU(uint* pValues, const ushort* pValueIncrements, uint valueCount)
{
    uint total = 0;
    for(uint i = 0; i < valueCount; i++) {
        total += pValueIncrements[i];
        pValues[i] = total;
    }
}

}
