#include "memtrace.h"
#include <cudaCompress/Instance.h>

#include <cudaCompress/InstanceImpl.h>


namespace cudaCompress {

Instance* createInstance(int cudaDevice, uint streamCountMax, uint elemCountPerStreamMax, uint codingBlockSize, uint log2HuffmanDistinctSymbolCountMax)
{
    Instance* pInstance = new Instance();
    if(pInstance->create(cudaDevice, streamCountMax, elemCountPerStreamMax, codingBlockSize, log2HuffmanDistinctSymbolCountMax)) {
        return pInstance;
    } else {
        delete pInstance;
        return nullptr;
    }
}

void destroyInstance(Instance* pInstance)
{
    if(pInstance == nullptr) return;

    pInstance->destroy();
    delete pInstance;
}


int getInstanceCudaDevice(const Instance* pInstance)
{
    return pInstance->m_cudaDevice;
}

uint getInstanceStreamCountMax(const Instance* pInstance)
{
    return pInstance->m_streamCountMax;
}

uint getInstanceElemCountPerStreamMax(const Instance* pInstance)
{
    return pInstance->m_elemCountPerStreamMax;
}

uint getInstanceCodingBlockSize(const Instance* pInstance)
{
    return pInstance->m_codingBlockSize;
}

uint getInstanceLog2HuffmanDistinctSymbolCountMax(const Instance* pInstance)
{
    return pInstance->m_log2HuffmanDistinctSymbolCountMax;
}

bool getInstanceUseLongSymbols(const Instance* pInstance)
{
    return pInstance->m_log2HuffmanDistinctSymbolCountMax > 16;
}


void setInstanceStream(Instance* pInstance, cudaStream_t str)
{
    pInstance->m_stream = str;
}

cudaStream_t getInstanceStream(const Instance* pInstance)
{
    return pInstance->m_stream;
}


}
