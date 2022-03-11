#include "memtrace.h"
#include "reduce_plan.h"

#include <cstdlib>

#include <cuda_runtime_api.h>

#include <cudaCompress/util.h>
#include <cudaCompress/cudaUtil.h>

#include "reduce_globals.h"


namespace cudaCompress {

ReducePlan::ReducePlan(size_t elemSizeBytes, size_t numElements)
: m_numElements(numElements),
  m_elemSizeBytes(elemSizeBytes),
  m_threadsPerBlock(REDUCE_CTA_SIZE),
  m_maxBlocks(64),
  m_blockSums(0)
{
    uint blocks = min(m_maxBlocks, (uint(m_numElements) + m_threadsPerBlock - 1) / m_threadsPerBlock);
    cudaMalloc(&m_blockSums, blocks * m_elemSizeBytes);

    cudaCheckMsg("allocReduceStorage");
}

ReducePlan::~ReducePlan()
{
    cudaFree(m_blockSums);
    m_blockSums = 0;

    cudaCheckMsg("freeReduceStorage");
}

}
