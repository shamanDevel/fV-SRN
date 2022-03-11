#include "memtrace.h"
#include "scan_plan.h"

#include <cstdlib>

#include <cuda_runtime_api.h>

#include <cudaCompress/cudaUtil.h>

#include "scan_globals.h"


namespace cudaCompress {

ScanPlan::ScanPlan(size_t elemSizeBytes, size_t numElements)
: m_numElements(numElements),
  m_elemSizeBytes(elemSizeBytes),
  m_blockSums(0),
  m_numLevels(0),
  m_numRows(0),
  m_rowPitches(nullptr)
{
    allocate(elemSizeBytes, numElements, 1, 0);
}

ScanPlan::ScanPlan(size_t elemSizeBytes, size_t numElements, size_t numRows, size_t rowPitch)
: m_numElements(numElements),
  m_elemSizeBytes(elemSizeBytes),
  m_blockSums(0),
  m_numLevels(0),
  m_numRows(0),
  m_rowPitches(nullptr)
{
    allocate(elemSizeBytes, numElements, numRows, rowPitch);
}
  
void ScanPlan::allocate(size_t elemSizeBytes, size_t numElements, size_t numRows, size_t rowPitch)
{
    const size_t blockSize = SCAN_ELTS_PER_THREAD * SCAN_CTA_SIZE;

    m_numElements = numElements;
    m_numRows = numRows;
    m_elemSizeBytes = elemSizeBytes;

    // find required number of levels
    size_t level = 0;
    size_t numElts = m_numElements;
    do
    {
        size_t numBlocks = (numElts + blockSize - 1) / blockSize;
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    m_numLevels = level;

    m_blockSums = (void**) malloc(m_numLevels * sizeof(void*));

    if (m_numRows > 1)
    {
        m_rowPitches = (size_t*) malloc((m_numLevels + 1) * sizeof(size_t));
        m_rowPitches[0] = rowPitch;
    }

    // allocate storage for block sums
    numElts = m_numElements;
    level = 0;
    do
    {
        size_t numBlocks = (numElts + blockSize - 1) / blockSize;
        if (numBlocks > 1) 
        {
            // Use cudaMallocPitch for multi-row block sums to ensure alignment
            if (m_numRows > 1)
            {
                size_t dpitch;
                cudaSafeCall(cudaMallocPitch((void**)&(m_blockSums[level]), &dpitch, numBlocks * m_elemSizeBytes, numRows));
                m_rowPitches[level+1] = dpitch / m_elemSizeBytes;
            }
            else
            {
                cudaSafeCall(cudaMalloc((void**)&(m_blockSums[level]), numBlocks * m_elemSizeBytes));
            }
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    cudaCheckMsg("ScanPlan::allocate");
}

ScanPlan::~ScanPlan()
{
    for (unsigned int i = 0; i < m_numLevels; i++)
    {
        cudaFree(m_blockSums[i]);
    }

    cudaCheckMsg("ScanPlan::~ScanPlan");

    free(m_blockSums);
    m_blockSums = nullptr;
    if(m_numRows > 1)
    {
        free((void*)m_rowPitches);
        m_rowPitches = nullptr;
    }
    m_numElements = 0;
    m_numLevels = 0;
}

}
