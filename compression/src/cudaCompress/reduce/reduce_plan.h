#ifndef __REDUCE_PLAN_H__
#define __REDUCE_PLAN_H__


#include <cudaCompress/global.h>


namespace cudaCompress {

class ReducePlan
{
public:
    ReducePlan(size_t elemSizeBytes, size_t numElements);
    ~ReducePlan();

    size_t m_numElements;     // Maximum number of input elements
    size_t m_elemSizeBytes;   // Size of each element in bytes, i.e. sizeof(T)
    uint   m_threadsPerBlock; // number of threads to launch per block
    uint   m_maxBlocks;       // maximum number of blocks to launch
    void*  m_blockSums;       // Intermediate block sums array
};

}


#endif
