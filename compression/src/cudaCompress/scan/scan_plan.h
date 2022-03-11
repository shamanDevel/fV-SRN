#ifndef __SCAN_PLAN_H__
#define __SCAN_PLAN_H__


#include <cudaCompress/global.h>


namespace cudaCompress {

class ScanPlan
{
public:
    ScanPlan(size_t elemSizeBytes, size_t numElements);
    ScanPlan(size_t elemSizeBytes, size_t numElements, size_t numRows, size_t rowPitch);
    ~ScanPlan();

    size_t  m_numElements;   // Maximum number of input elements
    size_t  m_elemSizeBytes; // Size of each element in bytes, i.e. sizeof(T)
    void**  m_blockSums;     // Intermediate block sums array
    size_t  m_numLevels;     // Number of levels (in m_blockSums)
    size_t  m_numRows;       // Number of rows
    size_t* m_rowPitches;    // Pitch of each row in elements

private:
    void allocate(size_t elemSizeBytes, size_t numElements, size_t numRows, size_t rowPitch);
};

}


#endif
