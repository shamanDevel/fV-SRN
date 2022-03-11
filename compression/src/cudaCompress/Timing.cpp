#include "memtrace.h"
#include <cudaCompress/Timing.h>

#include <cudaCompress/cudaUtil.h>
#include <cudaCompress/util.h>

#include <cudaCompress/InstanceImpl.h>


namespace cudaCompress {

void setTimingDetail(Instance* pInstance, ETimingDetail detail)
{
    pInstance->setTimingDetail(detail);
}

void getTimings(Instance* pInstance, std::vector<std::string>& names, std::vector<float>& times)
{
    pInstance->getTimings(names, times);
}

void printTimings(Instance* pInstance)
{
    pInstance->printTimings();
}

void resetTimings(Instance* pInstance)
{
    pInstance->resetTimings();
}

}
