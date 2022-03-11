#ifndef __TUM3D_CUDACOMPRESS__HISTOGRAM_H__
#define __TUM3D_CUDACOMPRESS__HISTOGRAM_H__


#include <cudaCompress/global.h>


namespace cudaCompress {

class Instance;


size_t histogramGetRequiredMemory(const Instance* pInstance);
bool histogramInit(Instance* pInstance);
bool histogramShutdown(Instance* pInstance);

uint histogramGetElemCountIncrement();
uint histogramGetPaddedElemCount(uint elemCount);
void histogramPadData(Instance* pInstance, ushort* dpData, uint elemCount);
void histogramPadData(Instance* pInstance, uint*   dpData, uint elemCount);

bool histogram(Instance* pInstance, uint* pdpHistograms[], uint histogramCount, const ushort* pdpData[], const uint* pElemCount, uint binCount);
bool histogram(Instance* pInstance, uint* pdpHistograms[], uint histogramCount, const uint*   pdpData[], const uint* pElemCount, uint binCount);

}


#endif
