#ifndef __TUM3D_CUDACOMPRESS__INSTANCE_H__
#define __TUM3D_CUDACOMPRESS__INSTANCE_H__


#include <cudaCompress/global.h>

#include <cuda_runtime.h>


namespace cudaCompress {

// A cudaCompress::Instance encapsulates all resources required for entropy coding.
// The resources are pre-allocated for efficiency.
class Instance;

// Create a new cudaCompress::Instance.
// cudaDevice is the index of the CUDA device to use; pass -1 to use the current (default) one.
// streamCountMax is the max number of symbol streams to be encoded at once.
// elemCountPerStreamMax is the max number of values per stream.
// codingBlockSize specifies the parallelization granularity for the Huffman coder.
//   Smaller values result in more parallelism, but may hurt the compression rate.
//   Default: 128
// log2HuffmanDistinctSymbolCountMax is the max number of bits in the input values for the Huffman coder.
//   Should preferably be <= 16, must be <= 24; large values will result in higher memory usage and reduced performance.
//   Default: 14
CUCOMP_DLL Instance* createInstance(
    int cudaDevice,
    uint streamCountMax,
    uint elemCountPerStreamMax,
    uint codingBlockSize = 0,
    uint log2HuffmanDistinctSymbolCountMax = 0);
// Destroy a cudaCompress::Instance created previously by createInstance.
CUCOMP_DLL void  destroyInstance(Instance* pInstance);

// Query Instance parameters.
CUCOMP_DLL int getInstanceCudaDevice(const Instance* pInstance);
CUCOMP_DLL uint getInstanceStreamCountMax(const Instance* pInstance);
CUCOMP_DLL uint getInstanceElemCountPerStreamMax(const Instance* pInstance);
CUCOMP_DLL uint getInstanceCodingBlockSize(const Instance* pInstance);
CUCOMP_DLL uint getInstanceLog2HuffmanDistinctSymbolCountMax(const Instance* pInstance);
CUCOMP_DLL bool getInstanceUseLongSymbols(const Instance* pInstance);

// Set a cudaStream to use for all cudaCompress kernels.
CUCOMP_DLL void setInstanceCudaStream(Instance* pInstance, cudaStream_t str);
CUCOMP_DLL cudaStream_t getInstanceCudaStream(const Instance* pInstance);

}


#endif
