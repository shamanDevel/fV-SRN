#pragma once

#ifndef __CUDACC__
#include <memory>
#endif

namespace kernel
{
#define VOLUME_HISTOGRAM_NUM_BINS 512
	struct VolumeHistogram
	{
		static constexpr int NUM_BINS = VOLUME_HISTOGRAM_NUM_BINS;
		unsigned int bins[NUM_BINS];
		float minDensity;
		float maxDensity;
		unsigned int maxBinValue;
	};
#ifndef __CUDACC__
	typedef std::shared_ptr<VolumeHistogram> VolumeHistogram_ptr;
#endif
}