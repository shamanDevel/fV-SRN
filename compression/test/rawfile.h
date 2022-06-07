#ifndef __rawfile_h__
#define __rawfile_h__

#include <string>
#include <math.h>

#include <cudaCompress/global.h>
using cudaCompress::byte;
using cudaCompress::uint;


template<typename T>
double computeRange(const T* pData, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double min = std::numeric_limits<double>::max();
    double max = -std::numeric_limits<double>::max();

    for (unsigned int i = 0; i < count; i++) {
        double val = double(pData[i * numcomps + comp]);
        if (val < min)
            min = val;
        if (val > max)
            max = val;
    }

    return max - min;
}

template<typename T>
double computeAverage(const T* pData, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double sum = 0.0;
    for (unsigned int i = 0; i < count; i++) {
        sum += double(pData[i * numcomps + comp]);
    }

    return sum / double(count);
}

template<typename T>
double computeVariance(const T* pData, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double sum = 0.0;
    double sumSq = 0.0;
    for (unsigned int i = 0; i < count; i++) {
        double val = double(pData[i * numcomps + comp]);
        sum += val;
        sumSq += val * val;
    }
    double avg = sum / double(count);
    double avgSq = sumSq / double(count);

    return avgSq - avg * avg;
}

template<typename T>
double computeAvgError(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double result = 0.0;
    for (unsigned int i = 0; i < count; i++) {
        double diff = double(pReconst[i * numcomps + comp]) - double(pData[i * numcomps + comp]);
        result += diff;
    }
    result /= double(count);

    return result;
}

template<typename T>
double computeAvgAbsError(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double result = 0.0;
    for (unsigned int i = 0; i < count; i++) {
        double diff = double(pReconst[i * numcomps + comp]) - double(pData[i * numcomps + comp]);
        result += abs(diff);
    }
    result /= double(count);

    return result;
}

template<typename T>
double computeMaxAbsError(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double result = 0.0;
    for (unsigned int i = 0; i < count; i++) {
        double diff = double(pReconst[i * numcomps + comp]) - double(pData[i * numcomps + comp]);
        result = std::max(result, abs(diff));
    }

    return result;
}

template<typename T>
double computeRMSError(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double result = 0.0;
    for (unsigned int i = 0; i < count; i++) {
        double diff = double(pData[i * numcomps + comp]) - double(pReconst[i * numcomps + comp]);
        result += diff * diff;
    }
    result /= double(count);
    result = sqrt(result);

    return result;
}

template<typename T>
double computeSNR(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double var = computeVariance(pData, count, numcomps, comp);
    double rmse = computeRMSError(pData, pReconst, count, numcomps, comp);

    return 20.0 * log10(sqrt(var) / rmse);
}

template<typename T>
double computePSNR(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double range = computeRange(pData, count, numcomps, comp);
    double rmse = computeRMSError(pData, pReconst, count, numcomps, comp);

    return 20.0 * log10(range / rmse);
}

// normalized cross-correlation
template<typename T>
double computeNCC(const T* pData, const T* pReconst, unsigned int count, unsigned int numcomps = 1, unsigned int comp = 0)
{
    double avgData = computeAverage(pData, count, numcomps, comp);
    double varData = computeVariance(pData, count, numcomps, comp);
    double avgReconst = computeAverage(pReconst, count, numcomps, comp);
    double varReconst = computeVariance(pReconst, count, numcomps, comp);

    double ncc = 0.0;
    for (unsigned int i = 0; i < count; i++) {
        ncc += (double(pData[i * numcomps + comp]) - avgData) * (double(pReconst[i * numcomps + comp]) - avgReconst);
    }

    ncc /= double(count - 1) * sqrt(varData) * sqrt(varReconst);

    return ncc;
}


bool readByteRaw(const std::string& filename, uint elemCount, byte* pResult);
bool writeByteRaw(const std::string& filename, uint elemCount, const byte* pData);

bool readByteRawAsShort(const std::string& filename, uint elemCount, short* pResult);
bool writeByteRawFromShort(const std::string& filename, uint elemCount, const short* pData);

bool readByteRawAsFloat(const std::string& filename, uint elemCount, float* pResult);
bool writeByteRawFromFloat(const std::string& filename, uint elemCount, const float* pData);

bool compareByteRaws(const std::string& filename1, const std::string& filename2, uint elemCount);

float computePSNRByteRaws(const std::string& filename1, const std::string& filename2, uint elemCount);
bool  computePSNRByteRaws(const std::string& filename1, const std::string& filename2, uint elemCount, float* pResult, uint channelCount);


bool readShortRaw(const std::string& filename, uint elemCount, short* pResult);
bool writeShortRaw(const std::string& filename, uint elemCount, const short* pData);

bool compareShortRaws(const std::string& filename1, const std::string& filename2, uint elemCount);


bool readFloatRaw(const std::string& filename, uint elemCount, float* pResult);
bool writeFloatRaw(const std::string& filename, uint elemCount, const float* pData);

void computeStatsFloatArrays(const float* pData1, const float* pData2, uint elemCount, float* pRange, float* pMaxError, float* pRMSError, float* pPSNR, float* pSNR, uint channelCount = 1);
bool computeStatsFloatRaws(const std::string& filename1, const std::string& filename2, uint elemCount, float* pRange, float* pMaxError, float* pRMSError, float* pPSNR, float* pSNR, uint channelCount = 1);


#endif