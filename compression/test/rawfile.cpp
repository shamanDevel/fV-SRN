#include "rawfile.h"

#include <algorithm>
#include <fstream>


bool readByteRaw(const std::string& filename, uint elemCount, byte* pResult)
{
    std::ifstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    file.read((char*)pResult, elemCount);
    file.close();

    return true;
}

bool writeByteRaw(const std::string& filename, uint elemCount, const byte* pData)
{
    std::ofstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    file.write((char*)pData, elemCount);

    return true;
}

bool readByteRawAsShort(const std::string& filename, uint elemCount, short* pResult)
{
    std::ifstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    byte* pData = new byte[elemCount];
    file.read((char*)pData, elemCount);
    file.close();

    for (uint i = 0; i < elemCount; i++) {
        pResult[i] = short(pData[i]) - 128;
    }

    delete[] pData;
    return true;
}

bool writeByteRawFromShort(const std::string& filename, uint elemCount, const short* pData)
{
    std::ofstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    byte* pDataByte = new byte[elemCount];
    for (uint i = 0; i < elemCount; i++) {
        pDataByte[i] = byte(std::min(255, std::max(0, pData[i] + 128)));
    }

    file.write((char*)pDataByte, elemCount);

    delete[] pDataByte;
    return true;
}

bool readByteRawAsFloat(const std::string& filename, uint elemCount, float* pResult)
{
    std::ifstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    byte* pData = new byte[elemCount];
    file.read((char*)pData, elemCount);
    file.close();

    for (uint i = 0; i < elemCount; i++) {
        pResult[i] = float(pData[i]) - 128.0f;
    }

    delete[] pData;
    return true;
}

bool writeByteRawFromFloat(const std::string& filename, uint elemCount, const float* pData)
{
    std::ofstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    byte* pDataByte = new byte[elemCount];
    for (uint i = 0; i < elemCount; i++) {
        pDataByte[i] = byte(std::min(255.0f, std::max(0.0f, pData[i] + 128.0f + 0.5f)));
    }

    file.write((char*)pDataByte, elemCount);

    delete[] pDataByte;
    return true;
}

bool compareByteRaws(const std::string& filename1, const std::string& filename2, uint elemCount)
{
    std::ifstream file1(filename1.c_str(), std::ios_base::binary);
    if (!file1.good())
        return false;

    std::ifstream file2(filename2.c_str(), std::ios_base::binary);
    if (!file2.good()) {
        file1.close();
        return false;
    }

    byte* pData1 = new byte[elemCount];
    file1.read((char*)pData1, elemCount);
    file1.close();

    byte* pData2 = new byte[elemCount];
    file2.read((char*)pData2, elemCount);
    file2.close();

    //for(int z = 0; z < 256; z++) {
    //    for(int y = 0; y < 256; y++) {
    //        for(int x = 0; x < 256; x++) {
    //            int index = x + 256 * (y + 256 * z);
    //            if(pData1[index] != pData2[index]) {
    //                printf("index %3i %3i %3i: %4i %4i\n", x, y, z, pData1[index], pData2[index]);
    //                break;
    //            }
    //        }
    //    }
    //}
    bool result = true;
    uint count = 0;
    for (uint elem = 0; elem < elemCount; elem++) {
        if (pData1[elem] != pData2[elem]) {
            result = false;
            count++;
            //break;
        }
    }
    if (count) printf("Different count: %i\n", count);

    delete[] pData2;
    delete[] pData1;

    return result;
}

float computePSNRByteRaws(const std::string& filename1, const std::string& filename2, uint elemCount)
{
    float result;
    if (!computePSNRByteRaws(filename1, filename2, elemCount, &result, 1))
        return -1.0f;
    return result;
}

bool computePSNRByteRaws(const std::string& filename1, const std::string& filename2, uint elemCount, float* pResult, uint channelCount)
{
    std::ifstream file1(filename1.c_str(), std::ios_base::binary);
    if (!file1.good())
        return false;

    std::ifstream file2(filename2.c_str(), std::ios_base::binary);
    if (!file2.good()) {
        file1.close();
        return false;
    }

    byte* pData1 = new byte[elemCount * channelCount];
    file1.read((char*)pData1, elemCount * channelCount);
    file1.close();

    byte* pData2 = new byte[elemCount * channelCount];
    file2.read((char*)pData2, elemCount * channelCount);
    file2.close();

    for (uint channel = 0; channel < channelCount; channel++)
        pResult[channel] = (float)computePSNR(pData1, pData2, elemCount, channelCount, channel);

    delete[] pData2;
    delete[] pData1;

    return true;
}


bool readShortRaw(const std::string& filename, uint elemCount, short* pResult)
{
    std::ifstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    size_t bytes = elemCount * sizeof(short);
    file.read((char*)pResult, bytes);
    size_t bytesRead = file.gcount();
    file.close();

    return bytes == bytesRead;;
}

bool writeShortRaw(const std::string& filename, uint elemCount, const short* pData)
{
    std::ofstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    file.write((char*)pData, elemCount * sizeof(short));

    return true;
}

bool compareShortRaws(const std::string& filename1, const std::string& filename2, uint elemCount)
{
    std::ifstream file1(filename1.c_str(), std::ios_base::binary);
    if (!file1.good())
        return false;

    std::ifstream file2(filename2.c_str(), std::ios_base::binary);
    if (!file2.good()) {
        file1.close();
        return false;
    }

    short* pData1 = new short[elemCount];
    file1.read((char*)pData1, elemCount * sizeof(short));
    file1.close();

    short* pData2 = new short[elemCount];
    file2.read((char*)pData2, elemCount * sizeof(short));
    file2.close();

    bool result = true;
    //for(int z = 0; z < size.z(); z++) {
    //    for(int y = 0; y < size.y(); y++) {
    //        for(int x = 0; x < size.x(); x++) {
    //            int index = x + size.x() * (y + size.y() * z);
    //            if(pData1[index] != pData2[index]) {
    //                result = false;
    //                //printf("index %2i %2i %2i: %3i %3i\n", x, y, z, pData1[index], pData2[index]);
    //            }
    //        }
    //    }
    //}
    for (uint elem = 0; elem < elemCount; elem++) {
        if (pData1[elem] != pData2[elem]) {
            //printf("%i  %i %i\n", elem, (int)pData1[elem], (int)pData2[elem]);
            result = false;
            break;
        }
    }

    delete[] pData2;
    delete[] pData1;

    return result;
}

bool readFloatRaw(const std::string& filename, uint elemCount, float* pResult)
{
    std::ifstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    file.read((char*)pResult, elemCount * sizeof(float));
    file.close();

    return true;
}

bool writeFloatRaw(const std::string& filename, uint elemCount, const float* pData)
{
    std::ofstream file(filename.c_str(), std::ios_base::binary);
    if (!file.good())
        return false;

    file.write((char*)pData, elemCount * sizeof(float));

    return true;
}

void computeStatsFloatArrays(const float* pData1, const float* pData2, uint elemCount, float* pRange, float* pMaxError, float* pRMSError, float* pPSNR, float* pSNR, uint channelCount)
{
    for (uint channel = 0; channel < channelCount; channel++) {
        if (pRange)
            pRange[channel] = (float)computeRange(pData1, elemCount, channelCount, channel);

        if (pMaxError)
            pMaxError[channel] = (float)computeMaxAbsError(pData1, pData2, elemCount, channelCount, channel);
        if (pRMSError)
            pRMSError[channel] = (float)computeRMSError(pData1, pData2, elemCount, channelCount, channel);
        if (pPSNR)
            pPSNR[channel] = (float)computePSNR(pData1, pData2, elemCount, channelCount, channel);
        if (pSNR)
            pSNR[channel] = (float)computeSNR(pData1, pData2, elemCount, channelCount, channel);
    }
}

bool computeStatsFloatRaws(const std::string& filename1, const std::string& filename2, uint elemCount, float* pRange, float* pMaxError, float* pRMSError, float* pPSNR, float* pSNR, uint channelCount)
{
    std::ifstream file1(filename1.c_str(), std::ios_base::binary);
    if (!file1.good())
        return false;

    std::ifstream file2(filename2.c_str(), std::ios_base::binary);
    if (!file2.good()) {
        file1.close();
        return false;
    }

    float* pData1 = new float[elemCount * channelCount];
    file1.read((char*)pData1, elemCount * channelCount * sizeof(float));
    if (file1.fail()) printf("FAIL\n");
    file1.close();

    float* pData2 = new float[elemCount * channelCount];
    file2.read((char*)pData2, elemCount * channelCount * sizeof(float));
    if (file2.fail()) printf("FAIL\n");
    file2.close();

    computeStatsFloatArrays(pData1, pData2, elemCount, pRange, pMaxError, pRMSError, pPSNR, pSNR, channelCount);

    delete[] pData2;
    delete[] pData1;

    return true;
}