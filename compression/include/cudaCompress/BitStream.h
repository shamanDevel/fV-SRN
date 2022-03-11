#ifndef __TUM3D_CUDACOMPRESS__BIT_STREAM_H__
#define __TUM3D_CUDACOMPRESS__BIT_STREAM_H__


#include <cudaCompress/global.h>

#include <algorithm>
#include <cassert>
#include <vector>


namespace cudaCompress {

class BitStreamReadOnly
{
public:
    BitStreamReadOnly(const uint* pBuffer, uint bitSize);
    BitStreamReadOnly(BitStreamReadOnly&& other);
    virtual ~BitStreamReadOnly();

    uint getBitPosition() const;
    bool setBitPosition(uint bitPos);

    uint getBitSize() const { return m_bitSize; }

    // read a single bit into the least significant bit of data
    uint readBit(uint& data);
    // read the next bitcount bits into the bitcount least significant bits of data
    // the rest of data is left unmodified
    // returns the number of bits actually read (-> eof!)
    uint readBits(uint& data, uint bitCount);
    // skip the specified number of bits
    // returns the number of bits actually skipped (-> eof!)
    uint skipBits(uint bitCount);

    // read "count" naturally aligned elements of type T
    // the current bit position will be properly aligned automatically
    template<typename T>
    uint readAligned(T* pTarget, uint count);
    // skip "count" naturally aligned elements of type T
    // the current bit position will be properly aligned automatically
    template<typename T>
    uint skipAligned(uint count);
    // align the current position to the next T, and return the corresponding pointer
    template<typename T>
    const T* getAlignedPointer();

    // align the current position to sizeof(T) bytes
    // returns true if successful, false if align would take us outside the stream
    template<typename T>
    bool align();
    // align the current position to "byteCount" bytes
    bool align(uint byteCount);

    const uint* getRaw() const;
    uint getRawSizeUInts() const;
    uint getRawSizeBytes() const;

protected:
    const uint* m_pBuffer;
    uint m_bitSize;

    uint m_bufferPos;
    uint m_bitPos;

private:
    BitStreamReadOnly(const BitStreamReadOnly&);
    BitStreamReadOnly& operator=(const BitStreamReadOnly&);
};


class BitStream : public BitStreamReadOnly
{
public:
    // if pVector is nullptr, construct own storage
    BitStream(std::vector<uint>* pVector = nullptr);
    BitStream(BitStream&& other);
    virtual ~BitStream();

    void setBitSize(uint bitSize);
    void reserveBitSize(uint bitSize);

    // write the bitcount least significant bits of data, from more to less significant
    // buffer will be expanded if necessary
    void writeBits(uint data, uint bitCount);

    // write "count" naturally aligned elements of type T
    // the current bit position will be properly aligned automatically
    template<typename T>
    void writeAligned(const T* pSource, uint count);

    // align the current position to sizeof(T) bytes
    // even if it takes us beyond the current end of the bitstream (fill with zero bits)
    template<typename T>
    void writeAlign();
    // align the current position to "byteCount" bytes
    // even if it takes us beyond the current end of the bitstream (fill with zero bits)
    void writeAlign(uint byteCount);

    uint* getRaw();

private:
    std::vector<uint>* m_pVector;
    bool m_ownBuffer;

    void resizeVector(uint newSize);

    BitStream(const BitStream&);
    BitStream& operator=(const BitStream&);
};



inline BitStreamReadOnly::BitStreamReadOnly(const uint* pBuffer, uint bitSize)
    : m_pBuffer(pBuffer), m_bitSize(bitSize), m_bufferPos(0), m_bitPos(0)
{
}

inline BitStreamReadOnly::BitStreamReadOnly(BitStreamReadOnly&& other)
    : m_pBuffer(other.m_pBuffer), m_bitSize(other.m_bitSize), m_bufferPos(other.m_bufferPos), m_bitPos(other.m_bitPos)
{
    other.m_pBuffer = nullptr;
    other.m_bitSize = 0;
    other.m_bufferPos = 0;
    other.m_bitPos = 0;
}

inline BitStreamReadOnly::~BitStreamReadOnly()
{
}

inline uint BitStreamReadOnly::getBitPosition() const
{
    return m_bufferPos * 32 + m_bitPos;
}

inline bool BitStreamReadOnly::setBitPosition(uint bitPos)
{
    if(bitPos > m_bitSize) {
        // eof, set position to end
        m_bufferPos = m_bitSize / 32;
        m_bitPos = m_bitSize % 32;
        return false;
    }

    m_bufferPos = bitPos / 32;
    m_bitPos = bitPos % 32;
    return true;
}

inline uint BitStreamReadOnly::readBit(uint& data)
{
    if(getBitPosition() == m_bitSize)
        return 0;

    // clear lsb
    data &= ~uint(1);
    // set lsb as next bit from buffer
    data |= ((m_pBuffer[m_bufferPos] >> (31 - m_bitPos)) & 1);

    // advance in stream
    m_bitPos++;
    m_bufferPos += m_bitPos / 32;
    m_bitPos %= 32;

    return 1;
}

inline uint BitStreamReadOnly::readBits(uint& data, uint bitCount)
{
    assert(bitCount <= 8 * sizeof(data));

    uint bitCountRead = std::min(bitCount, m_bitSize - getBitPosition());

    bitCount = bitCountRead;
    //TODO optimize? this needs at most 2 iterations...
    while(bitCount > 0) {
        // number of bits to read from current uint
        uint bitCountCur = std::min(bitCount, 32 - m_bitPos);
        // bitmask for this number of bits
        uint mask = (bitCountCur == 32 ? ~uint(0) : uint((1 << bitCountCur) - 1));

        // get bits from buffer
        uint offset = 32 - bitCountCur - m_bitPos;
        uint buffer = (m_pBuffer[m_bufferPos] >> offset) & mask;

        // assign to data
        uint dataPos = bitCount - bitCountCur;
        data = (data & ~(mask << dataPos)) | (buffer << dataPos);

        // advance in stream
        m_bitPos += bitCountCur;
        m_bufferPos += m_bitPos / 32;
        m_bitPos %= 32;

        bitCount -= bitCountCur;
    }

    return bitCountRead;
}

inline uint BitStreamReadOnly::skipBits(uint bitCount)
{
    uint bitCountSkipped = std::min(bitCount, m_bitSize - getBitPosition());

    m_bitPos += bitCountSkipped;
    m_bufferPos += m_bitPos / 32;
    m_bitPos %= 32;

    return bitCountSkipped;
}

template<typename T>
inline uint BitStreamReadOnly::readAligned(T* pTarget, uint count)
{
    // HACK: align to uint before and after *Aligned, otherwise they don't
    //       mix with the *Bits functions because of little endian byte order
    align<uint>();

    align<T>();

    // get read position
    const byte* pSource = (byte*)(m_pBuffer + m_bufferPos) + (m_bitPos / 8);

    // compute number of elements to read
    uint bytesAvailable = (m_bitSize - getBitPosition()) / 8;
    uint elemsAvailable = bytesAvailable / sizeof(T);
    uint elemsRead = std::min(elemsAvailable, count);

    // read data from buffer
    memcpy(pTarget, pSource, elemsRead * sizeof(T));

    // update position
    m_bitPos += elemsRead * sizeof(T) * 8;
    m_bufferPos += m_bitPos / 32;
    m_bitPos %= 32;

    align<uint>();

    return elemsRead;
}

template<typename T>
inline uint BitStreamReadOnly::skipAligned(uint count)
{
    // HACK: align to uint before and after *Aligned, otherwise they don't
    //       mix with the *Bits functions because of little endian byte order
    align<uint>();

    align<T>();

    // compute number of elements to skip
    uint bytesAvailable = (m_bitSize - getBitPosition()) / 8;
    uint elemsAvailable = bytesAvailable / sizeof(T);
    uint elemsRead = std::min(elemsAvailable, count);

    // update position
    m_bitPos += elemsRead * sizeof(T) * 8;
    m_bufferPos += m_bitPos / 32;
    m_bitPos %= 32;

    align<uint>();

    return elemsRead;
}

template<typename T>
const T* BitStreamReadOnly::getAlignedPointer()
{
    // HACK: align to uint before and after *Aligned, otherwise they don't
    //       mix with the *Bits functions because of little endian byte order
    align<uint>();

    align<T>();

    const T* pCur = reinterpret_cast<const T*>(getRaw()) + getBitPosition() / (8 * sizeof(T));

    align<uint>();

    return pCur;
}


template<typename T>
inline bool BitStreamReadOnly::align()
{
    return align(sizeof(T));
}

inline bool BitStreamReadOnly::align(uint byteCount)
{
    uint bitCount = byteCount * 8;

    uint bitOffset = m_bitPos % bitCount;
    uint bitOffsetMax = getBitSize() - getBitPosition();
    bool result = true;
    if(bitOffset > bitOffsetMax) {
        bitOffset = bitOffsetMax;
        result = false;
    }
    if(bitOffset != 0) {
        m_bitPos += bitCount - bitOffset;
        m_bufferPos += m_bitPos / 32;
        m_bitPos %= 32;
    }

    return result;
}

inline const uint* BitStreamReadOnly::getRaw() const
{
    return m_pBuffer;
}

inline uint BitStreamReadOnly::getRawSizeUInts() const
{
    return (m_bitSize + 8 * sizeof(uint) - 1) / (8 * sizeof(uint));
}

inline uint BitStreamReadOnly::getRawSizeBytes() const
{
    return (m_bitSize + 7) / 8;
}



inline BitStream::BitStream(std::vector<uint>* pVector)
    : BitStreamReadOnly(nullptr, 0)
{
    if(pVector != nullptr) {
        m_ownBuffer = false;
        m_pVector = pVector;
        m_bitSize = (uint)m_pVector->size() * 32;
    } else {
        m_ownBuffer = true;
        m_pVector = new std::vector<uint>;
        m_bitSize = 0;
    }
    m_pBuffer = m_pVector->data();
}

inline BitStream::BitStream(BitStream&& other)
    : BitStreamReadOnly(std::move(other))
    , m_ownBuffer(other.m_ownBuffer), m_pVector(other.m_pVector)
{
    other.m_ownBuffer = false;
    other.m_pVector = nullptr;
}

inline BitStream::~BitStream()
{
    if(m_ownBuffer) {
        delete m_pVector;
    }
}


inline void BitStream::setBitSize(uint bitSize)
{
    // adjust size
    m_bitSize = bitSize;
    uint bufferSize = (m_bitSize + 31) / 32;
    resizeVector(bufferSize);

    // clamp position to new size
    uint bitPos = getBitPosition();
    bitPos = std::min(bitPos, m_bitSize);
    setBitPosition(bitPos);
}

inline void BitStream::reserveBitSize(uint bitSize)
{
    uint bufferSize = (bitSize + 31) / 32;
    m_pVector->reserve(bufferSize);
}

inline void BitStream::writeBits(uint data, uint bitCount)
{
    assert(bitCount <= 8 * sizeof(data));

    // increase buffer size if necessary
    uint bitSizeNew = getBitPosition() + bitCount;
    if(bitSizeNew > m_bitSize) {
        m_bitSize = bitSizeNew;

        uint bufferSizeNew = (m_bitSize + 31) / 32;
        if(bufferSizeNew > m_pVector->size()) {
            resizeVector(bufferSizeNew);
        }
    }


    //TODO optimize? this needs at most 2 iterations...
    while(bitCount > 0) {
        // number of bits to write to current uint
        uint bitCountCur = std::min(bitCount, 32 - m_bitPos);
        // bitmask for this number of bits
        uint mask = (bitCountCur == 32 ? ~uint(0) : uint((1 << bitCountCur) - 1));

        // get corresponding bits from data
        uint maskPos = bitCount - bitCountCur;
        uint value = (data >> maskPos) & mask;

        // write to buffer
        uint& buffer = (*m_pVector)[m_bufferPos];
        uint offset = 32 - bitCountCur - m_bitPos;
        buffer = (buffer & ~(mask << offset)) | (value << offset);

        // advance in stream
        m_bitPos += bitCountCur;
        m_bufferPos += m_bitPos / 32;
        m_bitPos %= 32;

        bitCount -= bitCountCur;
    }

    assert(getBitPosition() <= m_bitSize);
}

template<typename T>
inline void BitStream::writeAligned(const T* pSource, uint count)
{
    // HACK: align to uint before and after *Aligned, otherwise they don't
    //       mix with the *Bits functions because of little endian byte order
    writeAlign<uint>();

    writeAlign<T>();

    if(count == 0)
        return;

    // store write position
    uint bufferPos = m_bufferPos;
    uint bitPos = m_bitPos;

    // update position
    m_bitPos += count * sizeof(T) * 8;
    m_bufferPos += m_bitPos / 32;
    m_bitPos %= 32;

    // increase buffer size if necessary
    if(getBitPosition() > m_bitSize) {
        m_bitSize = getBitPosition();
        uint bufferSizeNew = (m_bitSize + 31) / 32;
        resizeVector(bufferSizeNew);
    }

    // get write ptr
    byte* pTarget = (byte*)&(*m_pVector)[bufferPos] + (bitPos / 8);

    // write data into buffer
    memcpy(pTarget, pSource, count * sizeof(T));

    writeAlign<uint>();
}

template<typename T>
inline void BitStream::writeAlign()
{
    writeAlign(sizeof(T));
}

inline void BitStream::writeAlign(uint byteCount)
{
    uint bitCount = byteCount * 8;

    uint bitOffset = m_bitPos % bitCount;
    if(bitOffset != 0) {
        m_bitPos += bitCount - bitOffset;
        m_bufferPos += m_bitPos / 32;
        m_bitPos %= 32;
    }

    // increase buffer size if necessary
    if(getBitPosition() > m_bitSize) {
        m_bitSize = getBitPosition();
        uint bufferSizeNew = (m_bitSize + 31) / 32;
        resizeVector(bufferSizeNew);
    }
}

inline uint* BitStream::getRaw()
{
    return m_pVector->data();
}

inline void BitStream::resizeVector(uint newSize)
{
    m_pVector->resize(newSize, 0);
    // re-set buffer pointer of BitStreamReadOnly
    m_pBuffer = m_pVector->data();
}

}


#endif
