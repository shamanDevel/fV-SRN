#ifndef __TUM3D_CUDACOMPRESS__BITSTREAM_GPU_CUH__
#define __TUM3D_CUDACOMPRESS__BITSTREAM_GPU_CUH__


#include <cudaCompress/global.h>


namespace cudaCompress {

struct BitStreamGPU
{
    __device__ inline BitStreamGPU(const uint* pData, uint bitoffset) : m_pData(pData)
    {
        m_bitpos = bitoffset % 32u;

        uint datapos = bitoffset / 32u;
        m_pData += datapos;

        m_cache = uint64(*m_pData++) << 32u;
        m_cache |= uint64(*m_pData++);
        //m_cache.high = *m_pData++;
        //m_cache.low  = *m_pData++;
    }

    __device__ inline uint peekBits(uint count) const
    {
        // assert(m_bitpos <= 64-count);
        return (m_cache >> (64u - count - m_bitpos)) & ((1u << count) - 1u);
        //return (m_cache.all >> (64u - count - m_bitpos)) & ((1u << count) - 1);
    }

    __device__ inline uint peekByte() const
    {
        // assert(m_bitpos <= 56);
        return uint(m_cache >> (56u - m_bitpos)) & 0xFFu;
        //return (m_cache.all >> (56u - m_bitpos)) & 0xFFu;
    }

    __device__ inline uint peekUInt() const
    {
        // assert(m_bitpos <= 32);
        return uint(m_cache >> (32u - m_bitpos));
        //return (m_cache.all >> (32u - m_bitpos));
    }

    __device__ inline void stepBits(uint count)
    {
        m_bitpos += count;
    }

    __device__ inline uint readBit()
    {
        // assert(m_bitpos < 64);
        return uint(m_cache >> (63u - m_bitpos++)) & 1u;
        //return (m_cache.all >> (63u - m_bitpos++)) & 1u;
    }

    __device__ inline uint readBits(uint count)
    {
        uint result = peekBits(count);
        m_bitpos += count;
        return result;
    }

    __device__ inline void fillCache()
    {
        // note: this assumes that there is another uint available,
        //       ie the bitstream has to be padded by at least one uint at the end
        if(m_bitpos >= 32u) {
            m_cache = (m_cache << 32u) | uint64(*m_pData++);
            //m_cache.high = m_cache.low;
            //m_cache.low  = *m_pData++;
            m_bitpos -= 32u;
        }
    }

    const uint* m_pData;
    uint64 m_cache;
    //union {
    //    uint64 all;
    //    struct { uint low; uint high; };
    //} m_cache;
    uint m_bitpos;
};

}


#endif
