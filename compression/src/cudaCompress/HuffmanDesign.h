#ifndef __TUM3D_CUDACOMPRESS__HUFFMAN_DESIGN_H__
#define __TUM3D_CUDACOMPRESS__HUFFMAN_DESIGN_H__


#include <cudaCompress/global.h>

#include <cudaCompress/EncodeCommon.h>
#include <cudaCompress/util.h>


namespace cudaCompress {

struct HuffmanTreeNode
{
    void init(Symbol32 symbol, uint probability)
    {
        m_symbol = symbol;
        m_probability = probability;
        m_codewordLength = -1;
        m_pLeftChild = 0;
        m_pRightChild = 0;
    }

    Symbol32 m_symbol;
    uint     m_probability;

    int      m_codewordLength;

    HuffmanTreeNode* m_pLeftChild;
    HuffmanTreeNode* m_pRightChild;

    int assignCodewordLength(int length)
    {
        m_codewordLength = length;

        int result = length;

        if(m_pLeftChild)
            result = max(result, m_pLeftChild->assignCodewordLength(length + 1));
        if(m_pRightChild)
            result = max(result, m_pRightChild->assignCodewordLength(length + 1));

        return result;
    }
};

struct HuffmanTreeNodeProbabilityInvComparer
{
    bool operator()(const HuffmanTreeNode* lhs, const HuffmanTreeNode* rhs) const
    {
        // inverted comparison!
        return rhs->m_probability < lhs->m_probability;
        //TODO? if probability is equal, prefer shorter codewords
    }
};

struct HuffmanTreeNodeCodewordLengthComparer
{
    bool operator()(const HuffmanTreeNode* lhs, const HuffmanTreeNode* rhs) const
    {
        return lhs->m_codewordLength < rhs->m_codewordLength;
    }
};

}


#endif
