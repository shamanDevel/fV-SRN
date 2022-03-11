/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __DECOMPRESS_HPP__
#define __DECOMPRESS_HPP__

#include <istream>
#include <vector>

namespace tthresh
{
    /**
     * Decompresses the volume from the given input stream.
     * \param in the input stream
     * \param dimensions [out] will contain the dimensions of the array
     * \param verbose add verbose information message
     * \param debug debug messages
     * \return the C-order decompressed values
     */
    std::vector<double> decompress(std::istream& in,
        std::vector<uint32_t>& dimensions,
        bool verbose, bool debug);
}


#endif // DECOMPRESS_HPP
