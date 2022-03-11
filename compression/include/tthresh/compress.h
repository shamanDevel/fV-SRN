/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __COMPRESS_HPP__
#define __COMPRESS_HPP__

#include <ostream>
#include <vector>
#include <Eigen/Dense>
#include <compression.h>

namespace tthresh
{

    using Target = compression::TThreshTarget;

    /**
     * Compresses the input data given as doubles.
     *
     * Dimensions of the n-dimensional input data is given in \c dimensions, the data
     * itself is given in c-style ordering.
     * This means, the first dimension \c dimensions[0] is the slowest,
     * the last dimension \c dimensions[n-1] with \c n=dimensions.size()
     * moves fastest, i.e. with stride 1.
     *
     * The data is compressed so that the error metric specified by \c target
     * stays within \c targetValue.
     *
     * The compressed data is written to \c out.
     *
     * \param out the output stream
     * \param inputData the c-continous input data
     * \param dimensions the dimensions of the input data
     * \param target the error metric
     * \param targetValue the value of the error metric that needs to be fulfilled after compression
     */
    void compress(std::ostream& out, const double* inputData, const std::vector<uint32_t>& dimensions,
        Target target, double targetValue, bool verbose);
}

#endif // COMPRESS_HPP
