/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __SLICE_HPP__
#define __SLICE_HPP__

#include "memtrace.h"
#include <cassert>

#include <string>
#include <sstream>
#include <cmath>


// Cutout/downsampling modes
enum Reduction { Downsampling, Box, Lanczos };

class Slice {

public:

    int32_t points[3] = {INT32_MAX, INT32_MAX, 1};
    int32_t max_upper = INT32_MAX;
    Reduction reduction = Downsampling;

    Slice(int32_t lower, int32_t stride, int32_t upper, bool downsample);  // Create a slice from its data
    Slice(std::string description); // Create a slice from its NumPy-like description
    const uint32_t get_size(); // Number of elements encompassed by the slice
    const bool is_standard(); // Whether it is the (0,1,-1) slice (equivalent to doing nothing)
    void update(uint32_t size);

    friend std::ostream& operator<<(std::ostream& os, const Slice& slice);
};

inline Slice::Slice(int32_t lower, int32_t upper, int32_t stride, bool downsample=false) {
       points[0] = lower;
       points[1] = upper;
       points[2] = stride;
       downsample = downsample; // If true, new samples are computed as averages; otherwise, no interpolation is used
}

inline Slice::Slice(std::string description) {
    char delim = 0;
    if (description.find(':') != std::string::npos)
        delim = ':';
    if (description.find('/') != std::string::npos) {
        if (delim != 0)
            std::cerr << "Slicing argument \""+description+"\" not understood" << std::endl;
        delim = '/';
        reduction = Box;
    }
    if (description.find('l') != std::string::npos) {
        if (delim != 0)
            std::cerr << "Slicing argument \""+description+"\" not understood" << std::endl;
        delim = 'l';
        reduction = Lanczos;
    }
    std::stringstream ss1(description);
    std::string token;
    uint8_t n_parts = 0; // Should become 1, 2 or 3
    while(std::getline(ss1, token, delim)) {
        n_parts++;
        if (n_parts > 3)
            std::cerr << "Slicing argument \""+description+"\" not understood" << std::endl;
        std::stringstream ss2(token);
        int32_t point = INT32_MAX; // Default value, used to detect missing parts (e.g. "::")
        ss2 >> point;
        if (point != INT32_MAX)
            points[n_parts-1] = point;
    }
    if (n_parts == 1 && description[description.size()-1] != delim) // E.g. "3"; indicates a single slice
        points[1] = points[0]+1;
    if (points[2] < 0) {
        if (points[0] == INT32_MAX) points[0] = -1;
        if (points[1] == INT32_MAX) points[1] = -1;
    }
    else if (points[2] > 0) {
        if (points[0] == INT32_MAX) points[0] = 0;
        if (points[1] == INT32_MAX) points[1] = -1;
    }
    else
        std::cerr << "Slicing argument \""+description+"\" not understood" << std::endl;
    assert(points[2] != 0);
}

inline const uint32_t Slice::get_size() {
    return ceil((points[1]-points[0])/double(points[2]));
}

inline const bool Slice::is_standard() {
    return points[0] == 0 && (points[1] == -1 || points[1] == max_upper) && points[2] == 1;
}

inline void Slice::update(uint32_t size) {
    if (points[0] == -1)
        points[0] = size-1;
    else if (points[2] > 0 && points[1] == -1)
        points[1] = size;
    max_upper = size;
}

inline std::ostream& operator<<(std::ostream& os, const Slice& slice)
{
    char delim = ':';
    if (slice.reduction == Box) delim = '/';
    else if (slice.reduction == Lanczos) delim = 'l';
    os << slice.points[0] << delim << slice.points[1] << delim << slice.points[2];
    return os;
}

#endif // SLICE_HPP
