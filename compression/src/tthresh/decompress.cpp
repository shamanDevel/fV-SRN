/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#include "memtrace.h"
#include <tthresh/decompress.h>

#include <chrono>
#include <iostream>

#include "utils.hpp"
#include "tucker.hpp"
#include "io.hpp"
#include "decode.hpp"
#include "Slice.hpp"
#include <Eigen/Dense>


//double maximum;
//int q;
//size_t pointer;

/////

/////

struct DecodeTimes
{
    double decode_rle_time;
    double decode_raw_time;
    double unscramble_time;
};

struct DecodeState
{
    double maximum; //written by decode_array
    int q; //written by decode_array
    size_t pointer; //written by decode_array
};

static std::vector<uint64_t> decode_array(tthresh::zs& zs, size_t size, bool is_core, DecodeState& state, bool verbose, bool debug, DecodeTimes& times)
{
    using namespace tthresh;

    uint64_t tmp = read_bits(zs, 64);
    memcpy(&state.maximum, (void*)&tmp, sizeof(tmp));

    std::vector<uint64_t> current(size, 0);

    times.decode_rle_time = 0;
    times.decode_raw_time = 0;
    times.unscramble_time = 0;

    int zeros = 0;
    bool all_raw = false;
    if (verbose && is_core)
        std::cout << "Decoding core..." << std::endl;

    size_t gpointer = 0;
    for (state.q = 63; state.q >= 0; --state.q) {
        if (verbose && is_core)
            std::cout << "Decoding core's bit plane p = " << state.q << std::endl;
        uint64_t rawsize = read_bits(zs, 64);

        size_t read_from_rle = 0;
        size_t read_from_raw = 0;

        if (all_raw) {
            std::chrono::high_resolution_clock::time_point timenow = std::chrono::high_resolution_clock::now();
            for (uint64_t pointer = 0; pointer < rawsize; ++pointer) {
                current[pointer] |= read_bits(zs, 1) << state.q;
            }
            times.unscramble_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count()/1000.;
            std::vector<size_t> rle;
            decode(zs, rle);
        }
        else {
            std::vector<bool> raw;
            std::chrono::high_resolution_clock::time_point timenow = std::chrono::high_resolution_clock::now();
            for (uint64_t i = 0; i < rawsize; ++i)
                raw.push_back(read_bits(zs, 1));
            times.decode_raw_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count()/1000.;

            std::vector<size_t> rle;
            timenow = std::chrono::high_resolution_clock::now();
            decode(zs, rle);
            times.decode_rle_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count()/1000.;

            int64_t raw_index = 0;
            int64_t rle_value = -1;
            int64_t rle_index = -1;

            timenow = std::chrono::high_resolution_clock::now();
            for (gpointer = 0; gpointer < size; ++gpointer) {
                uint64_t this_bit = 0;
                if (!all_raw && current[gpointer] == 0) { // Consume bit from RLE
                    if (rle_value == -1) {
                        rle_index++;
                        if (rle_index == int64_t(rle.size()))
                            break;
                        rle_value = rle[rle_index];
                    }
                    if (rle_value >= 1) {
                        read_from_rle++;
                        this_bit = 0;
                        rle_value--;
                    }
                    else if (rle_value == 0) {
                        read_from_rle++;
                        this_bit = 1;
                        rle_index++;
                        if (rle_index == int64_t(rle.size()))
                            break;
                        rle_value = rle[rle_index];
                    }
                }
                else { // Consume bit from raw
                    if (raw_index == int64_t(raw.size()))
                        break;
                    this_bit = raw[raw_index];
                    read_from_raw++;
                    raw_index++;
                }
                if (this_bit)
                    current[gpointer] |= this_bit << state.q;
            }
            times.unscramble_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - timenow).count()/1000.;
        }

        all_raw = read_bits(zs, 1);

        bool done = read_bits(zs, 1);
        if (done)
            break;
        else
            zeros++;
    }
    if (debug)
        std::cout << "decode_rle_time=" << times.decode_rle_time << ", decode_raw_time=" << times.decode_raw_time << ", unscramble_time=" << times.unscramble_time << std::endl;
    state.pointer = gpointer;
    return current;
}

static std::vector<double> dequantize(tthresh::zs& zs, std::vector<uint64_t>& current, const DecodeState& state) { // TODO after resize
    using namespace tthresh;

    size_t size = current.size();
    std::vector<double> c(size, 0);
    for (size_t i = 0; i < size; ++i) {
        if (current[i] > 0) {
            if (i < state.pointer) {
                if (state.q >= 1)
                    current[i] += 1ULL<<(state.q-1);
            }
            else
                current[i] += 1ULL<< state.q;
            char sign = read_bits(zs, 1);
            c[i] = double(current[i]) / state.maximum * (sign*2-1);
        }
    }
    return c;
}

std::vector<double> tthresh::decompress(std::istream& in,
    std::vector<uint32_t>& dimensions,
    bool verbose, bool debug)
{
    /*************/
    // Read shapes
    /*************/
    zs zs(&in);
    int n;
    read_stream(zs, reinterpret_cast<uint8_t*>(&n), sizeof(n));
    std::vector<uint32_t> s(n);
    read_stream(zs, reinterpret_cast<uint8_t*>(s.data()), sizeof(uint32_t)*n);
    dimensions = s;

    std::vector<size_t> sprod;
    cumulative_products(s, sprod);
    size_t size = sprod[n];

    if (verbose) {
        std::cout << std::endl << "/***** Decompression: " << n << "D tensor of size ";
        std::cout << s[0];
        for (uint8_t i = 1; i < n; ++i)
            std::cout << " x " << s[i];
        std::cout << " *****/" << std::endl << std::endl;
    }

    /*************/
    // Decode core
    /*************/
    DecodeState state {0};
    DecodeTimes times {0};
    std::vector<uint64_t> current = decode_array(zs, sprod[n], true, state, verbose, debug, times);
    std::vector<double> c = dequantize(zs, current, state);
    close_rbit(zs);

    /*******************/
    // Read tensor ranks
    /*******************/

    std::vector<uint32_t> r(n);
    read_stream(zs, reinterpret_cast<uint8_t*> (&r[0]), n * sizeof(r[0]));
    std::vector<size_t> rprod(n + 1);
    rprod[0] = 1;
    for (uint8_t i = 0; i < n; ++i)
        rprod[i + 1] = rprod[i] * r[i];
    if (verbose) {
        std::cout << "Compressed tensor ranks:";
        for (uint8_t i = 0; i < n; ++i)
            std::cout << " " << r[i];
        std::cout << std::endl;
    }

    std::vector<Eigen::RowVectorXd> slicenorms(n);
    for (uint8_t i = 0; i < n; ++i) {
        slicenorms[i] = Eigen::RowVectorXd(r[i]);
        for (uint64_t col = 0; col < r[i]; ++col) { // TODO faster
            double norm;
            read_stream(zs, reinterpret_cast<uint8_t*> (&norm), sizeof(double));
            slicenorms[i][col] = norm;
        }
    }

    //**********************/
    // Reshape core in place
    //**********************/

    size_t index = 0; // Where to read from in the original core
    std::vector<size_t> indices(n, 0);
    uint8_t pos = 0;
    for (size_t i = 0; i < rprod[n]; ++i) { // i marks where to write in the new rank-reduced core
        c[i] = c[index];
        indices[0]++;
        index++;
        pos = 0;
        // We update all necessary indices in cascade, left to right. pos == n-1 => i == rprod[n]-1 => we are done
        while (indices[pos] >= r[pos] && pos < n - 1) {
            indices[pos] = 0;
            index += sprod[pos + 1] - r[pos] * sprod[pos];
            pos++;
            indices[pos]++;
        }
    }

    //*****************/
    // Reweight factors
    //*****************/

    std::vector<Eigen::MatrixXd> Us;
    for (uint8_t i = 0; i < n; ++i) {
        std::vector<uint64_t> factorq = decode_array(zs, s[i] * r[i], false, state, verbose, debug, times);
        std::vector<double> factor = dequantize(zs, factorq, state);
        Eigen::MatrixXd Uweighted(s[i], r[i]);
        memcpy(Uweighted.data(), (void*)factor.data(), sizeof(double) * s[i] * r[i]);
        Eigen::MatrixXd U(s[i], r[i]);
        for (size_t col = 0; col < r[i]; ++col) {
            if (slicenorms[i][col] > 1e-10)
                U.col(col) = Uweighted.col(col) / slicenorms[i][col];
            else
                U.col(col) *= 0;
        }
        Us.push_back(U);
    }
    close_rbit(zs);

    /************************/
    // Reconstruct the tensor
    /************************/

    if (verbose)
        std::cout << "Reconstructing tensor..." << std::endl;
    std::vector<Slice> cutout;
    for (int i = 0; i < n; ++i) {
        cutout.push_back(Slice(0, -1, 1));
    }
    tthresh::hosvd_decompress(c, Us, r, rprod, sprod, verbose, cutout);

    return c;
}

