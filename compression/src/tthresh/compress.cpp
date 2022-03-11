/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */
#include "memtrace.h"
#include <tthresh/compress.h>

#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>

#include "encode.hpp"
#include "tucker.hpp"
#include "io.hpp"
#include "utils.hpp"

#ifdef _WIN32
typedef long double LLDOUBLE;
typedef long double LDOUBLE;
#else
#include <unistd.h>
typedef __float128 LLDOUBLE;
typedef __float80 LDOUBLE;
#endif


//int qneeded;
//
//double rle_time = 0;
//double raw_time = 0;
//
//double price = -1, total_bits_core = -1, eps_core = -1;
//size_t total_bits = 0;

struct EncodingStats
{
    int qneeded = 0;
    double price = -1, total_bits_core = -1, eps_core = -1;
    size_t total_bits = 0;
};

static std::vector<uint64_t> encode_array(
    tthresh::zs& zs, EncodingStats& stats, 
    const double* c, size_t size, double eps_target, bool is_core, bool verbose=false) {

    /**********************************************/
    // Compute and save maximum (in absolute value)
    /**********************************************/

    if (is_core && verbose)
        std::cout << "Preliminaries... " << std::endl;
    double maximum = 0;
    for (size_t i = 0; i < size; i++) {
        if (abs(c[i]) > maximum)
            maximum = abs(c[i]);
    }
    double scale = ldexp(1, 63-ilogb(maximum));

    uint64_t tmp;
    memcpy(&tmp, (void*)&scale, sizeof(scale)); //TODO: save maximum or scale?
    tthresh::write_bits(zs, tmp, 64);

    LLDOUBLE normsq = 0;
    std::vector<uint64_t> coreq(size);

    // 128-bit float arithmetics are slow, so we split the computation of normsq into partial sums
    size_t stepsize = 100;
    size_t nsteps = ceil(size/double(stepsize));
    size_t pos = 0;
    for (size_t i = 0; i < nsteps; ++i) {
        LDOUBLE partial_normsq = 0;
        for (size_t j = 0; j < stepsize; ++j) {
            coreq[pos] = uint64_t(abs(c[pos])*scale);
            partial_normsq += LDOUBLE(abs(c[pos]))*abs(c[pos]);
            pos++;
            if (pos == size)
                break;
        }
        normsq += partial_normsq;
        if (pos == size)
            break;
    }
    normsq *= LLDOUBLE(scale)*LLDOUBLE(scale);

    LLDOUBLE sse = normsq;
    LDOUBLE last_eps = 1;
    LDOUBLE thresh = eps_target*eps_target*normsq;

    /**************/
    // Encode array
    /**************/

    std::vector<uint64_t> current(size, 0);

    //if (is_core and verbose)
    //    stop_timer();
    bool done = false;
    stats.total_bits = 0;
    size_t last_total_bits = stats.total_bits;
    double eps_delta = 0, size_delta = 0, epsilon;
    int q;
    bool all_raw = false;
    if (verbose)
        std::cout << "Encoding core..." << std::endl;
    for (q = 63; q >= 0; --q) {
        if (verbose && is_core)
            std::cout << "Encoding core's bit plane p = " << q << std::flush;
        std::vector<uint64_t> rle;
        LDOUBLE plane_sse = 0;
        size_t plane_ones = 0;
        size_t counter = 0;
        size_t i;
        std::vector<bool> raw;
        for (i = 0; i < size; ++i) {
            bool current_bit = ((coreq[i]>>q)&1ULL);
            plane_ones += current_bit;
            if (!all_raw && current[i] == 0) { // Feed to RLE
                if (!current_bit)
                    counter++;
                else {
                    rle.push_back(counter);
                    counter = 0;
                }
            }
            else { // Feed to raw stream
                ++stats.total_bits;
                raw.push_back(current_bit);
            }

            if (current_bit) {
                plane_sse += (LDOUBLE(coreq[i] - current[i]));
                current[i] |= 1ULL<<q;
                if (plane_ones%100 == 0) {
                    LDOUBLE k = 1ULL<<q;
                    LDOUBLE sse_now = sse+(-2*k*plane_sse + k*k*plane_ones);
                    if (sse_now <= thresh) {
                        done = true;
                        if (verbose)
                            std::cout << " <- breakpoint: coefficient " << i << std::flush;
                        break;
                    }
                }

            }
        }
        if (verbose && is_core)
            std::cout << std::endl;

        LDOUBLE k = 1ULL<<q;
        sse += -2*k*plane_sse + k*k*plane_ones;
        rle.push_back(counter);

        uint64_t rawsize = raw.size();
        write_bits(zs, rawsize, 64);
        stats.total_bits += 64;

        {
            //high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
            for (size_t i = 0; i < raw.size(); ++i)
                write_bits(zs, raw[i], 1);
            //raw_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
        }
        {
            //high_resolution_clock::time_point timenow = chrono::high_resolution_clock::now();
            uint64_t this_part = tthresh::encode(zs, rle);
            //rle_time += std::chrono::duration_cast<std::chrono::microseconds>(chrono::high_resolution_clock::now() - timenow).count()/1000.;
            stats.total_bits += this_part;
        }

        epsilon = sqrt(double(sse/normsq));
        if (last_total_bits > 0) {
            if (is_core) {
                size_delta = (stats.total_bits - last_total_bits) / double(last_total_bits);
                eps_delta = (last_eps - epsilon) / epsilon;
            }
            else {
                if ((stats.total_bits/ stats.total_bits_core) / (epsilon/ stats.eps_core) >= stats.price)
                    done = true;
            }
        }
        last_total_bits = stats.total_bits;
        last_eps = epsilon;

        if (raw.size()/double(size) > 0.8)
            all_raw = true;

        write_bits(zs, all_raw, 1);
        stats.total_bits++;

        write_bits(zs, done, 1);
        stats.total_bits++;

        if (done)
            break;
    }
    //if (verbose)
    //    stop_timer();

    /****************************************/
    // Save signs of significant coefficients
    /****************************************/

    for (size_t i = 0; i < size; ++i) {
        if (current[i] > 0) {
            write_bits(zs, (c[i] > 0), 1);
            stats.total_bits++;
        }
    }

    if (is_core) {
        stats.price = size_delta / eps_delta;
        stats.eps_core = epsilon;
        stats.total_bits_core = stats.total_bits;
    }
    return current;
}



void tthresh::compress(
    std::ostream& out, const double* inputData, const std::vector<uint32_t>& dimensions,
    tthresh::Target target, double targetValue, bool verbose)
{
    const int n = dimensions.size();
    if (n == 0) {
        std::cerr << "Input array is zero-dimensional!" << std::endl;
        return;
    }
    size_t size = dimensions[0];
    for (int i = 1; i < n; ++i) size *= dimensions[i];
    if (size == 0) {
        std::cerr << "Input array is empty!" << std::endl;
        return;
    }

    std::vector<size_t> dimensionsProd;
    cumulative_products(dimensions, dimensionsProd);

    /********************************************/
    // Save tensor dimensionality and sizes
    /********************************************/
    out.write(reinterpret_cast <const char*> (&n), sizeof(n));
    out.write(reinterpret_cast <const char*> (dimensions.data()), n*sizeof(dimensions[0]));

    /*****************************/
    // Compute data norms
    /*****************************/

    double datamin = std::numeric_limits<double>::max(); // Tensor statistics
    double datamax = std::numeric_limits<double>::min();
    double datanorm = 0;
    for (size_t i = 0; i < size; ++i) {
        const auto data = inputData[i];
        datamin = std::min(datamin, data); // Compute statistics, since we're at it
        datamax = std::max(datamax, data);
        datanorm += data * data;
    }
    datanorm = sqrt(datanorm);
    if (verbose)
        std::cout << "Input statistics: min = " << datamin << ", max = " << datamax << ", norm = " << datanorm << std::endl;

    /**********************************************************************/
    // Compute the target SSE (sum of squared errors) from the given metric
    /**********************************************************************/

    double sse;
    if (target == compression::EPS)
        sse = pow(targetValue * datanorm, 2);
    else if (target == compression::RMSE)
        sse = pow(targetValue, 2) * size;
    else //PSNR
        sse = pow((datamax - datamin) / (2 * (pow(10, targetValue / 20))), 2) * size;
    double epsilon = sqrt(sse) / datanorm;
    if (verbose) {
        double rmse = sqrt(sse / size);
        double psnr = 20 * log10((datamax - datamin) / (2 * rmse));
        std::cout << "We target eps = " << epsilon << ", rmse = " << rmse << ", psnr = " << psnr << std::endl;
    }

    /*********************************/
    // Create and decompose the tensor
    /*********************************/

    if (verbose)
        std::cout << "Tucker decomposition..." << std::endl;
    std::unique_ptr<double[]> c = std::make_unique<double[]>(size);

    memcpy(c.get(), inputData, size * sizeof(double));

    std::vector<Eigen::MatrixXd> Us(n); // Tucker factor matrices
    tthresh::hosvd_compress(c.get(), Us, dimensions, dimensionsProd, verbose);

//    if (verbose) {
//        stop_timer();
////        cout << "RLE time (ms):" << rle_time << endl;
////        cout << "Raw time (ms):" << raw_time << endl;
//    }

    /**************************/
    // Encode and save the core
    /**************************/

    zs zs(&out);
    open_wbit(zs);
    EncodingStats encodingStats;
    std::vector<uint64_t> current = encode_array(zs, encodingStats, c.get(), size, epsilon, true, verbose);
    close_wbit(zs);

    /*******************************/
    // Compute and save tensor ranks
    /*******************************/

    if (verbose)
        std::cout << "Computing ranks... " << std::endl;;
    std::vector<uint32_t> r(n, 0);
    std::vector<size_t> indices(n, 0);
    std::vector<Eigen::RowVectorXd > slicenorms(n);
    for (int dim = 0; dim < n; ++dim) {
        slicenorms[dim] = Eigen::RowVectorXd(dimensions[dim]);
        slicenorms[dim].setZero();
    }
    for (size_t i = 0; i < size; ++i) {
        if (current[i] > 0) {
            for (int dim = 0; dim < n; ++dim) {
                slicenorms[dim][indices[dim]] += double(current[i])*current[i];
            }
        }
        indices[0]++;
        int pos = 0;
        while (indices[pos] >= dimensions[pos] && pos < n-1) {
            indices[pos] = 0;
            pos++;
            indices[pos]++;
        }
    }

    for (int dim = 0; dim < n; ++dim) {
        for (size_t i = 0; i < dimensions[dim]; ++i) {
            if (slicenorms[dim][i] > 0)
                r[dim] = i+1;
            slicenorms[dim][i] = sqrt(slicenorms[dim][i]);
        }
    }
    //if (verbose)
    //    stop_timer();

    if (verbose) {
        std::cout << "Compressed tensor ranks:";
        for (uint8_t i = 0; i < n; ++i)
            std::cout << " " << r[i];
        std::cout << std::endl;
    }
    write_stream(zs, reinterpret_cast<unsigned char*> (&r[0]), n*sizeof(r[0]));

    for (uint8_t i = 0; i < n; ++i) {
        write_stream(zs, reinterpret_cast<uint8_t*> (slicenorms[i].data()), r[i]*sizeof(double));
    }

    std::vector<Eigen::MatrixXd> Uweighteds;
    open_wbit(zs);
    for (int dim = 0; dim < n; ++dim) {
        Eigen::MatrixXd Uweighted = Us[dim].leftCols(r[dim]);
        for (size_t col = 0; col < r[dim]; ++col)
            Uweighted.col(col) = Uweighted.col(col)*slicenorms[dim][col];
        Uweighteds.push_back(Uweighted);
        encode_array(zs, encodingStats, Uweighted.data(), dimensions[dim]*r[dim], 0, false);//*(s[i]*s[i]/sprod[n]));  // TODO flatten in F order?
    }
    close_wbit(zs);
    c.reset();
    size_t newbits = zs.total_written_bytes * 8;
    if (verbose) {
        constexpr int io_type_size = sizeof(double);
        std::cout << "oldbits = " << size * io_type_size * 8L << ", newbits = " << newbits << ", compressionratio = " << size * io_type_size * 8L / double(newbits)
            << ", bpv = " << newbits / double(size) << std::endl << std::flush;
    }
}

