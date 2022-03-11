/*
 * Copyright (c) 2016-2017, Rafael Ballester-Ripoll
 *                          (Visualization and MultiMedia Lab, University of Zurich),
 *                          rballester@ifi.uzh.ch
 *
 * Licensed under the LGPLv3.0 (https://github.com/rballester/tthresh/blob/master/LICENSE)
 */

#ifndef __IO_HPP__
#define __IO_HPP__

#include "memtrace.h"
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <ostream>
//#include "zlib.h"

// Avoids corruption of the input and output data on Windows/MS-DOS systems
#if defined(MSDOS) || defined(OS2) || defined(WIN32) || defined(__CYGWIN__)
#  include <fcntl.h>
#  include <io.h>
#  define SET_BINARY_MODE(file) setmode(fileno(file), O_BINARY)
#else
#  define SET_BINARY_MODE(file)
#endif

namespace tthresh {

    struct zs {
        uint64_t rbytes, wbytes;
        int rbit, wbit;
        std::ostream* fileOut;
        std::istream* fileIn;
        //FILE *file; // File handle to read/write from/to

        //uint8_t inout[CHUNK]; // Buffer to write the results of inflation/deflation
        //uint8_t buf[CHUNK]; // Buffer used for the read/write operations
        //int32_t bufstart = 0;
        //int32_t bufend = 0;
        size_t total_written_bytes = 0; // Used to compute the final file size

        //open for writing
        zs(std::ostream* file)
            : fileOut(file), fileIn(nullptr), rbytes(0), wbytes(0), rbit(0), wbit(0)
        {}

        //open for reading
        zs(std::istream* file)
            : fileIn(file), fileOut(nullptr), rbytes(0), rbit(-1), wbytes(0), wbit(0)
        {}
    }; // Read/write state for zlib interfacing

    /*********/
    // Writing
    /*********/

    // Call open_wbit() before write_bits()
    // If write_bits() has been called, call close_wbit() before write_stream()

    //void open_write(string output_file) {
    //    SET_BINARY_MODE(output_file.c_str());
    //    zs.file = fopen(output_file.c_str(), "w");
    //}

    inline void write_stream(zs& zs, unsigned char* buf, size_t bytes_to_write)
    {
        zs.fileOut->write(reinterpret_cast<const char*>(buf), bytes_to_write);
        zs.total_written_bytes += bytes_to_write;
    }

    inline void open_wbit(zs& zs) {
        zs.wbytes = 0;
        zs.wbit = 63;
    }

    // Assumption: to_write <= 64
    inline void write_bits(zs& zs, uint64_t bits, int to_write) {
        assert(to_write > 0 && to_write <= 64);
        if (to_write <= zs.wbit + 1) {
            zs.wbytes |= bits << (zs.wbit + 1 - to_write);
            zs.wbit -= to_write;
        }
        else {
            if (zs.wbit > -1)
                zs.wbytes |= bits >> (to_write - (zs.wbit + 1));
            write_stream(zs, reinterpret_cast<unsigned char*> (&zs.wbytes), sizeof(zs.wbytes));
            to_write -= zs.wbit + 1;
            zs.wbytes = 0;
            zs.wbytes |= bits << (64 - to_write);
            zs.wbit = 63 - to_write;
        }
    }

    inline void close_wbit(zs& zs) {
        // Write any reamining bits
        if (zs.wbit < 63)
            write_stream(zs, reinterpret_cast <unsigned char*> (&zs.wbytes), sizeof(zs.wbytes));
    }

    //void close_write() {
    //    fclose(zs.file);
    //}

    /*********/
    // Reading
    /*********/

    // If read_bits() has been called, call close_rbit() before read_stream()

    inline void read_stream(zs& zs, uint8_t* buf, size_t bytes_to_read)
    {
        zs.fileIn->read(reinterpret_cast<char*>(buf), bytes_to_read);
    }

    inline void close_rbit(zs& zs)
    {
        zs.rbytes = 0;
        zs.rbit = -1;
    }

    // Assumption: to_read <= BITS
    inline uint64_t read_bits(zs& zs, char to_read) {
        uint64_t result = 0;
        if (to_read <= zs.rbit + 1) {
            result = zs.rbytes << (63 - zs.rbit) >> (64 - to_read);
            zs.rbit -= to_read;
        }
        else {
            if (zs.rbit > -1)
                result = zs.rbytes << (64 - zs.rbit - 1) >> (64 - to_read);
            read_stream(zs, reinterpret_cast<uint8_t*> (&zs.rbytes), sizeof(zs.rbytes));
            to_read -= zs.rbit + 1;
            result |= zs.rbytes >> (64 - to_read);
            zs.rbit = 63 - to_read;
        }
        return result;
    }

}

#endif // IO_HPP
