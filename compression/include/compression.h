#pragma once

#include <cassert>
#include <vector>
#include <memory>
#include <unordered_map>
#include <cuda_runtime.h>

namespace compression
{
    template<typename TOut, typename TIn>
    void cast(TOut* dst, const TIn* src, ptrdiff_t numel)
    {
#pragma omp parallel for
        for (ptrdiff_t i = 0; i < numel; ++i)
            dst[i] = static_cast<TOut>(src[i]);
    }

    template<typename TOut, typename TIn>
    std::vector<TOut> cast(const std::vector<TIn>& v)
    {
        const size_t n = v.size();
        std::vector<TOut> o(n);
        cast<TOut, TIn>(o.data(), v.data(), n);
        return o;
    }

    /**
     * Specifies a slice through an axis (python slice notation)
     */
    struct Slice
    {
        ptrdiff_t start, end, step;
        Slice(ptrdiff_t start=0, ptrdiff_t end=-1, ptrdiff_t step=1)
            : start(start), end(end), step(step)
        {}
        /**
         * Realizes this slice with the actual size of that dimension.
         * End-indices that start from the back (i.e. negative numbers)
         * are converted to their actual index.
         * Example:
         * end=-1, s=42 -> end=42
         * end=-2, s=42 -> end=41
         */
        void realize(ptrdiff_t s)
        {
            if (end < 0) end = s + end + 1;
        }
        ptrdiff_t numel() const
        {
            return (end - start) / step;
        }
        ptrdiff_t at(ptrdiff_t i) const
        {
            return start + i * step;
        }
    };

    /**
     * Raw volume in c-style order (last dimension is fastest)
     */
    template<typename T, bool Fortran>
    class RawVolume
    {
    private:
        std::vector<T> memory_;
        std::vector<size_t> dimensions_;
        std::vector<size_t> strides_;
        enum {
            IsFortran = Fortran, //Fortran-order
            IsC = !Fortran //C-order
        };

    public:
        RawVolume(RawVolume const&) = delete;
        RawVolume(RawVolume&&) = delete;
        RawVolume& operator=(RawVolume const&) = delete;
        RawVolume& operator=(RawVolume&&) = delete;

        static size_t prod(const std::vector<size_t>& d)
        {
            if (d.empty()) return 0;
            size_t v = d[0];
            for (size_t i = 1; i < d.size(); ++i)
                v *= d[i];
            return v;
        }
        static std::vector<size_t> computeStrides(const std::vector<size_t>& d, std::bool_constant<false> cOrder)
        {
            //c-style
            std::vector<size_t> strides(d.size(), 1);
            for (int i=d.size()-2; i>=0; --i)
            {
                strides[i] = strides[i + 1] * d[i + 1];
            }
            return strides;
        }
        static std::vector<size_t> computeStrides(const std::vector<size_t>& d, std::bool_constant<true> fortranOrder)
        {
            //fortran-style
            std::vector<size_t> strides(d.size(), 1);
            for (int i = 1; i < d.size(); ++i)
            {
                strides[i] = strides[i - 1] * d[i - 1];
            }
            return strides;
        }

        RawVolume(const std::vector<size_t>& dimensions)
            : memory_(prod(dimensions)), dimensions_(dimensions)
            , strides_(computeStrides(dimensions, std::bool_constant<Fortran>()))
        {}
        RawVolume(const std::vector<size_t>& dimensions, const std::vector<size_t>& strides)
            : memory_(prod(dimensions)), dimensions_(dimensions)
            , strides_(strides)
        {}
        RawVolume(std::vector<T>&& memory, const std::vector<size_t>& dimensions)
            : memory_(memory), dimensions_(dimensions)
            , strides_(computeStrides(dimensions, std::bool_constant<Fortran>()))
        {
            assert(memory_.size() == prod(dimensions));
        }

        [[nodiscard]] constexpr bool isCOrder() const { return IsC; }
        [[nodiscard]] constexpr bool isFortranOrder() const { return IsFortran; }

        [[nodiscard]] size_t numel() const { return memory_.size(); }
        [[nodiscard]] const std::vector<size_t>& dimensions() const
        {
            return dimensions_;
        }
        [[nodiscard]] const std::vector<size_t>& strides() const
        {
            return strides_;
        }
        [[nodiscard]] T* data() { return memory_.data(); }
        [[nodiscard]] const T* data() const { return memory_.data(); }


        template<typename TOut>
        [[nodiscard]] std::shared_ptr<RawVolume<TOut, Fortran>> cast() const
        {
            auto o = std::make_shared<RawVolume<TOut, Fortran>>(dimensions(), strides());
            compression::cast<TOut, T>(o->data(), data(), numel());
            return o;
        }

    private:
        template<bool F>
        void fillDimension(std::shared_ptr<RawVolume<T, F>> out, const std::vector<Slice>& slices, int dim, ptrdiff_t offsetIn, ptrdiff_t offsetOut) const
        {
            if (dim==dimensions_.size())
            {
                //recursion end
                out->data()[offsetOut] = memory_[offsetIn];
                return;
            }

            const Slice& slice = slices[dim];
            ptrdiff_t strideIn = strides_[dim];
            ptrdiff_t strideOut = out->strides()[dim];
            auto numel = slice.numel();
            for (size_t iOut=0; iOut<numel; ++iOut)
            {
                auto iIn = slice.at(iOut);
                fillDimension(out, slices, dim + 1, offsetIn + strideIn * iIn, offsetOut + strideOut * iOut);
            }
        }
    public:
        template<bool FortranOut = Fortran>
        [[nodiscard]] std::shared_ptr<RawVolume<T, FortranOut>> slice(const std::vector<Slice>& slices) const
        {
            //assemble slices
            std::vector<Slice> slices2(dimensions_.size());
            std::vector<size_t> newdim(dimensions_.size());
            for (int i=0; i<dimensions_.size(); ++i)
            {
                Slice s = i < slices.size() ? slices[i] : Slice();
                s.realize(dimensions_[i]);
                slices2[i] = s;
                assert(s.numel() > 0);
                newdim[i] = s.numel();
            }

            //create and fill data
            auto out = std::make_shared<RawVolume<T, FortranOut>>(newdim);
            fillDimension(out, slices2, 0, 0, 0);
            return out;
        }

        [[nodiscard]] std::shared_ptr<RawVolume<T, false>> toCStyle() const
        {
            return slice<false>({});
        }
        [[nodiscard]] std::shared_ptr<RawVolume<T, true>> toFortranStyle() const
        {
            return slice<true>({});
        }
    };
    template<typename T>
    using RawVolumeCStyle_ptr = std::shared_ptr<RawVolume<T, false>>;
    template<typename T>
    using RawVolumeFortranStyle_ptr = std::shared_ptr<RawVolume<T, true>>;

    class CompressedVolume
    {
    private:
        bool owned_;
        void* memory_;
        size_t size_;

    public:
        CompressedVolume(CompressedVolume const&) = delete;
        CompressedVolume(CompressedVolume&&) = delete;
        CompressedVolume& operator=(CompressedVolume const&) = delete;
        CompressedVolume& operator=(CompressedVolume&&) = delete;

        CompressedVolume(size_t size)
            : owned_(true), memory_(malloc(size)), size_(size)
        {}
        CompressedVolume(void* memory, size_t size)
            : owned_(false), memory_(memory), size_(size)
        {}
        ~CompressedVolume()
        {
            if (owned_)
                free(memory_);
        }

        /**
         * The size in bytes
         */
        [[nodiscard]] size_t size() const { return size_; }
        /**
         * The raw pointer to the data
         */
        [[nodiscard]] const void* data() const { return memory_; }
        /**
         * The raw pointer to the data (non-const)
         */
        [[nodiscard]] void* data() { return memory_; }
    };
    typedef std::shared_ptr<CompressedVolume> CompressedVolume_ptr;

    /**
     * The metric for computing the target compression rate
     */
    enum TThreshTarget
    {
        EPS, //relative error
        RMSE, //
        PSNR //
    };

    /**
     * Knows keys, provided by all compression and decompression methods below
     * - "time_ms" : execution time in milliseconds
     * - "total_memory_cpu" : The total cumulative memory that is allocated on the CPU throughout the algorithm in bytes
     * - "peak_memory_cpu" : The peak memory usage on the CPU in bytes. It holds peak_memory_cpu <= total_memory_cpu
     * - "total_memory_gpu" : The total cumulative memory that is allocated on the GPU throughout the algorithm in bytes
     * - "peak_memory_gpu" : The peak memory usage on the GPU in bytes. It holds peak_memory_gpu <= total_memory_gpu
     */
    typedef std::unordered_map<std::string, long long> Statistics_t;

    std::tuple<CompressedVolume_ptr, Statistics_t> compressTThresh(RawVolumeFortranStyle_ptr<double> volume, TThreshTarget target, double targetValue, bool verbose);

    std::tuple<RawVolumeFortranStyle_ptr<double>, Statistics_t> decompressTThresh(CompressedVolume_ptr v, bool verbose);

    std::tuple<CompressedVolume_ptr, Statistics_t> compressTThreshChunked(const std::vector<RawVolumeFortranStyle_ptr<double>>& volume, TThreshTarget target, double targetValue, bool verbose);

    std::tuple<std::vector<RawVolumeFortranStyle_ptr<double>>, Statistics_t> decompressTThreshChunked(CompressedVolume_ptr v, bool verbose);

    std::tuple<CompressedVolume_ptr, Statistics_t> compressCUDA(RawVolumeCStyle_ptr<float> volume, int numLevels, float quantizationStep, bool verbose, int numChunks=1);

    std::tuple<RawVolumeCStyle_ptr<float>, Statistics_t> decompressCUDA(CompressedVolume_ptr v, bool verbose);

    //Compression using manual chunking. Each chunk must have the same size
    std::tuple<CompressedVolume_ptr, Statistics_t> compressCUDAChunked(const std::vector<RawVolumeCStyle_ptr<float>>& volumes, int numLevels, float quantizationStep, bool verbose);

    std::tuple<std::vector<RawVolumeCStyle_ptr<float>>, Statistics_t> decompressCUDAChunked(CompressedVolume_ptr v, bool verbose);

    /**
     * Interactive decompressed of chunked cudaCompress.
     * The original volume is compressed with \ref compressCUDAChunked
     * and can be decompressed chunk-by-chunk here.
     */
    class CudaCompressInteractiveDecompression
    {
        struct impl;
        std::unique_ptr<impl> pImpl;

    public:
        CudaCompressInteractiveDecompression(CompressedVolume_ptr v);
        ~CudaCompressInteractiveDecompression();

        [[nodiscard]] int chunkWidth() const;
        [[nodiscard]] int chunkHeight() const;
        [[nodiscard]] int chunkDepth() const;
        [[nodiscard]] int numChunks() const;

        enum class DataType
        {
            TypeUChar,
            TypeUShort,
            TypeFloat,
            _TypeCount_
        };
        Statistics_t decompress(int chunk, cudaSurfaceObject_t target, DataType targetDtype);

        //The accumulated statistics over all decompression calls
        Statistics_t globalStatistics();

        CudaCompressInteractiveDecompression(const CudaCompressInteractiveDecompression& other) = delete;
        CudaCompressInteractiveDecompression(CudaCompressInteractiveDecompression&& other) noexcept = delete;
        CudaCompressInteractiveDecompression& operator=(const CudaCompressInteractiveDecompression& other) = delete;
        CudaCompressInteractiveDecompression& operator=(CudaCompressInteractiveDecompression&& other) noexcept = delete;
    };
    typedef std::shared_ptr<CudaCompressInteractiveDecompression> CudaCompressInteractiveDecompression_ptr;
}