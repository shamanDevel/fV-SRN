#include <torch/extension.h>
#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <tinyformat.h>
#include <cuMat/src/Context.h>
#include <glm/gtx/string_cast.hpp>
#include <third-party/Eigen/Core> // in cuMat

#include <helper_math.cuh>
#include <kernel_loader.h>
#include <module_registry.h>
#include <opengl_utils.h>
#include <compression.h>
#include <volume.h>

#ifdef WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <Windows.h>
#endif

#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME pyrenderer
#endif

namespace py = pybind11;
using namespace renderer;

#include <cmrc/cmrc.hpp>
CMRC_DECLARE(kernels);

static void staticCudaSourcesLoaderRec(
    std::vector<renderer::KernelLoader::NameAndContent>& fileList,
    const cmrc::directory_entry& e, const cmrc::embedded_filesystem& fs,
    const std::string& currentPath)
{
    if (e.is_file())
    {
        //std::cout << "Load file " << e.filename() << std::endl;
        auto f = fs.open(currentPath + e.filename());
        std::string content(f.size(), '\0');
        memcpy(content.data(), f.begin(), f.size());
        fileList.push_back({ e.filename(), content });
    } else
    {
        //std::cout << "Walk directory " << currentPath << std::endl;
        for (const auto& e2 : fs.iterate_directory(currentPath + e.filename()))
            staticCudaSourcesLoaderRec(fileList, e2, fs, currentPath + e.filename() + "/");
    }
}
static void staticCudaSourcesLoader(
    std::vector<renderer::KernelLoader::NameAndContent>& fileList)
{
    cmrc::embedded_filesystem fs = cmrc::kernels::get_filesystem();
    for (const auto& e : fs.iterate_directory(""))
        staticCudaSourcesLoaderRec(fileList, e, fs, "");
}

std::filesystem::path getCacheDir()
{
    //suffix and default (if default, it is a relative path)
    static const std::filesystem::path SUFFIX{ "kernel_cache" };
#ifdef WIN32
    //get the path to this dll as base path
    char path[MAX_PATH];
    HMODULE hm = NULL;

    if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)&getCacheDir, &hm) == 0)
    {
        int ret = GetLastError();
        fprintf(stderr, "GetModuleHandle failed, error = %d\n", ret);
        return SUFFIX;
    }
    if (GetModuleFileName(hm, path, sizeof(path)) == 0)
    {
        int ret = GetLastError();
        fprintf(stderr, "GetModuleFileName failed, error = %d\n", ret);
        return SUFFIX;
    }

    std::filesystem::path out = path;
    out = out.parent_path();
    const auto out_str = out.string();
    fprintf(stdout, "This DLL is located at %s, use that as cache dir\n", out_str.c_str());
    out /= SUFFIX;
    return out;
    
#else
    return SUFFIX; //default
#endif
}

class GPUTimer
{
    cudaEvent_t start_, stop_;
public:
    GPUTimer()
        : start_(0), stop_(0)
    {
        CUMAT_SAFE_CALL(cudaEventCreate(&start_));
        CUMAT_SAFE_CALL(cudaEventCreate(&stop_));
    }
    ~GPUTimer()
    {
        CUMAT_SAFE_CALL(cudaEventDestroy(start_));
        CUMAT_SAFE_CALL(cudaEventDestroy(stop_));
    }
    void start()
    {
        CUMAT_SAFE_CALL(cudaEventRecord(start_));
    }
    void stop()
    {
        CUMAT_SAFE_CALL(cudaEventRecord(stop_));
    }
    float getElapsedMilliseconds()
    {
        CUMAT_SAFE_CALL(cudaEventSynchronize(stop_));
        float ms;
        CUMAT_SAFE_CALL(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

void bindCompressionUtils(pybind11::module_& m);

//BINDINGS
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // initialize context
#if RENDERER_OPENGL_SUPPORT==1
    try {
        OffscreenContext::setup();
    } catch (const std::exception& ex)
    {
        std::cerr << "Unable to initialize OpenGL, attempting to use rasterization will crash later\n" << ex.what() << std::endl;
    }
#endif
    KernelLoader::Instance().initCuda();
    KernelLoader::Instance().setCudaCacheDir(getCacheDir());
    KernelLoader::Instance().setCustomCudaSourcesLoader(staticCudaSourcesLoader);
    cuMat::Context& ctx = cuMat::Context::current();
    ((void)ctx);
    
    m.doc() = "python bindings for the differentiable volume renderer";
    
    m.def("set_cuda_cache_dir", [](const std::string& path)
    {
            KernelLoader::Instance().setCudaCacheDir(path);
    });
    m.def("set_kernel_cache_file", [](const std::string& filename)
        {
            KernelLoader::Instance().setKernelCacheFile(filename);
        }, py::doc("Explicitly sets the kernel cache file."));

    auto cleanup_callback = []() {
        KernelLoader::Instance().cleanup();
#if RENDERER_OPENGL_SUPPORT==1
        OffscreenContext::teardown();
#endif
    };
    m.def("cleanup", cleanup_callback, py::doc("Explicit cleanup of all CUDA references"));
    m.add_object("_cleanup", py::capsule(cleanup_callback));
    

    py::class_<float3>(m, "float3")
        .def(py::init<>())
        .def(py::init([](float x, float y, float z) {return make_float3(x, y, z); }))
        .def_readwrite("x", &float3::x)
        .def_readwrite("y", &float3::y)
        .def_readwrite("z", &float3::z)
        .def("__str__", [](const float3& v)
    {
        return tinyformat::format("(%f, %f, %f)", v.x, v.y, v.z);
    })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(float()* py::self)
        .def(py::self* float())
        ;
    
    py::class_<float4>(m, "float4")
        .def(py::init<>())
        .def(py::init([](float x, float y, float z, float w)
    {
        return make_float4(x, y, z, w);
    }))
        .def_readwrite("x", &float4::x)
        .def_readwrite("y", &float4::y)
        .def_readwrite("z", &float4::z)
        .def_readwrite("w", &float4::w)
        .def("__str__", [](const float4& v)
    {
        return tinyformat::format("(%f, %f, %f, %f)", v.x, v.y, v.z, v.w);
    })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(float() * py::self)
        .def(py::self * float())
        ;

    py::class_<double3>(m, "double3")
        .def(py::init<>())
        .def(py::init([](double x, double y, double z) {return make_double3(x, y, z); }))
        .def_readwrite("x", &double3::x)
        .def_readwrite("y", &double3::y)
        .def_readwrite("z", &double3::z)
        .def("__str__", [](const double3& v)
    {
        return tinyformat::format("(%f, %f, %f)", v.x, v.y, v.z);
    })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(double() * py::self)
        .def(py::self * double())
        ;

    py::class_<double4>(m, "double4")
        .def(py::init<>())
        .def(py::init([](double x, double y, double z, double w)
            {
                return make_double4(x, y, z, w);
            }))
        .def_readwrite("x", &double4::x)
        .def_readwrite("y", &double4::y)
        .def_readwrite("z", &double4::z)
        .def_readwrite("w", &double4::w)
        .def("__str__", [](const double4& v)
    {
        return tinyformat::format("(%f, %f, %f, %f)", v.x, v.y, v.z, v.w);
    })
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(double()* py::self)
        .def(py::self* double())
        ;

    py::class_<int2>(m, "int2")
        .def(py::init<>())
        .def(py::init([](int x, int y) {return make_int2(x, y); }))
        .def_readwrite("x", &int2::x)
        .def_readwrite("y", &int2::y)
        .def("__str__", [](const int2& v)
    {
        return tinyformat::format("(%d, %d)", v.x, v.y);
    });
    
    py::class_<int3>(m, "int3")
        .def(py::init<>())
        .def(py::init([](int x, int y, int z) {return make_int3(x, y, z); }))
        .def_readwrite("x", &int3::x)
        .def_readwrite("y", &int3::y)
        .def_readwrite("z", &int3::z)
        .def("__str__", [](const int3& v)
    {
        return tinyformat::format("(%d, %d, %d)", v.x, v.y, v.z);
    });

    py::class_<GPUTimer>(m, "GPUTimer")
        .def(py::init<>())
        .def("start", &GPUTimer::start)
        .def("stop", &GPUTimer::stop)
        .def("elapsed_milliseconds", &GPUTimer::getElapsedMilliseconds)
        ;

    m.def("sync", []() {CUMAT_SAFE_CALL(cudaDeviceSynchronize()); });
    
    ModuleRegistry::Instance().populateModules();
    ModuleRegistry::Instance().registerPybindModules(m);

    /*
     * Compression utils for comparisons
     */
    bindCompressionUtils(m);
}

void bindCompressionUtils(pybind11::module_& m)
{
    using namespace compression;
    auto comp = m.def_submodule("compression");

    const auto torch2rawFloat = [](const torch::Tensor& t)
    {
        TORCH_CHECK(t.device()==at::kCPU, "Tensor required to reside on the CPU");
        TORCH_CHECK(t.dim() == 3, "3D Tensor required, but got shape ", t.sizes());
        TORCH_CHECK(t.dtype() == c10::kFloat, "Float tensor required, but got ", t.dtype());
        TORCH_CHECK(t.is_contiguous(c10::MemoryFormat::Contiguous), "Tensor required to be in c-contiguous format");

        std::vector<size_t> sizes(t.dim());
        for (int i = 0; i < t.dim(); ++i) sizes[i] = t.size(i);
        auto vol = std::make_shared<RawVolume<float, false>>(sizes);
        memcpy(vol->data(), t.data_ptr<float>(), sizeof(float) * t.numel());
        return vol;
    };
    const auto torch2rawDouble = [](const torch::Tensor& t)
    {
        TORCH_CHECK(t.device()==at::kCPU, "Tensor required to reside on the CPU");
        //TORCH_CHECK(t.dim() == 3, "3D Tensor required, but got shape ", t.sizes());
        TORCH_CHECK(t.dtype() == c10::kDouble, "Double tensor required, but got ", t.dtype());
        TORCH_CHECK(t.is_contiguous(c10::MemoryFormat::Contiguous), "Tensor required to be in c-contiguous format");

        std::vector<size_t> sizes(t.dim());
        for (int i = 0; i < t.dim(); ++i) sizes[i] = t.size(i);
        auto vol = std::make_shared<RawVolume<double, false>>(sizes);
        memcpy(vol->data(), t.data_ptr<double>(), sizeof(double) * t.numel());
        return vol->toFortranStyle();
    };

    const auto rawFloat2torch = [](RawVolumeCStyle_ptr<float> v)
    {
        std::vector<int64_t> dim = cast<int64_t>(v->dimensions());
        auto t = torch::empty(
            c10::IntArrayRef(dim.data(), dim.size()), 
            c10::TensorOptions().dtype(c10::kFloat).memory_format(c10::MemoryFormat::Contiguous));
        memcpy(t.data_ptr<float>(), v->data(), sizeof(float) * t.numel());
        return t;
    };
    const auto rawDouble2torch = [](RawVolumeFortranStyle_ptr<double> v)
    {
        auto vc = v->toCStyle();
        std::vector<int64_t> dim = cast<int64_t>(v->dimensions());

        auto t = torch::empty(
            c10::IntArrayRef(dim.data(), dim.size()),
            c10::TensorOptions().dtype(c10::kDouble).memory_format(c10::MemoryFormat::Contiguous));
        memcpy(t.data_ptr<double>(), vc->data(), sizeof(double) * t.numel());

        return t;
    };

    const auto compressed2python = [](CompressedVolume_ptr c)
    {
        return py::bytes(static_cast<const char*>(c->data()), c->size());
    };
    const auto python2compressed = [](const py::bytes& b)
    {
        char* buffer;
        ssize_t length;
        if (PYBIND11_BYTES_AS_STRING_AND_SIZE(b.ptr(), &buffer, &length))
            throw std::runtime_error("Unable to extract bytes contents!");
        CompressedVolume_ptr p = std::make_shared<CompressedVolume>(buffer, length);
        return p;
    };

    comp.def("compress_tthresh",
        [&](const torch::Tensor& t, const std::string& target, double targetValue, bool verbose)
        {
            TThreshTarget tt;
            if (target == "EPS") tt = TThreshTarget::EPS;
            else if (target == "RMSE") tt = TThreshTarget::RMSE;
            else if (target == "PSNR") tt = TThreshTarget::PSNR;
            else throw std::runtime_error("Unknown compression target");

            const auto v1 = torch2rawDouble(t);
            const auto [v2, stats] = compressTThresh(v1, tt, targetValue, verbose);
            return std::make_tuple(compressed2python(v2), stats);
        }, py::arg("tensor"), py::arg("target"), py::arg("target_value"), py::arg("verbose"), py::doc(R"doc(
Volume compression using TThresh.

The returned tuple's second entries contains a map with statistics of the form string->int.
Known keys:
 - "time_ms" : execution time in milliseconds
 - "total_memory_cpu" : The total cumulative memory that is allocated on the CPU throughout the algorithm in bytes
 - "peak_memory_cpu" : The peak memory usage on the CPU in bytes. It holds peak_memory_cpu <= total_memory_cpu
 - "total_memory_gpu" : The total cumulative memory that is allocated on the GPU throughout the algorithm in bytes
 - "peak_memory_gpu" : The peak memory usage on the GPU in bytes. It holds peak_memory_gpu <= total_memory_gpu

:param tensor: the input 3D-double tensor
:param target: the target metric, either 'EPS', 'RMSE' or 'PSNR'
:param target_value: the value of the metric that should be achieved
:return: tuple (the compressed tensor, statistics)
            )doc"));

    comp.def("decompress_tthresh",
        [&](const py::bytes& b, bool verbose)
        {
            const auto v1 = python2compressed(b);
            const auto [v2, stats] = decompressTThresh(v1, verbose);
            return std::make_tuple(rawDouble2torch(v2), stats);
        }, py::arg("bytes"), py::arg("verbose"), py::doc(R"doc(
Volume decompression using TThresh.

The returned tuple's second entries contains a map with statistics of the form string->int.
Known keys:
 - "time_ms" : execution time in milliseconds
 - "total_memory_cpu" : The total cumulative memory that is allocated on the CPU throughout the algorithm in bytes
 - "peak_memory_cpu" : The peak memory usage on the CPU in bytes. It holds peak_memory_cpu <= total_memory_cpu
 - "total_memory_gpu" : The total cumulative memory that is allocated on the GPU throughout the algorithm in bytes
 - "peak_memory_gpu" : The peak memory usage on the GPU in bytes. It holds peak_memory_gpu <= total_memory_gpu

:param bytes: the bytes sequence containing the compressed volume
:return: tuple (the decompressed double tensor, statistics)
            )doc"));

    comp.def("compress_tthresh_chunked",
        [&](const std::vector<torch::Tensor>& tx, const std::string& target, double targetValue, bool verbose)
        {
            TThreshTarget tt;
            if (target == "EPS") tt = TThreshTarget::EPS;
            else if (target == "RMSE") tt = TThreshTarget::RMSE;
            else if (target == "PSNR") tt = TThreshTarget::PSNR;
            else throw std::runtime_error("Unknown compression target");

            std::vector<RawVolumeFortranStyle_ptr<double>> t2(tx.size());
            for (size_t i = 0; i < tx.size(); ++i) {
                t2[i] = torch2rawDouble(tx[i]);
            }
            const auto [v2, stats] = compressTThreshChunked(t2, tt, targetValue, verbose);
            return std::make_tuple(compressed2python(v2), stats);
        }, py::arg("tensors"), py::arg("target"), py::arg("target_value"), py::arg("verbose"), py::doc(R"doc(
Volume compression using TThresh with explicit chunking.

The returned tuple's second entries contains a map with statistics of the form string->int.
Known keys:
 - "time_ms" : execution time in milliseconds
 - "total_memory_cpu" : The total cumulative memory that is allocated on the CPU throughout the algorithm in bytes
 - "peak_memory_cpu" : The peak memory usage on the CPU in bytes. It holds peak_memory_cpu <= total_memory_cpu
 - "total_memory_gpu" : The total cumulative memory that is allocated on the GPU throughout the algorithm in bytes
 - "peak_memory_gpu" : The peak memory usage on the GPU in bytes. It holds peak_memory_gpu <= total_memory_gpu

:param tensors: the list of input 3D-double tensors
:param target: the target metric, either 'EPS', 'RMSE' or 'PSNR'
:param target_value: the value of the metric that should be achieved
:return: tuple (the compressed tensor, statistics)
            )doc"));

    comp.def("decompress_tthresh_chunked",
        [&](const py::bytes& b, bool verbose)
        {
            const auto v1 = python2compressed(b);
            const auto [v2, stats] = decompressTThreshChunked(v1, verbose);
            std::vector<torch::Tensor> t2(v2.size());
            for (size_t i = 0; i < v2.size(); ++i) {
                t2[i] = rawDouble2torch(v2[i]);
            }
            return std::make_tuple(t2, stats);
        }, py::arg("bytes"), py::arg("verbose"), py::doc(R"doc(
Volume decompression using TThresh with explicit chunking.

The returned tuple's second entries contains a map with statistics of the form string->int.
Known keys:
 - "time_ms" : execution time in milliseconds
 - "total_memory_cpu" : The total cumulative memory that is allocated on the CPU throughout the algorithm in bytes
 - "peak_memory_cpu" : The peak memory usage on the CPU in bytes. It holds peak_memory_cpu <= total_memory_cpu
 - "total_memory_gpu" : The total cumulative memory that is allocated on the GPU throughout the algorithm in bytes
 - "peak_memory_gpu" : The peak memory usage on the GPU in bytes. It holds peak_memory_gpu <= total_memory_gpu

:param bytes: the bytes sequence containing the compressed volume
:return: tuple (list of decompressed double tensors, statistics)
            )doc"));

    comp.def("compress_cuda",
        [&](const torch::Tensor& t, int numLevels, float quantizationStep, int numChunks, bool verbose)
        {
            const auto v1 = torch2rawFloat(t);
            const auto [v2, stats] = compressCUDA(v1, numLevels, quantizationStep, verbose, numChunks);
            return std::make_tuple(compressed2python(v2), stats);
        }, py::arg("tensor"), py::arg("numLevels"), py::arg("quantizationStep"), py::arg("num_chunks"), py::arg("verbose"), py::doc(R"doc(
Volume compression using cudaCompress.

The returned tuple's second entries contains a map with statistics of the form string->int.
Known keys:
 - "time_ms" : execution time in milliseconds
 - "total_memory_cpu" : The total cumulative memory that is allocated on the CPU throughout the algorithm in bytes
 - "peak_memory_cpu" : The peak memory usage on the CPU in bytes. It holds peak_memory_cpu <= total_memory_cpu
 - "total_memory_gpu" : The total cumulative memory that is allocated on the GPU throughout the algorithm in bytes
 - "peak_memory_gpu" : The peak memory usage on the GPU in bytes. It holds peak_memory_gpu <= total_memory_gpu

:param tensor: the input 3D float tensor
:param numLevels:
:param quantizationStep:
:return: tuple (the compressed tensor, statistics)
            )doc"));

    comp.def("decompress_cuda",
        [&](const py::bytes& b, bool verbose)
        {
            const auto v1 = python2compressed(b);
            const auto [v2, stats] = decompressCUDA(v1, verbose);
            return std::make_tuple(rawFloat2torch(v2), stats);
        }, py::arg("bytes"), py::arg("verbose"), py::doc(R"doc(
Volume decompression using cudaCompress.

The returned tuple's second entries contains a map with statistics of the form string->int.
Known keys:
 - "time_ms" : execution time in milliseconds
 - "total_memory_cpu" : The total cumulative memory that is allocated on the CPU throughout the algorithm in bytes
 - "peak_memory_cpu" : The peak memory usage on the CPU in bytes. It holds peak_memory_cpu <= total_memory_cpu
 - "total_memory_gpu" : The total cumulative memory that is allocated on the GPU throughout the algorithm in bytes
 - "peak_memory_gpu" : The peak memory usage on the GPU in bytes. It holds peak_memory_gpu <= total_memory_gpu

:param bytes: the bytes sequence containing the compressed volume
:return: tuple (the decompressed float tensor, statistics)
            )doc"));

    comp.def("compress_cuda_chunked",
        [&](const std::vector<torch::Tensor>& t, int numLevels, float quantizationStep, bool verbose)
        {
            std::vector<RawVolumeCStyle_ptr<float>> t2(t.size());
            for (size_t i = 0; i < t.size(); ++i)
                t2[i] = torch2rawFloat(t[i]);
            const auto [v2, stats] = compressCUDAChunked(t2, numLevels, quantizationStep, verbose);
            return std::make_tuple(compressed2python(v2), stats);
        }, py::arg("tensors"), py::arg("numLevels"), py::arg("quantizationStep"), py::arg("verbose"), py::doc(R"doc(
Volume compression using cudaCompress with explicit chunking.

The returned tuple's second entries contains a map with statistics of the form string->int.
Known keys:
 - "time_ms" : execution time in milliseconds
 - "total_memory_cpu" : The total cumulative memory that is allocated on the CPU throughout the algorithm in bytes
 - "peak_memory_cpu" : The peak memory usage on the CPU in bytes. It holds peak_memory_cpu <= total_memory_cpu
 - "total_memory_gpu" : The total cumulative memory that is allocated on the GPU throughout the algorithm in bytes
 - "peak_memory_gpu" : The peak memory usage on the GPU in bytes. It holds peak_memory_gpu <= total_memory_gpu

:param tensors: the list of input 3D float tensors. Note that all chunks must have the same shape
:param numLevels:
:param quantizationStep:
:return: tuple (the compressed tensor, statistics)
            )doc"));

    comp.def("decompress_cuda_chunked",
        [&](const py::bytes& b, bool verbose)
        {
            const auto v1 = python2compressed(b);
            const auto [v2, stats] = decompressCUDAChunked(v1, verbose);
            std::vector<torch::Tensor> t2(v2.size());
            for (size_t i = 0; i < v2.size(); ++i)
                t2[i] = rawFloat2torch(v2[i]);
            return std::make_tuple(t2, stats);
        }, py::arg("bytes"), py::arg("verbose"), py::doc(R"doc(
Volume decompression using cudaCompress with explicit chunking.

The returned tuple's second entries contains a map with statistics of the form string->int.
Known keys:
 - "time_ms" : execution time in milliseconds
 - "total_memory_cpu" : The total cumulative memory that is allocated on the CPU throughout the algorithm in bytes
 - "peak_memory_cpu" : The peak memory usage on the CPU in bytes. It holds peak_memory_cpu <= total_memory_cpu
 - "total_memory_gpu" : The total cumulative memory that is allocated on the GPU throughout the algorithm in bytes
 - "peak_memory_gpu" : The peak memory usage on the GPU in bytes. It holds peak_memory_gpu <= total_memory_gpu

:param bytes: the bytes sequence containing the compressed volume
:return: tuple (the decompressed float tensors, statistics)
            )doc"));

    const auto cciFactory = [python2compressed](const py::bytes& b)
    {
        return std::make_shared<CudaCompressInteractiveDecompression>(python2compressed(b));
    };
    py::class_<compression::CudaCompressInteractiveDecompression,
        compression::CudaCompressInteractiveDecompression_ptr> cci(
            comp, "CudaCompressInteractiveDecompression");
    cci.def(py::init(cciFactory))
        .def("chunk_width", &CudaCompressInteractiveDecompression::chunkWidth)
        .def("chunk_height", &CudaCompressInteractiveDecompression::chunkHeight)
        .def("chunk_depth", &CudaCompressInteractiveDecompression::chunkDepth)
        .def("num_chunks", &CudaCompressInteractiveDecompression::numChunks)
        .def("decompress", [](CudaCompressInteractiveDecompression* self, int chunk, Volume::MipmapLevel_ptr target)
            {
                TORCH_CHECK(self->chunkWidth() == target->sizeX(),
                    "chunkWidth() != target->sizeX(): ",
                    self->chunkWidth(), "!=", target->sizeX());
                TORCH_CHECK(self->chunkHeight() == target->sizeY(),
                    "chunkHeight() != target->sizeY(): ",
                    self->chunkHeight(), "!=", target->sizeY());
                TORCH_CHECK(self->chunkDepth() == target->sizeZ(),
                    "chunkDepth() != target->sizeZ(): ",
                    self->chunkDepth(), "!=", target->sizeZ());
                TORCH_CHECK(0 <= chunk && chunk < self->numChunks(),
                    "Chunk index out of bounds, index ", chunk, " must be within [0,", self->numChunks(), "]");
                //convert dtype
                compression::CudaCompressInteractiveDecompression::DataType dtype;
                switch (target->type())
                {
                case Volume::TypeFloat:
                    dtype = CudaCompressInteractiveDecompression::DataType::TypeFloat;
                    break;
                case Volume::TypeUChar:
                    dtype = CudaCompressInteractiveDecompression::DataType::TypeUChar;
                    break;
                case Volume::TypeUShort:
                    dtype = CudaCompressInteractiveDecompression::DataType::TypeUShort;
                    break;
                default:
                    throw std::runtime_error("Unknown dtype of the mipmap level");
                }
                //launch decompression
                return self->decompress(chunk, target->gpuSurface(), dtype);
            }, py::arg("chunk"), py::arg("target"), py::doc(R"doc(
Decompresses the chunk with index 'chunk' into the target mipmap level.
The dimension of the mipmap level must match the chunk dimension.
Returns the statistics
)doc"))
    .def("global_statistics", &compression::CudaCompressInteractiveDecompression::globalStatistics,
        py::doc("Returns the accumulated statistics over all decompression calls since construction."))
    ;
}
