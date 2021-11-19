#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <tinyformat.h>
#include <cuMat/src/Context.h>
#include <glm/gtx/string_cast.hpp>
#include <third-party/Eigen/Core> // in cuMat

#include <kernel_loader.h>
#include <module_registry.h>
#include <opengl_utils.h>

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
		std::cout << "Load file " << e.filename() << std::endl;
		auto f = fs.open(currentPath + e.filename());
		std::string content(f.size(), '\0');
		memcpy(content.data(), f.begin(), f.size());
		fileList.push_back({ e.filename(), content });
	} else
	{
		std::cout << "Walk directory " << currentPath << std::endl;
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
	});
	
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
	});

	py::class_<double3>(m, "double3")
		.def(py::init<>())
		.def(py::init([](double x, double y, double z) {return make_double3(x, y, z); }))
		.def_readwrite("x", &double3::x)
		.def_readwrite("y", &double3::y)
		.def_readwrite("z", &double3::z)
		.def("__str__", [](const double3& v)
			{
				return tinyformat::format("(%f, %f, %f)", v.x, v.y, v.z);
			});

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
			});

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
}