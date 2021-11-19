cmake_minimum_required(VERSION 3.10)
project(SceneNetworkRenderer)

cmake_policy(SET CMP0074 NEW) # use _ROOT environment variables
cmake_policy(SET CMP0091 NEW) # set runtime library
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

####################################
# MAIN OPTIONS
####################################
OPTION(RENDERER_ONLY_RESOURCES "Only build the kernel resources, for the UNIX server where the main building is done in setup.py" OFF)
OPTION(RENDERER_SHARED_LIB "Build renderer as a shared library" OFF)
OPTION(RENDERER_BUILD_GUI "Build gui project" ON)
OPTION(RENDERER_BUILD_TESTS "Build unit test project" ON)
OPTION(RENDERER_BUILD_BINDINGS "Build the python bindings" ON)
OPTION(RENDERER_BUILD_CLI "Build command line interface project" ON)
OPTION(RENDERER_BUILD_OPENGL_SUPPORT "Build with OpenGL rasterization support" ON)
add_definitions(-DKERNEL_DOUBLE_PRECISION=1) # host: double precision for unit tests

####################################
# C++ standard
####################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_definitions(-DNOMINMAX)

####################################
# GENERAL THIRD-PARTY DEPENDENCIES
####################################

# CUDA is always required
find_package(CUDA REQUIRED)
if(COMMAND CUDA_SELECT_NVCC_ARCH_FLAGS)
	if (WIN32) # inference-gui
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
	else() # server
		CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS 6.1)
	endif()
	STRING(REPLACE ";" "=" ARCH_FLAGS2 "${ARCH_FLAGS}")
	LIST(APPEND CUDA_NVCC_FLAGS "${ARCH_FLAGS2}")
	message(STATUS "cuda flags: ${ARCH_FLAGS2}")
endif()
list(APPEND CUDA_NVCC_FLAGS "-std=c++17" "--use_fast_math" "--generate-line-info" "–Xptxas" "-v")
set(MY_CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})
set(CUDA_SEPARABLE_COMPILATION ON)
message(STATUS "final cuda flags: ${CMAKE_CUDA_FLAGS}")
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# cuMat-Settings
add_definitions(-DCUMAT_SINGLE_THREAD_CONTEXT=1 -DTHRUST_IGNORE_CUB_VERSION_CHECK=1)

####################################
# OpenGL
####################################
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
if(NOT ${RENDERER_ONLY_RESOURCES})
    find_package(GLM REQUIRED)
    if (RENDERER_BUILD_GUI)
        find_package(OpenGL REQUIRED)
        find_package(GLEW REQUIRED)
        find_package(GLFW)
        if(NOT GLFW_FOUND)
            message(STATUS "GLFW could not be found with normal lookup, use PkgConfig instead")
            find_package(PkgConfig REQUIRED)
            pkg_search_module(GLFW REQUIRED glfw3)
        else()
            set(GLFW_LIBRARIES ${GLFW_LIBRARY})
            set(GLFW_INCLUDE_DIRS ${GLFW_INCLUDE_DIR})
        endif()
        # copy shared libraries
        if (WIN32)
            # glew dll if running on windows
            string(REPLACE "/lib/" "/bin/" GLEW_BINARY_RELEASEa ${GLEW_SHARED_LIBRARY_RELEASE})
            string(REPLACE ${CMAKE_STATIC_LIBRARY_SUFFIX} ${CMAKE_SHARED_LIBRARY_SUFFIX} GLEW_BINARY_RELEASE ${GLEW_BINARY_RELEASEa})
            file(COPY ${GLEW_BINARY_RELEASE} DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
            # glfw dll
            get_filename_component(GLFW_BINARY_DIRECTORY ${GLFW_LIBRARY} DIRECTORY)
            file(COPY ${GLFW_BINARY_DIRECTORY}/glfw3.dll DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
        else()
            # copy glew, glfw, glm
            file(COPY ${GLEW_SHARED_LIBRARY_RELEASE} DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
        endif()
    endif() # ${RENDERER_BUILD_GUI}
endif() # NOT ${RENDERER_ONLY_RESOURCES}

####################################
# Python
####################################

if((NOT ${RENDERER_ONLY_RESOURCES}) AND ${RENDERER_BUILD_BINDINGS})
    set(Python3_ROOT_DIR ${CMAKE_SOURCE_DIR}/env)
    find_package(PythonInterp 3.6 REQUIRED)
    find_package(PythonLibs 3.6 REQUIRED)
    #find_program(VIRTUALENV virtualenv)
    #find_package(Python COMPONENTS Interpreter Development)
    get_filename_component(PYTHON_DIRECTORY ${PYTHON_EXECUTABLE} DIRECTORY)
endif()

####################################
# Pytorch
####################################

if(NOT ${RENDERER_ONLY_RESOURCES})
    # find installation path
    if(NOT DEFINED ${TORCH_PATH})
        # query torch path from python
        execute_process(COMMAND CMD /c python -c "import torch.utils.cpp_extension; print(torch.utils.cpp_extension.include_paths()[0])" OUTPUT_VARIABLE TORCH_FIRST_INCLUDE_DIR)
        get_filename_component(TORCH_ROOT ${TORCH_FIRST_INCLUDE_DIR}/../ ABSOLUTE)
        set(TORCH_PATH "${TORCH_ROOT}" CACHE FILEPATH "path to pytorch in the python installation")
        if(NOT (EXISTS ${TORCH_PATH}))
            message( FATAL_ERROR "Pytorch not found, is it not installed in the python distribution ${PYTHON_DIRECTORY}?")
        else()
            message(STATUS "Torch found at ${TORCH_PATH}")
        endif()
    else()
        message(STATUS "Manually specifying torch path as ${TORCH_PATH}")
    endif()
    # ask Torch's CMake configuration
    set(TORCH_CONFIG_PATH "${TORCH_PATH}/share/cmake/Torch" CACHE FILEPATH "possible path where TorchConfig.cmake is located")
    list(APPEND CMAKE_PREFIX_PATH ${TORCH_CONFIG_PATH})
    find_package(Torch REQUIRED)
    # get libraries (hard coded), copied from torch.utils.cpp_extension.CUDAExtension
    set(TORCH_LIBRARY_NAMES
        c10 c10_cuda torch torch_cpu torch_cuda torch_cuda_cpp torch_cuda_cu torch_python)
    set(TORCH_LIBRARIES ${TORCH_LIBRARY})
    FOREACH(LIB_NAME ${TORCH_LIBRARY_NAMES})
      set(LIB_VAR "TORCH_LIB_${LIB_NAME}") # Name of the variable which stores result of the search
      FIND_LIBRARY(${LIB_VAR} ${LIB_NAME} PATHS ${TORCH_PATH}/lib)
      list(APPEND TORCH_LIBRARIES ${${LIB_VAR}})
    ENDFOREACH()
    message(STATUS "Torch: full library list: ${TORCH_LIBRARIES}")
    # copy shared library to bin/
    file(MAKE_DIRECTORY bin)
    file(GLOB TORCH_SHARED_LIBRARIES
        ${TORCH_PATH}/lib/*${CMAKE_SHARED_LIBRARY_SUFFIX})
    message(STATUS "Torch: shared libraries to copy: ${TORCH_SHARED_LIBRARIES}")
    file(COPY ${TORCH_SHARED_LIBRARIES} DESTINATION ${CMAKE_SOURCE_DIR}/bin/)
    # get include directories
    set(TORCH_INCLUDE_DIR "${TORCH_PATH}/include;${TORCH_PATH}/include/torch/csrc/api/include" CACHE FILEPATH "include directory for the pytorch headers")
    message(STATUS "Torch: include directories: ${TORCH_INCLUDE_DIR}")
endif()

####################################
# THIRD PARTY
####################################
add_subdirectory(third-party/lz4)
set_property(TARGET lz4 PROPERTY CUDA_ARCHITECTURES 61 72)
set_target_properties(lz4 PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
add_subdirectory(imgui)
add_subdirectory(third-party/pybind11)

####################################
# THE LIBRARY
####################################
add_subdirectory(renderer)
message(STATUS "renderer kernel files: ${RENDERER_KERNEL_FILES}")
# resources
file(DOWNLOAD "https://raw.githubusercontent.com/vector-of-bool/cmrc/master/CMakeRC.cmake"
				"${CMAKE_BINARY_DIR}/CMakeRC.cmake")
include("${CMAKE_BINARY_DIR}/CMakeRC.cmake")
cmrc_add_resource_library(
	${LIBRARY_NAME}-kernels

	ALIAS ${LIBRARY_NAME}::rc
	NAMESPACE kernels

	${RENDERER_KERNEL_FILES}
	)
set_property(TARGET ${LIBRARY_NAME}-kernels PROPERTY CUDA_ARCHITECTURES 61 72)
set_target_properties(${LIBRARY_NAME}-kernels PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

####################################
# TEST APPLICATION
# Depends on the library
####################################
if(NOT ${RENDERER_ONLY_RESOURCES} AND ${RENDERER_BUILD_GUI})
add_subdirectory(gui)
endif()

####################################
# COMMAND LINE INTERFACE
# for converting volumes.
# Depends on the library
####################################
if(NOT ${RENDERER_ONLY_RESOURCES} AND ${RENDERER_BUILD_CLI})
add_subdirectory(cli)
endif()

####################################
# PYTHON BINDINGS
# depend on the library
####################################
if((NOT ${RENDERER_ONLY_RESOURCES}) AND ${RENDERER_BUILD_BINDINGS})
add_subdirectory(bindings)
endif()

if((NOT ${RENDERER_ONLY_RESOURCES}) AND ${RENDERER_BUILD_TESTS})
add_subdirectory(unittests)
endif()
