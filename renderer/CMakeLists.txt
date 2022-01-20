cmake_minimum_required(VERSION 3.10)

find_package(OpenMP)

# your configuraton
set(LIBRARY_HEADER_FILES
	commons.h
    errors.h
	volume.h
    halton_sampler.h

    # old
	halton_sampler.h

    #new
    json_utils.h
    lru_cache.h
    parameter.h
	imodule.h
    pytorch_utils.h
    module_registry.h
    transfer_function.h
    volume_interpolation.h
    volume_interpolation_grid.h
    background_worker.h
    blending.h
    camera.h
    ray_evaluation.h
    ray_evaluation_stepping.h
    ray_evaluation_monte_carlo.h
    sha1.h
    kernel_loader.h
    iimage_evaluator.h
    image_evaluator_simple.h
    brdf.h
    phase_function.h
    spherical_harmonics.h
    irasterization.h
    particle_integration.h
    pytorch_functions.h
    volume_interpolation_network.h

    opengl_framebuffer.h
	opengl_mesh.h
	opengl_oit.h
	opengl_shader.h
	opengl_utils.h
    rasterization_meshes.h
    rasterization_meshes.cpp

    ../third-party/tinyformat/tinyformat.h
	../third-party/nlohmann/json.hpp
	)
set(LIBRARY_CUDA_HEADER_FILES
    # Helper, also used outside of kernels
    helper_math.cuh
    helper_matrixmath.cuh
	renderer_commons.cuh
	renderer_tensor.cuh
	renderer_utils.cuh
	renderer_adjoint.cuh
    renderer_cudad_bridge.cuh
	cooperative_groups.cuh

    # Implementations, used only kernels with extra preprocessor defines
    renderer_tf_identity.cuh
    renderer_tf_texture.cuh
    renderer_tf_piecewise.cuh
    renderer_tf_gaussian.cuh
    renderer_tf_kernels.cuh
    renderer_volume_grid.cuh
    renderer_volume_kernels1.cuh
    renderer_volume_kernels2.cuh
    renderer_volume_kernels3.cuh
    renderer_blending.cuh
    renderer_camera.cuh
    renderer_camera_kernels.cuh
    renderer_ray_evaluation_stepping_iso.cuh
    renderer_ray_evaluation_stepping_dvr.cuh
    renderer_ray_evaluation_monte_carlo.cuh
    renderer_image_evaluator_simple.cuh
    renderer_brdf_lambert.cuh
    renderer_sampler_curand.cuh
    renderer_phase_function.cuh
    renderer_spherical_harmonics.cuh
    renderer_particle_integration_kernels.cuh
    renderer_volume_tensorcores.cuh
    )
set(LIBRARY_CUDA_FILES
    iimage_evaluator_cuda.cu
    spherical_harmonics_cuda.cu
    pytorch_functions_cuda.cu
    transfer_function_texture_cuda.cu

    opengl_framebuffer_cuda.cu

	)
set(LIBRARY_SOURCE_FILES
    volume.cpp
	volume_implicit.cpp

    #new
    module_registry.cpp
    transfer_function.cpp
    transfer_function_identity.cpp
    transfer_function_texture.cpp
    transfer_function_piecewise.cpp
    transfer_function_gaussian.cpp
    volume_interpolation.cpp
    volume_interpolation_grid.cpp
    blending.cpp
    camera.cpp
    ray_evaluation.cpp
    ray_evaluation_stepping.cpp
    ray_evaluation_monte_carlo.cpp
    sha1.cpp
    kernel_loader.cpp
    iimage_evaluator.cpp
    image_evaluator_simple.cpp
    brdf.cpp
    phase_function.cpp
    spherical_harmonics.cpp
    irasterization.cpp
    particle_integration.cpp
    pytorch_functions.cpp
    volume_interpolation_network.cpp

    opengl_utils.cpp
    opengl_mesh.cpp
	opengl_oit.cpp
	opengl_shader.cpp
    opengl_framebuffer.cpp
	)
set(LIBRARY_RESOURCES
	shaders/PassThrough.vs
	shaders/ShowUVs.vs
	shaders/ShowUVs.fs
	shaders/ShowNormals.fs
	shaders/ScreenQuad.vs
    shaders/SimpleDiffuse.fs
    shaders/Particles.vs
    shaders/Particles.gs
    shaders/Particles.fs
	)
# the target name of the library
set(LIBRARY_NAME Renderer CACHE INTERNAL "the target name of the library, also used for the binding")

# get list of filenames
file(GLOB RENDERER_KERNEL_FILES1
	list_directories false
	${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)
file(GLOB RENDERER_KERNEL_FILES2
	list_directories false
	${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD/*.cuh
	${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD/*.h)
set(RENDERER_KERNEL_FILES
	${RENDERER_KERNEL_FILES1} ${RENDERER_KERNEL_FILES2}
	PARENT_SCOPE)

if(NOT ${RENDERER_ONLY_RESOURCES})

    # the library, compiled as a shared library
    set(LIBRARY_ALL_FILES
        ${LIBRARY_HEADER_FILES}
        ${LIBRARY_CUDA_HEADER_FILES}
        ${LIBRARY_SOURCE_FILES}
        ${LIBRARY_CUDA_FILES}
        ${LIBRARY_RESOURCES}
    )
    if(RENDERER_SHARED_LIB)
        add_library(${LIBRARY_NAME} SHARED
            ${LIBRARY_ALL_FILES})
    else(RENDERER_SHARED_LIB)
        add_library(${LIBRARY_NAME} STATIC
            ${LIBRARY_ALL_FILES})
    endif(RENDERER_SHARED_LIB)
    source_group("CUDA Headers" REGULAR_EXPRESSION ".*\.cuh")

    set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES 61 72)
    set_target_properties(${LIBRARY_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_compile_options(${LIBRARY_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17 --expt-relaxed-constexpr --extended-lambda>)

    # openmp
    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            ${LIBRARY_NAME} PRIVATE optimized OpenMP::OpenMP_CXX)
    endif()
    set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_STANDARD 17)
    set_property(TARGET ${LIBRARY_NAME} PROPERTY CUDA_ARCHITECTURES 61 72)
    
    # get include path to the first library header, to be used in the test application
    list(GET ${LIBRARY_HEADER_FILES} 0 LIBRARY_INCLUDE_DIRb)
    get_filename_component(LIBRARY_INCLUDE_DIRa ${LIBRARY_INCLUDE_DIRb} ABSOLUTE)
    get_filename_component(LIBRARY_INCLUDE_DIRc ${LIBRARY_INCLUDE_DIRa} DIRECTORY)
    set(LIBRARY_INCLUDE_DIR ${LIBRARY_INCLUDE_DIRc} CACHE FILEPATH "include directory of the custom library") 
    target_include_directories(${LIBRARY_NAME}
        PUBLIC 
            ${LIBRARY_INCLUDE_DIR}
            ${PYTHON_INCLUDE_DIR} 
            ${CMAKE_SOURCE_DIR}/third-party/cuMat 
            ${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD
            ${CMAKE_SOURCE_DIR}/third-party/portable-file-dialogs
            ${CMAKE_SOURCE_DIR}/third-party/magic_enum/include
            ${CMAKE_SOURCE_DIR}/third-party/thread-pool/include
            ${GLM_INCLUDE_DIRS}
            ${GLEW_INCLUDE_DIR} 
            ${GLFW_INCLUDE_DIRS}
            ${TORCH_INCLUDE_DIR}
            ../third-party/tinyformat/
            ../third-party/nlohmann/
        )
    
    # libraries
    target_link_libraries(${LIBRARY_NAME}
        PUBLIC
            ${CUDA_LIBRARIES}
            CUDA::cuda_driver CUDA::nvrtc CUDA::curand
            ImGui
            lz4
            #pybind11::module
            pybind11::lto
            ${TORCH_LIBRARIES}
            )
    if (RENDERER_BUILD_OPENGL_SUPPORT)
        target_link_libraries(${LIBRARY_NAME}
            PUBLIC
                ${OPENGL_LIBRARY} 
                ${GLEW_SHARED_LIBRARY_RELEASE} 
                ${GLFW_LIBRARIES}
                )
        target_compile_definitions(${LIBRARY_NAME} PUBLIC 
            RENDERER_OPENGL_SUPPORT=1)
    else()
        target_compile_definitions(${LIBRARY_NAME} PUBLIC 
            RENDERER_OPENGL_SUPPORT=0)
    endif()
    set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    
    # CUDA runtime compilation
    list(GET ARCH_FLAGS -1 RENDERER_NVCC_FLAGS)
    set(RENDERER_KERNEL_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/third-party/cudad/include/cudAD)
    set(RENDERER_KERNEL_DIRS_STR "{")
    foreach(line IN LISTS RENDERER_KERNEL_DIRS)
        string(APPEND RENDERER_KERNEL_DIRS_STR "\"${line}\",")
    endforeach() 
    string(APPEND RENDERER_KERNEL_DIRS_STR "}")
    message(STATUS "renderer kernel directories: " ${RENDERER_KERNEL_DIRS_STR})
    if (WIN32)
        target_compile_definitions(${LIBRARY_NAME} PUBLIC 
            RENDERER_SHADER_DIRS=${RENDERER_KERNEL_DIRS_STR}
            RENDERER_RUNTIME_COMPILATION=1
            NVCC_ARGS="${RENDERER_NVCC_FLAGS}"
            NVCC_INCLUDE_DIR=${CUDA_INCLUDE_DIRS})
    else()
        # there are some errors that RENDERER_SHADER_DIRS is not properly passed to the compiler
        # therefore, it is required on Unix to provide a custom source lodaer
        # (Or you hard-code it in renderer/kernel_loader.cpp, line 151ff
        target_compile_definitions(${LIBRARY_NAME} PUBLIC 
            RENDERER_RUNTIME_COMPILATION=1
            NVCC_ARGS="${RENDERER_NVCC_FLAGS}"
            NVCC_INCLUDE_DIR=${CUDA_INCLUDE_DIRS})
    endif()
    
    # Shared or static
    if(RENDERER_SHARED_LIB)
        target_compile_definitions(${LIBRARY_NAME}
            PUBLIC RENDERER_BUILD_SHARED 
            PRIVATE BUILD_MAIN_LIB)
    else(RENDERER_SHARED_LIB)
        # no extra definitions
    endif(RENDERER_SHARED_LIB)

    if(RENDERER_SHARED_LIB)
    # post-build commands, copies the dll to the bin/ folder
    get_filename_component(LIBRARY_INCLUDE_DIR ${LIBRARY_INCLUDE_DIRa} DIRECTORY)
    add_custom_command(TARGET ${LIBRARY_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${LIBRARY_NAME}> ${CMAKE_SOURCE_DIR}/bin/${LIBRARY_NAME}${CMAKE_SHARED_LIBRARY_SUFFIX}

        COMMENT "Copies the libarary .dll to bin/"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
        
        VERBATIM
        )
    endif(RENDERER_SHARED_LIB)

    # Include shaders in executable
    file(DOWNLOAD "https://raw.githubusercontent.com/vector-of-bool/cmrc/master/CMakeRC.cmake"
					"${CMAKE_BINARY_DIR}/CMakeRC.cmake")
    include("${CMAKE_BINARY_DIR}/CMakeRC.cmake")
    cmrc_add_resource_library(
		${LIBRARY_NAME}-shaders

		ALIAS ${LIBRARY_NAME}::s
		NAMESPACE shaders

		${LIBRARY_RESOURCES}
		)
	target_link_libraries(${LIBRARY_NAME} PUBLIC ${LIBRARY_NAME}::s)
	set_property(TARGET ${LIBRARY_NAME}-shaders PROPERTY
	  MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set_property(TARGET ${LIBRARY_NAME}-shaders PROPERTY CUDA_ARCHITECTURES 61 72)
endif()