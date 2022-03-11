cmake_minimum_required(VERSION 3.10)

find_package(OpenMP)

file(GLOB_RECURSE COMPRESSION_SRC
    list_directories false
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*)
    
file(GLOB_RECURSE COMPRESSION_INCLUDE
    list_directories false
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*)
    
file(GLOB_RECURSE COMPRESSION_TEST
    list_directories false
    ${CMAKE_CURRENT_SOURCE_DIR}/test/*)

if(NOT ${RENDERER_ONLY_RESOURCES})

    # the library, compiled as a shared library
    set(COMPRESSION_ALL_FILES
        ${COMPRESSION_SRC}
        ${COMPRESSION_INCLUDE}
    )
    if(RENDERER_SHARED_LIB)
        add_library(VolumetricCompression SHARED
            ${COMPRESSION_ALL_FILES})
    else(RENDERER_SHARED_LIB)
        add_library(VolumetricCompression STATIC
            ${COMPRESSION_ALL_FILES})
    endif(RENDERER_SHARED_LIB)
    
    foreach(_source IN ITEMS ${COMPRESSION_ALL_FILES})
        get_filename_component(_source_path "${_source}" PATH)
        file(RELATIVE_PATH _source_path_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source_path}")
        string(REPLACE "/" "\\" _group_path "${_source_path_rel}")
        source_group("${_group_path}" FILES "${_source}")
    endforeach()

    set_property(TARGET VolumetricCompression PROPERTY CUDA_ARCHITECTURES 61 72)
    set_target_properties(VolumetricCompression PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
    )
    target_compile_options(VolumetricCompression PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--std=c++17 --expt-relaxed-constexpr --extended-lambda>)

    # openmp
    if(OpenMP_CXX_FOUND)
        target_link_libraries(
            VolumetricCompression PRIVATE optimized OpenMP::OpenMP_CXX)
    endif()
    set_property(TARGET VolumetricCompression PROPERTY CUDA_STANDARD 17)
    set_property(TARGET VolumetricCompression PROPERTY CUDA_ARCHITECTURES 61 72)
    
    target_include_directories(VolumetricCompression
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_SOURCE_DIR}/third-party/cuMat
                ${CMAKE_SOURCE_DIR}/third-party/cuMat/third-party
        )
    
    # libraries
    target_link_libraries(VolumetricCompression
        PUBLIC
            ${CUDA_LIBRARIES}
            CUDA::cuda_driver
            )
    set_target_properties(VolumetricCompression PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
    
    # Shared or static
    if(RENDERER_SHARED_LIB)
        target_compile_definitions(VolumetricCompression
            PUBLIC RENDERER_BUILD_SHARED 
            PRIVATE BUILD_MAIN_LIB)
    else(RENDERER_SHARED_LIB)
        # no extra definitions
    endif(RENDERER_SHARED_LIB)

    if(RENDERER_SHARED_LIB)
    # post-build commands, copies the dll to the bin/ folder
    add_custom_command(TARGET ${LIBRARY_NAME}
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:VolumetricCompression> ${CMAKE_SOURCE_DIR}/bin/VolumetricCompression${CMAKE_SHARED_LIBRARY_SUFFIX}

        COMMENT "Copies the libarary .dll to bin/"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
        
        VERBATIM
        )
    endif(RENDERER_SHARED_LIB)

    # Test 
    add_executable(VolumetricCompressionTest ${COMPRESSION_TEST})
    target_link_libraries(VolumetricCompressionTest 
        PRIVATE VolumetricCompression)
    set_property(TARGET VolumetricCompressionTest PROPERTY CUDA_STANDARD 17)
    set_property(TARGET VolumetricCompressionTest PROPERTY CUDA_ARCHITECTURES 61 72)

endif()