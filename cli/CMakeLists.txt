cmake_minimum_required(VERSION 3.10)

if (RENDERER_BUILD_CLI)
    if (NOT RENDERER_BUILD_OPENGL_SUPPORT)
        message(FATAL_ERROR "Requested CLI, but OpenGL-support is disabled, which is a dependency")
    endif()

    set(CLI_FILES
        cli.cpp
        CLI11.hpp)
    
    set(CLI_NAME CLI CACHE INTERNAL "target name for the CLI")
    add_executable(${CLI_NAME} ${CLI_FILES})
    target_include_directories(${CLI_NAME}
		PRIVATE 
		${LIBRARY_INCLUDE_DIR}
		${CMAKE_SOURCE_DIR}/third-party/indicators/include)
    target_link_libraries(${CLI_NAME}
		${LIBRARY_NAME}
		${PYTHON_LIBRARIES})
        
    set_property(TARGET ${CLI_NAME} PROPERTY CXX_STANDARD 17)
	set_property(TARGET ${CLI_NAME} PROPERTY CUDA_ARCHITECTURES 61 72)
	add_custom_command(TARGET ${CLI_NAME}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${CLI_NAME}> ${CMAKE_SOURCE_DIR}/bin/${CLI_NAME}${CMAKE_EXECUTABLE_SUFFIX}
		
		COMMENT "Copies the executable to bin/"
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
		
		VERBATIM
		)

endif()