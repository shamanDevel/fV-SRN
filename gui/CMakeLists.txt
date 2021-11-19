cmake_minimum_required(VERSION 3.10)

if (${RENDERER_BUILD_GUI})
    if (NOT RENDERER_BUILD_OPENGL_SUPPORT)
        message(FATAL_ERROR "Requested CLI, but OpenGL-support is disabled, which is a dependency")
    endif()

	# your configuration
	set(GUI_FILES
		main.cpp
		#utils.h
		visualizer_commons.h
		visualizer.h
		visualizer.cpp
		visualizer_kernels.h
		#visualizer_kernels_cuda.cu
		quad_drawer.h
		quad_drawer.cpp

		../third-party/lodepng/lodepng.cpp
		../third-party/lodepng/lodepng.h
		../third-party/portable-file-dialogs/portable-file-dialogs.h
		)
	set(GUI_NAME Gui CACHE INTERNAL "target name for the GUI")

	# create the executable
	add_executable(${GUI_NAME}
		${GUI_FILES}
		)
	target_include_directories(${GUI_NAME}
		PRIVATE 
		${LIBRARY_INCLUDE_DIR} 
		${GLM_INCLUDE_DIRS} 
		${GLEW_INCLUDE_DIR} 
		${GLFW_INCLUDE_DIRS}
		${CMAKE_SOURCE_DIR}/third-party/cuMat
		${CMAKE_SOURCE_DIR}/third-party/nlohmann
		${CMAKE_SOURCE_DIR}/third-party/lodepng
		${CMAKE_SOURCE_DIR}/third-party/thread-pool/include
		${CMAKE_SOURCE_DIR}/third-party/portable-file-dialogs
		${CMAKE_SOURCE_DIR}/third-party/magic_enum/include
		)
	target_link_libraries(${GUI_NAME}
		${LIBRARY_NAME} 
		${OPENGL_LIBRARY} 
		${GLEW_SHARED_LIBRARY_RELEASE} 
		${GLFW_LIBRARIES}
		${PYTHON_LIBRARIES}
		ImGui)
	set_property(TARGET ${GUI_NAME} PROPERTY CXX_STANDARD 17)
	set_property(TARGET ${GUI_NAME} PROPERTY CUDA_ARCHITECTURES 61 72)
	add_custom_command(TARGET ${GUI_NAME}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/bin
		COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${GUI_NAME}> ${CMAKE_SOURCE_DIR}/bin/${GUI_NAME}${CMAKE_EXECUTABLE_SUFFIX}
		
		COMMENT "Copies the executable to bin/"
		WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/..
		
		VERBATIM
		)
	set_property(TARGET ${GUI_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
	target_compile_definitions(${GUI_NAME}
	PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLEW=1)

endif()