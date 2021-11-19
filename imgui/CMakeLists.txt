cmake_minimum_required(VERSION 3.10)

# your configuration
set(IMGUI_FILES
	imconfig.h
	imgui.h
	imgui.cpp
	imgui_demo.cpp
	imgui_draw.cpp
	imgui_internal.h
	imgui_widgets.cpp
	imstb_rectpack.h
	imstb_textedit.h
	imstb_truetype.h
	IconsFontAwesome5.h
	imgui_extension.h
	imgui_extension.cpp
	)
set(IMGUI_IMPL_FILES
	imgui_impl_opengl3.cpp
	imgui_impl_opengl3.h
	imgui_impl_glfw.cpp
	imgui_impl_glfw.h
	)


# create the executable
if (${RENDERER_BUILD_GUI})
    add_library(ImGui ${IMGUI_FILES} ${IMGUI_IMPL_FILES})
else()
    add_library(ImGui ${IMGUI_FILES})
endif()
target_include_directories(ImGui
	PRIVATE 
	${GLM_INCLUDE_DIRS} 
	PUBLIC
	${CMAKE_CURRENT_SOURCE_DIR}
	)
if (${RENDERER_BUILD_GUI})
    target_include_directories(ImGui
        PRIVATE 
        ${GLEW_INCLUDE_DIR} 
        ${GLFW_INCLUDE_DIRS})
    target_link_libraries(ImGui
        ${OPENGL_LIBRARY} 
        ${GLEW_SHARED_LIBRARY_RELEASE} 
        ${GLFW_LIBRARIES})
endif()
set_property(TARGET ImGui PROPERTY CXX_STANDARD 17)
set_property(TARGET ImGui PROPERTY CUDA_ARCHITECTURES 61 72)
target_compile_definitions(ImGui
	PRIVATE IMGUI_IMPL_OPENGL_LOADER_GLEW=1)

# Include resources in executable
file(DOWNLOAD "https://raw.githubusercontent.com/vector-of-bool/cmrc/master/CMakeRC.cmake"
					"${CMAKE_BINARY_DIR}/CMakeRC.cmake")
include("${CMAKE_BINARY_DIR}/CMakeRC.cmake")
cmrc_add_resource_library(
	ImGui-resources

	ALIAS ImGui::rc
	NAMESPACE ui

	resources/fa-regular-400.ttf
	resources/fa-solid-900.ttf
	)
target_link_libraries(ImGui ImGui::rc)
set_property(TARGET ImGui-resources PROPERTY CUDA_ARCHITECTURES 61 72)

set_target_properties(ImGui PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
set_target_properties(ImGui-resources PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

