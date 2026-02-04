# Find packages
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(EnTT CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

# Fetch ImGui backends from GitHub
include(FetchContent)
FetchContent_Declare(
    imgui_backends
    URL "https://github.com/ocornut/imgui/archive/refs/tags/v1.91.9.tar.gz"
    DOWNLOAD_EXTRACT_TIMESTAMP ON
)
FetchContent_MakeAvailable(imgui_backends)

# Source files
set (DYNAMIC_FOAM_SOURCES
    kernels.cu
    ${imgui_backends_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_backends_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
add_library(dynamic_foam ${DYNAMIC_FOAM_SOURCES})

# Link Header files and dependencies
target_include_directories(dynamic_foam 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${imgui_backends_SOURCE_DIR}>
        $<BUILD_INTERFACE:${imgui_backends_SOURCE_DIR}/backends>
        $<INSTALL_INTERFACE:include>
    PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(dynamic_foam
    PUBLIC
        glfw
        imgui::imgui
        OpenGL::GL
        CUDA::cudart
        EnTT::EnTT
        fmt::fmt
        glm::glm
)

# CUDA settings
set_target_properties(dynamic_foam PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "75;86"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Compiler warnings
if(MSVC)
    target_compile_options(dynamic_foam PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W3> $<$<COMPILE_LANGUAGE:CXX>:/MP> $<$<COMPILE_LANGUAGE:CXX>:/WX>)
else()
    target_compile_options(dynamic_foam PRIVATE -Wall -Wextra -Wpedantic)
endif()