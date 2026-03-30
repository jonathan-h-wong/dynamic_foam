# Find packages
find_package(glfw3 CONFIG REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(EnTT CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)

# Add CGAL header-only include directory
set(CGAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/CGAL-6.0/include" CACHE PATH "CGAL header-only includes")
set(BOOST_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/boost" CACHE PATH "Boost header-only includes")

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
    Sim2D/bvh.cu
    Sim2D/render.cu
    Sim2D/collision.cu
    Sim2D/physics.cpp
    Sim2D/simulation.cpp
    Sim2D/topology.cpp
    Sim2D/user_input.cpp
    Sim2D/sample_scene.cpp
    ${imgui_backends_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_backends_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
add_library(dynamic_foam ${DYNAMIC_FOAM_SOURCES})

# Link Header files and dependencies
target_include_directories(dynamic_foam 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CGAL_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${BOOST_INCLUDE_DIR}>
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
        CGAL::CGAL
)

# CGAL header-only configuration
target_compile_definitions(dynamic_foam PRIVATE CGAL_HEADER_ONLY=1)

# CUDA settings
set_target_properties(dynamic_foam PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES "75;86"
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Compiler warnings
if(MSVC)
    target_compile_options(dynamic_foam PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
        $<$<COMPILE_LANGUAGE:CXX>:/MP>
        $<$<COMPILE_LANGUAGE:CXX>:/WX>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
else()
    target_compile_options(dynamic_foam PRIVATE
        -Wall -Wextra -Wpedantic
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
endif()

# Tier 2 sources: host-side subsystems that call into Tier 1 CUDA kernels.
# These .cpp files include .cuh headers (adjacency.cuh, bvh.cuh) which expose
# CUB/Thrust device code that MSVC cannot parse. Routing them through NVCC via
# LANGUAGE CUDA makes them part of the CUDA translation unit while still running
# entirely on the host. simulation.cpp and sample_scene.cpp are Tier 3 (pure
# MSVC) -- they own EnTT registries and call CGAL, both of which break under
# NVCC, so their .cuh include paths are guarded with #ifdef __CUDACC__.
set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/Sim2D/physics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Sim2D/topology.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Sim2D/user_input.cpp
    PROPERTIES LANGUAGE CUDA
)