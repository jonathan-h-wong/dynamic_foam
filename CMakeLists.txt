cmake_minimum_required(VERSION 3.20)
project(dynamic_foam CUDA CXX)

# Cpp standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Point to library and entrypoint, which link dependenices internally
add_subdirectory(src)
add_subdirectory(apps)