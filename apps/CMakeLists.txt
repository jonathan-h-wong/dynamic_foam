# Test renderer
add_executable(test_renderer test_renderer.cpp)
target_link_libraries(test_renderer PRIVATE dynamic_foam)
target_include_directories(test_renderer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)