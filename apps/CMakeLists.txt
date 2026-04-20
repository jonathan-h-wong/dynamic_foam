# Test renderer
add_executable(test_renderer test_renderer.cpp)
target_link_libraries(test_renderer PRIVATE dynamic_foam)
target_include_directories(test_renderer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Triangulation profiler
# add_executable(profile_triangulation profile_triangulation.cpp)
# target_link_libraries(profile_triangulation PRIVATE dynamic_foam)
# target_include_directories(profile_triangulation PRIVATE
#     ${CMAKE_SOURCE_DIR}/include
#     ${CGAL_INCLUDE_DIR}
#     ${BOOST_INCLUDE_DIR}
# )
# target_compile_definitions(profile_triangulation PRIVATE CGAL_HEADER_ONLY=1)