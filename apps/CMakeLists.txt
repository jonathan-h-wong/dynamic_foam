# Entrypoint for application
add_executable(app main.cpp)
target_link_libraries(app PRIVATE dynamic_foam)

# Need imgui backends headers for main.cpp
target_include_directories(app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)