# Entrypoint for application
add_executable(app main.cpp)
target_link_libraries(app PRIVATE dynamic_foam)

# Need imgui backends headers for main.cpp
target_include_directories(app PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Temporary tutorial application for ImGui usage
add_executable(tutorial imgui_tutorial.cpp)
target_link_libraries(tutorial PRIVATE dynamic_foam)
target_include_directories(tutorial PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)