// test_renderer.cpp
// Launches a GLFW window, runs the sample scene through the full simulation
// pipeline each frame, and blits the CUDA render output to the screen via an
// OpenGL texture displayed in an ImGui fullscreen window.

#include <GLFW/glfw3.h>
// GL_CLAMP_TO_EDGE is OpenGL 1.2+; Windows SDK gl.h only covers 1.1.
#ifndef GL_CLAMP_TO_EDGE
#  define GL_CLAMP_TO_EDGE 0x812F
#endif
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <chrono>
#include <iostream>
#include <vector>

#include "dynamic_foam/Sim2D/sample_scene.h"
#include "dynamic_foam/Sim2D/simulation.h"
#include "dynamic_foam/Sim2D/user_input.h"

using namespace DynamicFoam::Sim2D;

// ---------------------------------------------------------------------------
// GLFW error callback
// ---------------------------------------------------------------------------
static void glfwErrorCallback(int error, const char* description) {
    std::cerr << "GLFW Error " << error << ": " << description << "\n";
}

int main() {
    constexpr int WIDTH  = 1280;
    constexpr int HEIGHT = 720;

    // -----------------------------------------------------------------------
    // GLFW + OpenGL context
    // -----------------------------------------------------------------------
    glfwSetErrorCallback(glfwErrorCallback);
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Dynamic Foam – Sample Scene", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // vsync

    // -----------------------------------------------------------------------
    // ImGui setup (must happen before the first NewFrame)
    // -----------------------------------------------------------------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    // -----------------------------------------------------------------------
    // Build scene and initialise simulator
    // -----------------------------------------------------------------------
    SceneGraph sceneGraph = createSampleSceneGraph();
    Simulation sim(sceneGraph, glm::ivec2(WIDTH, HEIGHT));

    // -----------------------------------------------------------------------
    // Host pixel buffer used for GPU → CPU copy each frame
    // -----------------------------------------------------------------------
    std::vector<glm::vec4> hostPixels(WIDTH * HEIGHT);

    // -----------------------------------------------------------------------
    // OpenGL texture that receives the rendered frame
    // -----------------------------------------------------------------------
    GLuint renderTex = 0;
    glGenTextures(1, &renderTex);
    glBindTexture(GL_TEXTURE_2D, renderTex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, WIDTH, HEIGHT, 0, GL_RGBA, GL_FLOAT, nullptr);
    glBindTexture(GL_TEXTURE_2D, 0);

    // -----------------------------------------------------------------------
    // Main loop
    // -----------------------------------------------------------------------
    auto lastTime = std::chrono::high_resolution_clock::now();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // Compute delta time and clamp to avoid spiral-of-death
        auto now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastTime).count();
        lastTime  = now;
        dt = std::min(dt, 0.05f);

        // Start ImGui frame (PollUserInput reads ImGui IO, so call after NewFrame)
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Poll input and advance simulation
        UserInput input = PollUserInput();
        sim.step(input, dt);

        // Copy CUDA device output buffer → host → OpenGL texture
        const glm::vec4* d_output = sim.deviceOutputBuffer();
        if (d_output) {
            cudaError_t err = cudaMemcpy(
                hostPixels.data(),
                d_output,
                WIDTH * HEIGHT * sizeof(glm::vec4),
                cudaMemcpyDeviceToHost);

            if (err != cudaSuccess) {
                std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
            }

            glBindTexture(GL_TEXTURE_2D, renderTex);
            // OpenGL row 0 is bottom-left; flip UV in ImGui::Image instead of
            // re-ordering pixels to keep this copy cheap.
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, WIDTH, HEIGHT, GL_RGBA, GL_FLOAT, hostPixels.data());
            glBindTexture(GL_TEXTURE_2D, 0);
        }

        // Fullscreen render output window (no decorations, sits behind HUD)
        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImVec2(static_cast<float>(WIDTH), static_cast<float>(HEIGHT)));
        ImGui::Begin("##render", nullptr,
            ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_NoMove       |
            ImGuiWindowFlags_NoBackground |
            ImGuiWindowFlags_NoBringToFrontOnFocus);

        // No UV flip needed: glTexSubImage2D places CUDA row 0 at GL v=0 (texture
        // bottom), and ImGui maps screen-top to uv v=0, so the two conventions
        // cancel and the image is displayed right-side-up without any extra flip.
        ImGui::Image(
            (ImTextureID)(uintptr_t)renderTex,
            ImVec2(static_cast<float>(WIDTH), static_cast<float>(HEIGHT)),
            ImVec2(0, 0), ImVec2(1, 1));

        ImGui::End();

        // Small HUD overlay
        ImGui::SetNextWindowPos(ImVec2(8, 8));
        ImGui::SetNextWindowBgAlpha(0.6f);
        ImGui::Begin("##hud", nullptr,
            ImGuiWindowFlags_NoDecoration |
            ImGuiWindowFlags_NoMove       |
            ImGuiWindowFlags_AlwaysAutoResize);
        ImGui::Text("Dynamic Foam – Sample Scene");
        ImGui::Separator();
        ImGui::Text("%.1f FPS  (%.2f ms)", 1.0f / dt, dt * 1000.0f);
        ImGui::Separator();
        // --- Camera projection ---
        ImGui::Text("Camera");
        int projType = static_cast<int>(sim.camera_.type);
        if (ImGui::RadioButton("Orthographic", &projType, static_cast<int>(ProjectionType::Orthographic)))
            sim.camera_.type = ProjectionType::Orthographic;
        ImGui::SameLine();
        if (ImGui::RadioButton("Perspective", &projType, static_cast<int>(ProjectionType::Perspective)))
            sim.camera_.type = ProjectionType::Perspective;
        if (sim.camera_.type == ProjectionType::Orthographic) {
            ImGui::SliderFloat("Width", &sim.camera_.width, 0.5f, 20.0f);
        } else {
            float fovDeg = glm::degrees(sim.camera_.fovY);
            if (ImGui::SliderFloat("FoV", &fovDeg, 5.0f, 120.0f))
                sim.camera_.fovY = glm::radians(fovDeg);
        }
        ImGui::Separator();
        ImGui::Text("Overlays");

        // --- Cell centers ---
        ImGui::Checkbox("Cell centers", &sim.overlayParams.show_centers);
        if (sim.overlayParams.show_centers) {
            // glm::vec4 is contiguous floats — safe to pass as float[4]
            ImGui::ColorEdit4("Center color", &sim.overlayParams.center_color.x,
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
        }

        // --- Cell edges ---
        ImGui::Checkbox("Cell edges", &sim.overlayParams.show_edges);
        if (sim.overlayParams.show_edges) {
            ImGui::ColorEdit4("Edge color", &sim.overlayParams.edge_color.x,
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_AlphaBar);
        }

        ImGui::End();

        // Composite and present
        ImGui::Render();
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // -----------------------------------------------------------------------
    // Cleanup
    // -----------------------------------------------------------------------
    glDeleteTextures(1, &renderTex);

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
