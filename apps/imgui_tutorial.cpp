#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>

/*
 * IMGUI + GLFW CRASH COURSE
 * 
 * IMMEDIATE MODE GUI CONCEPT:
 * ============================
 * Unlike traditional retained-mode GUIs (like Qt/WinForms) where you create
 * widgets once and update their state, ImGUI is "immediate mode":
 * 
 * - You describe the entire UI every frame (60fps+)
 * - State is managed by YOUR code, not the library
 * - Widgets don't exist between frames; they're recreated each frame
 * - This makes it simple for dynamic, data-driven UIs
 * 
 * Example: A button doesn't "exist" until you call ImGui::Button() in that frame.
 * The library returns true if it was clicked during this frame.
 * 
 * GLFW:
 * =====
 * GLFW handles:
 * - Window creation and management
 * - OpenGL context setup
 * - Input events (keyboard, mouse, joystick)
 * - Monitor and display management
 * 
 * WORKFLOW:
 * =========
 * 1. Initialize GLFW and create window
 * 2. Initialize OpenGL
 * 3. Initialize ImGUI and its backends (GLFW + OpenGL)
 * 4. Main loop:
 *    a. Poll input events (GLFW handles this)
 *    b. Start ImGUI frame (ImGUI processes queued input)
 *    c. Describe your UI with ImGui calls
 *    d. Render ImGUI
 *    e. Swap buffers
 */

int main() {
    // ============================================================================
    // STEP 1: Initialize GLFW
    // ============================================================================
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Request OpenGL 3.3 core profile
    const char* glsl_version = "#version 330";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // ============================================================================
    // STEP 2: Create GLFW Window
    // ============================================================================
    GLFWwindow* window = glfwCreateWindow(1280, 720, "ImGUI + GLFW Tutorial", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the OpenGL context current
    glfwMakeContextCurrent(window);
    
    // Enable vsync (swap interval of 1 = 60fps on 60Hz monitor)
    glfwSwapInterval(1);

    // ============================================================================
    // STEP 3: Setup ImGUI
    // ============================================================================
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    ImGui::StyleColorsDark();  // Use dark theme

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);  // Enable input processing
    ImGui_ImplOpenGL3_Init(glsl_version);

    // ============================================================================
    // UI STATE: Your application maintains state between frames
    // ============================================================================
    float slider_value = 50.0f;           // Slider will range 0-100
    bool checkbox_state = false;           // Checkbox on/off
    int dropdown_selection = 0;            // Index of selected dropdown option
    const char* dropdown_items[] = { "Option A", "Option B", "Option C", "Option D" };
    
    ImVec2 dragged_position = ImVec2(100, 100);  // Position of draggable object
    bool is_dragging = false;
    
    float color[3] = { 1.0f, 0.5f, 0.2f };  // RGB color for demo

    // ============================================================================
    // MAIN LOOP
    // ============================================================================
    while (!glfwWindowShouldClose(window)) {
        // ========================================================================
        // INPUT PROCESSING: GLFW collects events and queues them
        // ========================================================================
        glfwPollEvents();

        // ========================================================================
        // START IMGUI FRAME: Tell ImGUI a new frame is starting
        // ========================================================================
        // ImGui processes the queued input events and updates mouse/keyboard state
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // ========================================================================
        // BUILD UI: Describe your UI here (called every frame)
        // ========================================================================

        // Create a simple window
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(500, 500), ImGuiCond_FirstUseEver);
        ImGui::Begin("Interactive Demo");

        ImGui::Text("FPS: %.1f", io.Framerate);
        ImGui::Separator();

        // ========================================================================
        // SLIDERS
        // ========================================================================
        ImGui::SliderFloat("Slider (0-100)", &slider_value, 0.0f, 100.0f);
        ImGui::Text("Slider value: %.1f", slider_value);

        // ========================================================================
        // BUTTONS
        // ========================================================================
        // ImGui::Button returns true if CLICKED this frame (not held down)
        if (ImGui::Button("Click Me!", ImVec2(100, 25))) {
            slider_value = 50.0f;  // Reset slider
            std::cout << "Button was clicked!" << std::endl;
        }

        ImGui::SameLine();  // Place next widget on same line
        
        static bool show_demo = false;
        if (ImGui::Button("Toggle Alert")) {
            show_demo = !show_demo;
        }
        
        if (show_demo) {
            ImGui::TextColored(ImVec4(1, 1, 0, 1), "Alert: Button was toggled!");
        }

        ImGui::Separator();

        // ========================================================================
        // CHECKBOXES
        // ========================================================================
        ImGui::Checkbox("Enable Feature", &checkbox_state);
        
        if (checkbox_state) {
            ImGui::Text("Feature is enabled!");
        }

        ImGui::Separator();

        // ========================================================================
        // DROPDOWNS (Combo)
        // ========================================================================
        // ImGui::Combo creates a dropdown selector
        // Args: label, current index pointer, item array, item count
        if (ImGui::Combo("Choose Option##combo", &dropdown_selection, dropdown_items, 4)) {
            std::cout << "Selected: " << dropdown_items[dropdown_selection] << std::endl;
        }
        ImGui::Text("Currently selected: %s", dropdown_items[dropdown_selection]);

        ImGui::Separator();

        // ========================================================================
        // COLOR PICKER
        // ========================================================================
        ImGui::ColorEdit3("Color Picker", color);

        ImGui::Separator();

        // ========================================================================
        // MOUSE INPUT & DRAGGING
        // ========================================================================
        ImGui::Text("Mouse Position: (%.0f, %.0f)", io.MousePos.x, io.MousePos.y);
        ImGui::Text("Left Mouse: %s", io.MouseDown[0] ? "PRESSED" : "Released");
        ImGui::Text("Right Mouse: %s", io.MouseDown[1] ? "PRESSED" : "Released");

        // Check if left mouse button was just pressed
        if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImGui::TextColored(ImVec4(0, 1, 0, 1), "Left mouse is being held!");
        }

        // Check if left mouse button was just clicked (this frame)
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
            std::cout << "Left click detected at frame!" << std::endl;
        }

        ImGui::Separator();

        // ========================================================================
        // DRAGGABLE ELEMENT EXAMPLE
        // ========================================================================
        ImGui::Text("Draggable Demo:");
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Try dragging the colored area below");
        
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size = ImGui::GetContentRegionAvail();
        
        if (canvas_size.x < 50.0f) canvas_size.x = 50.0f;
        if (canvas_size.y < 50.0f) canvas_size.y = 50.0f;

        // Draw canvas background
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        draw_list->AddRectFilled(
            canvas_pos,
            ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
            IM_COL32(50, 50, 50, 255)
        );

        // Draw draggable square
        ImVec2 square_size = ImVec2(40, 40);
        ImVec2 square_min = ImVec2(canvas_pos.x + dragged_position.x, canvas_pos.y + dragged_position.y);
        ImVec2 square_max = ImVec2(square_min.x + square_size.x, square_min.y + square_size.y);

        draw_list->AddRectFilled(square_min, square_max, IM_COL32(255, 100, 100, 255));
        draw_list->AddRect(square_min, square_max, IM_COL32(255, 255, 255, 255));

        // Handle dragging interaction
        ImGui::SetCursorScreenPos(canvas_pos);
        ImGui::InvisibleButton("drag_canvas", canvas_size);
        
        if (ImGui::IsItemHovered() && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            // Check if click is within the square
            ImVec2 mouse_pos = io.MousePos;
            if (mouse_pos.x >= square_min.x && mouse_pos.x <= square_max.x &&
                mouse_pos.y >= square_min.y && mouse_pos.y <= square_max.y) {
                is_dragging = true;
            }
        } else {
            is_dragging = false;
        }

        // Update position while dragging
        if (is_dragging && ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            ImVec2 delta = io.MouseDelta;
            dragged_position.x = std::max(0.0f, std::min(dragged_position.x + delta.x, canvas_size.x - square_size.x));
            dragged_position.y = std::max(0.0f, std::min(dragged_position.y + delta.y, canvas_size.y - square_size.y));
        }

        ImGui::End();

        // ========================================================================
        // RENDER PHASE
        // ========================================================================
        // Tell ImGUI to prepare rendering data
        ImGui::Render();

        // Clear the screen with background color
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Render ImGUI draw data
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Swap front and back buffers (display rendered frame)
        glfwSwapBuffers(window);
    }

    // ============================================================================
    // CLEANUP
    // ============================================================================
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
