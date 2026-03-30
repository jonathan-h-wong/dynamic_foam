#include "dynamic_foam/Sim2D/user_input.h"

namespace DynamicFoam {
    namespace Sim2D {
        UserInput PollUserInput() {
            UserInput input;
            ImGuiIO& io = ImGui::GetIO();
            input.mouse_pos = io.MousePos;
            input.left_mouse_clicked = io.MouseClicked[0];
            input.key_w = ImGui::IsKeyDown(ImGuiKey_W);
            input.key_a = ImGui::IsKeyDown(ImGuiKey_A);
            input.key_s = ImGui::IsKeyDown(ImGuiKey_S);
            input.key_d = ImGui::IsKeyDown(ImGuiKey_D);
            return input;
        }
    }
}
