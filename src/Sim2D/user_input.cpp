#include "dynamic_foam/sim2d/user_input.h"

namespace DynamicFoam {
    namespace Sim2D {
        UserInput PollUserInput() {
            UserInput input;
            ImGuiIO& io = ImGui::GetIO();
            input.mouse_pos = io.MousePos;
            input.left_mouse_clicked = io.MouseClicked[0];
            return input;
        }
    }
}
