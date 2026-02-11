#pragma once

#include "imgui.h"

namespace DynamicFoam {
    namespace Sim2D {
        struct UserInput {
            ImVec2 mouse_pos;
            bool left_mouse_clicked;
        };

        UserInput PollUserInput();
    }
}
