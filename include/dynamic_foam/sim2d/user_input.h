#pragma once

#include "imgui.h"

namespace DynamicFoam::Sim2D{

// Simple cursor-only input for 2D simulation
struct UserInput {
    ImVec2 mouse_pos;
    bool left_mouse_clicked;
};

UserInput PollUserInput();
}
