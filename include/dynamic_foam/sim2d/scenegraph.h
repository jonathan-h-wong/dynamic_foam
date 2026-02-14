#pragma once
#include <unordered_map>
#include <stdexcept>
#include "dynamic_foam/sim2d/foam.h"

namespace DynamicFoam::Sim2D {
    class SceneGraph {
    public:
        std::unordered_map<int, Foam> foams;
        std::unordered_map<int, glm::mat4> worldTransforms;
        std::unordered_map<int, bool> isController;
        std::unordered_map<int, bool> isDynamic;

        SceneGraph(
            const std::unordered_map<int, Foam>& foams_map,
            const std::unordered_map<int, glm::mat4>& transforms,
            const std::unordered_map<int, bool>& controller_map,
            const std::unordered_map<int, bool>& dynamic_map
        ) : foams(foams_map),
            worldTransforms(transforms),
            isController(controller_map),
            isDynamic(dynamic_map) {
            validate();
        }

    private:
        /*
        Controller | Dynamic
        Y        Y       Not allowed, validated here
        Y        N       Controller foam
        N        Y       Dynamic foam
        N        N       Static foam
        */
        void validate() {
            for (const auto& [id, controller] : isController) {
                if (controller) {
                    auto dynamicIt = isDynamic.find(id);
                    if (dynamicIt != isDynamic.end() && dynamicIt->second) {
                        throw std::runtime_error("A foam cannot be both a controller and dynamic.");
                    }
                }
            }
        }
    };
}
