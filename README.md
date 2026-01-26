# Dynamic Foam

Radiant Foam Physics Engine using C++ and CUDA.

## Prerequisites

- **CMake** 3.20 or higher
- **Visual Studio** (with C++ tools)
- **CUDA Toolkit** 11.0 or higher
- **vcpkg** (for dependency management)

## Setup Instructions

### 1. Install Dependencies with vcpkg

From the project root directory, run:

```bash
vcpkg install
```

This will install all dependencies listed in `vcpkg.json` (glfw3, imgui, etc.) and create a `vcpkg_installed` directory locally.

### 2. Configure the Project with CMake

```bash
cmake --preset default
```

This configures the project and generates build files in the `build/` directory.

### 3. Build the Project

```bash
cmake --build build
```

Or use your IDE (Visual Studio, etc.) to build the solution.

## CUDA Setup

> TODO: Add CUDA-specific configuration and build instructions

## Project Structure

- `src/` - Source files and library implementation
- `include/` - Header files
- `build/` - Build artifacts (generated, not in git)
- `vcpkg_installed/` - vcpkg-managed dependencies (generated, not in git)

## Notes

- Dependencies are installed locally via vcpkg manifest mode
- Each developer needs to run `vcpkg install` after cloning
- The project uses CMake presets for consistent configuration
