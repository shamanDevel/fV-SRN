# Basic Volume Renderer (Isosurface + DVR)

A simple volume renderer as a baseline for various projects in my Ph.D. and several Bachelor+Master theses.

Currently supported:
 - Iso-surface rendering 
 - Screen-space shading
 - Direct Volume Rendering

Additional datasets can be found at https://gitlab.com/shaman42/basic-volume-renderer-data

## Project structure:
 - renderer: a shared or static library exposing the rendering core (C++, CUDA)
 - gui: interactive gui for the renderer, allows to test all available options (C++, OpenGL)

## Requirements

 - CUDA 11.0
 - OpenGL with GLFW and GLM

Tested with CUDA 11.0, Windows 10, Visual Studio 2019
