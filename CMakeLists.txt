cmake_minimum_required(VERSION 3.10)
project(Ray-Tracer LANGUAGES CUDA CXX)

# set build type to debug if not set
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

# CUDA compiler options for debug
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build)

# Create the ray tracer executable
add_executable(ray_tracer src/main.cu)
set_target_properties(ray_tracer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Custom targets to match original Makefile functionality
add_custom_target(rebuild
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
  COMMENT "Rebuilding all targets"
)