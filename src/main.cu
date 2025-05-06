#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include <math.h>
#include "utils.h"
#include "hittable.h"
#include "camera.h"

int main() {
    // Image
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;

    // initialize world
    host_hittable_list h_world;

    sphere* h_sphere1 = new sphere(point3(0, 0, -1), 0.5);
    sphere* h_sphere2 = new sphere(point3(0, -100.5, -1), 100);

    sphere* d_sphere1;
    CHECK_CUDA_ERROR(cudaMalloc(&d_sphere1, sizeof(sphere)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sphere1, h_sphere1, sizeof(sphere), cudaMemcpyHostToDevice));

    sphere* d_sphere2;
    CHECK_CUDA_ERROR(cudaMalloc(&d_sphere2, sizeof(sphere)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_sphere2, h_sphere2, sizeof(sphere), cudaMemcpyHostToDevice));

    h_world.add(d_sphere1);
    h_world.add(d_sphere2);

    size_t stackSize = 0;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::clog << stackSize << std::endl;

    // To increase the stack size (e.g., to 16KB):
    cudaDeviceSetLimit(cudaLimitStackSize, 4 * 16384);

    // Verify the new stack size:
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::clog << stackSize << std::endl;

    device_hittable_list* d_world = h_world.create_device_copy();

    camera cam;
    cam.aspect_ratio = aspect_ratio;
    cam.focal_length = 1.0;
    cam.image_width = image_width;
    cam.max_depth = 50;
    cam.samples_per_pixel = 100;

    cudaEvent_t start, stop;
    float milliseconds = 0;

    // Create events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cam.render(d_world);

    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    std::clog << "Rendering time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    h_world.free_device_world(d_world);

    return 0;
}