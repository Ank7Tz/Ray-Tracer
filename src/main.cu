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

    // material init

    auto h_material_ground = lambertian(color(0.8, 0.8, 0.0));
    auto h_material_center = lambertian(color(0.1, 0.2, 0.5));
    auto h_material_left = metal(color(0.8, 0.8, 0.8));
    auto h_material_right = metal(color(0.8, 0.6, 0.2));

    material *d_material_ground, *d_material_center;
    material *d_material_left, *d_material_right;

    CHECK_CUDA_ERROR(cudaMalloc(&d_material_ground, sizeof(lambertian)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_material_ground, &h_material_ground, sizeof(lambertian), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc(&d_material_center, sizeof(lambertian)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_material_center, &h_material_center, sizeof(lambertian), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc(&d_material_left, sizeof(lambertian)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_material_left, &h_material_left, sizeof(lambertian), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc(&d_material_right, sizeof(lambertian)));
    CHECK_CUDA_ERROR(cudaMemcpy(d_material_right, &h_material_right, sizeof(lambertian), cudaMemcpyHostToDevice));

    sphere* h_sphere1 = new sphere(point3(0, -100.5, -1), 100.0, &h_material_ground);
    sphere* h_sphere2 = new sphere(point3(0, 0, -1.2), 0.5, &h_material_center);
    sphere* h_sphere3 = new sphere(point3(-1.0, 0.0, -1.0), 0.5, &h_material_left);
    sphere* h_sphere4 = new sphere(point3(1.0, 0.0, -1.0), 0.5, &h_material_left);

    h_world.add(h_sphere1);
    h_world.add(h_sphere2);
    h_world.add(h_sphere3);
    h_world.add(h_sphere4);

    size_t stackSize = 0;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::clog << stackSize << std::endl;

    // To increase the stack size (e.g., to 16KB):
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);

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