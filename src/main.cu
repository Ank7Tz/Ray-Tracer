#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include <math.h>
#include "utils.h"
#include "hittable.h"
#include "hittable_list.h"
#include "sphere.h"
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

    device_hittable_list* d_world = h_world.create_device_copy();

    camera cam;
    cam.aspect_ratio = aspect_ratio;
    cam.focal_length = 1.0;
    cam.image_width = image_width;

    cam.render(d_world);

    h_world.free_device_world(d_world);

    return 0;
}