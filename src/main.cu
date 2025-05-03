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

#define BLOCK_SIZE 256

__constant__ vec3 g_pixel00_loc;
__constant__ vec3 g_camera_center;
__constant__ vec3 g_pixel_delta_u;
__constant__ vec3 g_pixel_delta_v;

__device__ color ray_color(const ray& r, const device_hittable_list* world) {
    hit_record rec;

    // object's normal gradient
    if (world->hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }

    // sky gradient
    vec3 unit_direction = unit_vector(r.direction());
    auto a = 0.5 * (unit_direction.y() + 1.0);
    return (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
}

__global__ void generate_frame(unsigned int *buffer, int width, int height, device_hittable_list* world) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int i = idx % width;
        int j = idx / width;
        int buffer_idx = idx * 3;

        auto pixel_center = g_pixel00_loc + (i * g_pixel_delta_u) 
                                          + (j * g_pixel_delta_v);
        auto ray_direction = pixel_center - g_camera_center;
        ray r(g_camera_center, ray_direction);

        color pixel_color = ray_color(r, world);

        write_color(buffer, buffer_idx, pixel_color);
    }
}


int main() {
    // Image
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 400;
    int image_height = int(image_width / aspect_ratio);
    image_height = (image_height < 1) ? 1 : image_height;
    int total_pixels = image_width * image_height;

    // Camera
    auto focal_length = 1.0;
    auto viewport_height = 2.0;
    auto viewport_width = viewport_height * 
                            (float(image_width) / image_height);
    auto camera_center = point3(0, 0, 0);

    // viewport vectors
    auto viewport_u = vec3(viewport_width, 0, 0);
    auto viewport_v = vec3(0, -viewport_height, 0);

    // pixel delta
    auto pixel_delta_u = viewport_u / image_width;
    auto pixel_delta_v = viewport_v / image_height;

    // viewport upper left corner
    auto viewport_upper_left = camera_center
                                - vec3(0, 0, focal_length)
                                - (viewport_u / 2)
                                - (viewport_v / 2);

    auto pixel00_loc = viewport_upper_left
                        + (0.5 * (pixel_delta_u + pixel_delta_v));
    
    int buffer_size = total_pixels * 3;
    cudaEvent_t start, stop;
    cudaStream_t stream1;

    // (256 * 256 + 128 - 1) / 256 = 256 blocks
    int total_blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // The Host's frame buffer
    unsigned int *frame_buffer;

    // Device's frame buffer
    unsigned int *device_frame_buffer;

    // initialize world

    // Instead of creating objects on host:
    device_hittable_list** d_world_ptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_world_ptr, sizeof(device_hittable_list*)));

    // Create all objects on device
    setup_world<<<1,1>>>(d_world_ptr);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Get the device world pointer back to host
    device_hittable_list* d_world;
    CHECK_CUDA_ERROR(cudaMemcpy(&d_world, d_world_ptr, sizeof(device_hittable_list*), cudaMemcpyDeviceToHost));

    // device_hittable_list **d_world_ptr;
    // CHECK_CUDA_ERROR(cudaMalloc(&d_world_ptr, sizeof(device_hittable_list*)));
    // setup_world<<<1, 1>>>(d_world_ptr);

    // CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // device_hittable_list *world;
    // CHECK_CUDA_ERROR(cudaMallocManaged(&world, sizeof(device_hittable_list)));
    // new (world) device_hittable_list();
    // world->allocate(2);
    
    // // add objects in world
    // sphere* sphere1;
    // CHECK_CUDA_ERROR(cudaMallocManaged(&sphere1, sizeof(sphere)));
    // new (sphere1) sphere(point3(0, 0, -1), 0.5);
    // world->add(sphere1);

    // sphere* sphere2;
    // CHECK_CUDA_ERROR(cudaMallocManaged(&sphere2, sizeof(sphere)));
    // new (sphere2) sphere(point3(0, -100.5, -1), 100);
    // world->add(sphere2);




    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream1));

    // Copy copy constant values to device memory
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_pixel00_loc, &pixel00_loc, sizeof(pixel00_loc)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_camera_center, &camera_center, sizeof(camera_center)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_pixel_delta_u, &pixel_delta_u, sizeof(pixel_delta_u)));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_pixel_delta_v, &pixel_delta_v, sizeof(pixel_delta_v)));

    
    CHECK_CUDA_ERROR(cudaMallocHost(&frame_buffer, sizeof(int) * buffer_size));
    CHECK_CUDA_ERROR(cudaMallocAsync(&device_frame_buffer, sizeof(int) * buffer_size, stream1));

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    generate_frame<<<total_blocks, BLOCK_SIZE, 0, stream1>>>(device_frame_buffer, 
                                                             image_width, image_height,
                                                             d_world);

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(frame_buffer, device_frame_buffer, 
                sizeof(int) * buffer_size, cudaMemcpyDeviceToHost, stream1));

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            int idx = (j * image_width + i) * 3;
            std::cout << frame_buffer[idx] << ' ' << frame_buffer[idx + 1] << ' ' << frame_buffer[idx + 2] << '\n';
        }
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream1));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA_ERROR(cudaFreeHost(frame_buffer));
    CHECK_CUDA_ERROR(cudaFree(device_frame_buffer));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    std::clog << "Execution time: " << milliseconds << " milliseconds" << std::endl;

    return 0;
}