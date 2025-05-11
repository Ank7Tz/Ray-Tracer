#ifndef CAMERA_H
#define CAMERA_H

#include "hittable.h"
#include "color.h"
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "utils.h"

#define BLOCK_SIZE 256

__constant__ vec3 g_pixel00_loc;
__constant__ vec3 g_camera_center;
__constant__ vec3 g_pixel_delta_u;
__constant__ vec3 g_pixel_delta_v;
__constant__ float g_pixel_sample_scale;
__constant__ int g_samples_per_pixel;
__constant__ int g_max_depth;
__constant__ float g_defocus_angle;
__constant__ point3 g_defocus_u;
__constant__ point3 g_defocus_v;

__device__ color ray_color(const ray& r, const device_hittable_list* world, curandState* state) {
    ray current_ray = r;
    color current_attenuation(1.0, 1.0, 1.0);
    color final_color(0.00, 0.00, 0.00);

    for (int depth = 0; depth < g_max_depth; depth++) {
        hit_record rec;
        if (!world->hit(current_ray, interval(0.001, infinity), rec)) {
            vec3 unit_direction = unit_vector(current_ray.direction());
            auto a = 0.5 * (unit_direction.y() + 1.0);
            color surrounding = (1.0 - a) * color(1.0, 1.0, 1.0) + a * color(0.5, 0.7, 1.0);
            return final_color + current_attenuation * surrounding;
        }

        ray scattered;
        color attenuation;

        if (rec.mat->scatter(current_ray, rec, attenuation, scattered, state)) {
            current_attenuation = current_attenuation * attenuation;

            current_ray = scattered;
        } else {
            return final_color;            
        }
    }

    return final_color;
}

__device__ vec3 sample_square(curandState* state) {
    return vec3(random_float(state) - 0.5, random_float(state) - 0.5, 0);
}

__device__ point3 defocus_disk_sample(curandState* state) {
    auto p = random_in_unit_disk(state);
    return g_camera_center + (p[0] * g_defocus_u) + (p[1] * g_defocus_v);
}


__device__ ray get_ray(int i, int j, curandState* state) {
    auto offset = sample_square(state);
    auto pixel_sample = g_pixel00_loc
                        + ((i + offset.x()) * g_pixel_delta_u)
                        + ((j + offset.y()) * g_pixel_delta_v);

    auto ray_origin = (g_defocus_angle <= 0) ? g_camera_center : defocus_disk_sample(state);
    auto ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}

__global__ void generate_frame(int *buffer, int width, int height, device_hittable_list* world, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int i = idx % width;
        int j = idx / width;
        int buffer_idx = idx * 3;

        color pixel_color(0, 0, 0);

        for (int sample = 0; sample < g_samples_per_pixel; sample++) {
            ray r = get_ray(i, j, &states[idx]);
            pixel_color += ray_color(r, world, &states[idx]);
        }

        write_color(buffer, buffer_idx, g_pixel_sample_scale * pixel_color);
    }
}


class camera {
    public:
        float aspect_ratio = 1.0;
        int image_width = 100;
        int focal_length = 1.0;
        int samples_per_pixel = 10;
        int max_depth = 10;

        float vfov = 90;
        point3 lookfrom = point3(0, 0, 0);
        point3 lookat = point3(0, 0, -1);
        vec3 vup = vec3(0, 1, 0); // camera relative up

        float defocus_angle = 0; // variation angle of rays through each pixel
        float focus_dist = 10; // distance from camera lookfrom point to plane of perfect focus;

        void render(device_hittable_list* d_world) {
            initialize();

            curandState* d_states;
            
            CHECK_CUDA_ERROR(cudaMalloc((void**) &d_states, total_pixels * sizeof(curandStateXORWOW)));

            init_random_states<<<total_blocks, BLOCK_SIZE>>>(d_states, time(NULL), total_pixels);

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            generate_frame<<<total_blocks, BLOCK_SIZE>>>(device_frame_buffer, 
                image_width, image_height,
                d_world, d_states);

            CHECK_CUDA_ERROR(cudaDeviceSynchronize());

            CHECK_CUDA_ERROR(cudaMemcpy(frame_buffer, device_frame_buffer, 
                        sizeof(int) * buffer_size, cudaMemcpyDeviceToHost));

            std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

            for (int j = 0; j < image_height; j++) {
                for (int i = 0; i < image_width; i++) {
                    int idx = (j * image_width + i) * 3;
                    std::cout << frame_buffer[idx] << ' ' << frame_buffer[idx + 1] << ' ' << frame_buffer[idx + 2] << '\n';
                }
            }
        }

    private:
        int image_height;
        int total_blocks;
        int total_pixels;
        int buffer_size;
        // The Host's frame buffer
        int *frame_buffer;
        // Device frame buffer pointer
        int *device_frame_buffer;
        vec3 u, v, w;
        vec3 defocus_disk_u; // horizontal radius of defocus disk
        vec3 defocus_disk_v; // vertical radius of defocus disk

        void initialize() {
            image_height = int(image_width / aspect_ratio);
            image_height = (image_height < 1) ? 1 : image_height;
            total_pixels = image_width * image_height;

            auto theta = degrees_to_radians(vfov);
            auto h = tanf(theta/2);

            buffer_size = total_pixels * 3;

            frame_buffer = new int[buffer_size];

            // (256 * 256 + 128 - 1) / 256 = 256 blocks
            total_blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE;

            auto camera_center = lookfrom;
            // focal_length = (lookfrom - lookat).length();

            // viewport vectors
            auto viewport_height = 2 * h * focus_dist;
            auto viewport_width = viewport_height * 
                                    (float(image_width) / image_height);

            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);

            auto viewport_u = viewport_width * u;
            auto viewport_v = viewport_height * -v;

            
            // pixel delta
            auto pixel_delta_u = viewport_u / image_width;
            auto pixel_delta_v = viewport_v / image_height;

            // viewport upper left corner
            auto viewport_upper_left = camera_center
            - (focus_dist * w)
            - (viewport_u / 2)
            - (viewport_v / 2);

            auto pixel00_loc = viewport_upper_left
            + (0.5 * (pixel_delta_u + pixel_delta_v));

            float pixel_sample_scale = 1.0 / samples_per_pixel;

            interval intensity(0.000, 0.999);

            auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
            defocus_disk_u = u * defocus_radius;
            defocus_disk_v = v * defocus_radius;

            // setting up constants
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_pixel00_loc, &pixel00_loc, sizeof(pixel00_loc)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_camera_center, &camera_center, sizeof(camera_center)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_pixel_delta_u, &pixel_delta_u, sizeof(pixel_delta_u)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_pixel_delta_v, &pixel_delta_v, sizeof(pixel_delta_v)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_pixel_sample_scale, &pixel_sample_scale, sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_samples_per_pixel, &samples_per_pixel, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_intensity, &intensity, sizeof(interval)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_max_depth, &max_depth, sizeof(int)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_defocus_angle, &defocus_angle, sizeof(float)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_defocus_u, &defocus_disk_u, sizeof(point3)));
            CHECK_CUDA_ERROR(cudaMemcpyToSymbol(g_defocus_u, &defocus_disk_v, sizeof(point3)));
            
            // Device's frame buffer memory allocation
            CHECK_CUDA_ERROR(cudaMalloc(&device_frame_buffer, sizeof(int) * buffer_size));
        }
};

#endif