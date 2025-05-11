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

host_hittable_list *generate_world() {
    host_hittable_list *h_world = new host_hittable_list();
    material *ground_material = new lambertian(color(0.5, 0.5, 0.5));
    sphere *ground_sphere = new sphere(point3(0, -1000, 0), 1000, ground_material);
    h_world->add(ground_sphere);

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = host_random_float();
            auto height = host_random_float(0.1, 0.3);
            point3 *center = new point3(a + 0.9 * host_random_float(), height, b + 0.9 * host_random_float());
            if (((*center) - point3(4, 0.2, 0)).length() > 0.9) {
                material *sphere_material;
                if (choose_mat < 0.8) {
                    // diffuse material
                    auto albedo = color::random() * color::random();
                    sphere_material = new lambertian(albedo);
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = host_random_float(0, 0.5);
                    sphere_material = new metal(albedo, fuzz);
                } else {
                    sphere_material = new dielectric(1.5);
                }

                sphere *object = new sphere(*center, height, sphere_material);
                h_world->add(object);
            }
        }
    }

    return h_world;
}

int main() {
    // Image
    auto aspect_ratio = 16.0 / 9.0;
    int image_width = 1200;

    // // initialize world
    host_hittable_list *h_world = new host_hittable_list();

    // // material init
    // auto h_material_ground = lambertian(color(0.5, 0.5, 0.5));
    // auto h_material_center = lambertian(color(0.1, 0.2, 0.5));
    // auto h_material_left = dielectric(1.50);
    // auto material_bubble = dielectric(1.00 / 1.50);
    // auto h_material_right = metal(color(0.8, 0.6, 0.2), 1.0);

    // sphere* h_sphere1 = new sphere(point3(0, -100.5, -1.0), 100.0, &h_material_ground);
    // sphere* h_sphere2 = new sphere(point3(0, 0, -2.2), 0.5, &h_material_center);
    // sphere* h_sphere3 = new sphere(point3(-1.0, 0.0, -1.0), 0.5, &h_material_left);
    // sphere* h_sphere4 = new sphere(point3(1.0, 0.0, 1.0), 0.5, &h_material_right);
    // sphere* h_sphere5 = new sphere(point3(-1.0, 0.0, -1.0), 0.4, &material_bubble);

    // h_world->add(h_sphere1);
    // h_world->add(h_sphere2);
    // h_world->add(h_sphere3);
    // h_world->add(h_sphere4);
    // h_world->add(h_sphere5);

    // generate world (random)
    h_world = generate_world();

    auto material1 = dielectric(1.5);
    h_world->add(new sphere(point3(0, 1, 0), 1.0, &material1));

    auto material2 = lambertian(color(0.4, 0.2, 0.1));
    h_world->add(new sphere(point3(-4, 1, 0), 1.0, &material2));

    auto material3 = metal(color(0.7, 0.6, 0.5), 0.0);
    h_world->add(new sphere(point3(4, 1, 0), 1.0, &material3));

    size_t stackSize = 0;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::clog << "intial stack frame(size):\t" << stackSize << std::endl;

    // To increase the stack size (e.g., to 64KB):
    cudaDeviceSetLimit(cudaLimitStackSize, 2 * 16384);

    // Verify the new stack size:
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::clog << "new stack frame(size):\t" << stackSize << std::endl;

    device_hittable_list* d_world = h_world->create_device_copy();

    camera cam;
    cam.aspect_ratio = aspect_ratio;
    cam.focal_length = 1.0;
    cam.image_width = image_width;
    cam.max_depth = 10;
    cam.samples_per_pixel = 20;
    cam.vfov = 20;
    cam.lookfrom  = point3(13, 2, 3);
    cam.lookat = point3(0, 0, 0);
    cam.vup = vec3(0, 1, 0);
    cam.defocus_angle = 0.6;
    cam.focus_dist = 10.0;

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

    // h_world->free_device_world(d_world);

    return 0;
}