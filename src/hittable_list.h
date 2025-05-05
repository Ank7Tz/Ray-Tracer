#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "sphere.h"
#include "utils.h"
#include "interval.h"

class device_hittable_list : public hittable<device_hittable_list> {
    public:
        sphere** objects;
        int count;

        __host__ __device__ device_hittable_list() {}

        __device__ bool hit_impl(const ray& r, interval ray_t, hit_record& rec)
        const {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_t.max;

            for (int i = 0; i < count; i++) {
                if (objects[i]->hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
                    closest_so_far = temp_rec.t;
                    hit_anything = true;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }
};

class host_hittable_list {
    public:
        sphere** objects;
        int count;
        int capacity;

        host_hittable_list() : objects(nullptr), count(0), capacity(0) {}

        ~host_hittable_list() = default;

        void allocate(int new_capacity) {
            if (capacity >= new_capacity) return;

            sphere** new_objects = new sphere*[new_capacity];

            if (count > 0 && objects) {
                for (int i = 0; i < count; i++) {
                    new_objects[i] = objects[i];
                }

                delete[] objects;
            }

            objects = new_objects;
            capacity = new_capacity;
        }

        void add(sphere* object) {
            if (count >= capacity) {
                int new_capacity = capacity > 0 ? capacity * 2 : 8;
                allocate(new_capacity);
            }
            objects[count++] = object;
        }

        device_hittable_list* create_device_copy() {
            sphere** d_objects;
            CHECK_CUDA_ERROR(cudaMalloc(&d_objects, count * sizeof(sphere*)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_objects, objects, count * sizeof(sphere*), cudaMemcpyHostToDevice));

            device_hittable_list h_world;
            h_world.objects = d_objects;
            h_world.count = count;

            device_hittable_list* d_world;
            CHECK_CUDA_ERROR(cudaMalloc(&d_world, sizeof(device_hittable_list)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_world, &h_world, sizeof(h_world), cudaMemcpyHostToDevice));

            return d_world;
        }

        void free_device_world(device_hittable_list* d_world) {
            device_hittable_list h_world;
            CHECK_CUDA_ERROR(cudaMemcpy(&h_world, d_world, sizeof(d_world), cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERROR(cudaFree(h_world.objects));
            CHECK_CUDA_ERROR(cudaFree(d_world));
        }
};

#endif