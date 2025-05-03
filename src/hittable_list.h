#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H

#include "hittable.h"
#include "utils.h"
#include "sphere.h"

class device_hittable_list : public hittable {
    public:
        hittable** objects; // array of objects in the list.
        int objects_count;
        int objects_capacity;

        __device__ device_hittable_list() 
                            : objects(nullptr), objects_count(0), objects_capacity(0) {}

        __device__ void allocate(int capacity) {
            // Use regular device malloc instead of managed memory
            objects = new hittable*[capacity];
            objects_capacity = capacity;
            objects_count = 0;
        }

        __device__ void add(hittable* object) {
            if (objects_count >= objects_capacity) {
                int new_objects_capacity = objects_capacity > 0 ? objects_capacity * 2 : 8;
                hittable** new_objects = new hittable*[new_objects_capacity];

                // move objects
                for (int i = 0; i < objects_count; i++) {
                    new_objects[i] = objects[i];
                }

                if (objects) {
                    delete[] objects;
                }

                objects = new_objects;
                objects_capacity = new_objects_capacity;
            }

            objects[objects_count++] = object;
        }

        __device__ bool hit(const ray& r, float ray_tmin, float ray_tmax, hit_record& rec)
        const override {
            hit_record temp_rec;
            bool hit_anything = false;
            auto closest_so_far = ray_tmax;

            if (objects == nullptr || objects_count <= 0) return false;

            for (int i = 0; i < objects_count; i++) {
                if (objects[i]->hit(r, ray_tmin, closest_so_far, temp_rec)) {
                    hit_anything = true;
                    closest_so_far = temp_rec.t;
                    rec = temp_rec;
                }
            }

            return hit_anything;
        }

        __device__ void clear() {
            objects_count = 0;
        }

        __device__ ~device_hittable_list() {
            if (objects) {
                delete[] objects;
            }
        }
};

// Kernel to create a world with objects
__global__ void setup_world(device_hittable_list** world_ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Create the world
        device_hittable_list* world = new device_hittable_list();
        world->allocate(2);
        
        // Create sphere objects directly on device
        sphere* sphere1 = new sphere(point3(0, 0, -1), 0.5);
        world->add(sphere1);
        
        sphere* sphere2 = new sphere(point3(0, -100.5, -1), 100);
        world->add(sphere2);
        
        // Store the world pointer to the output parameter
        *world_ptr = world;
    }
}

// Kernel to free device memory
__global__ void cleanup_world(device_hittable_list* world) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Free each object in the world
        for (int i = 0; i < world->objects_count; i++) {
            delete world->objects[i];
        }
        
        // Delete the world itself
        delete world;
    }
}

#endif