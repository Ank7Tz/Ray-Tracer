#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"
#include "interval.h"
#include "utils.h"

class material;
class sphere;
class device_hittable_list;
class hit_record;
class lambertian;

class material {
    public:
        enum Type {
            Lambertian,
            Metal
        };

        Type type;

        __device__ __host__ material() {}

        __device__ __host__ material(Type t) : type(t) {}

        __device__ bool scatter(
            const ray& r_in, const hit_record& rec, color& attenuation, 
            ray& scattered, curandState* state) const;
};

class hit_record {
    public:
        point3 p;
        vec3 normal;
        float t;
        bool front_face;
        material* mat;

        __host__ __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};

class metal : public material {
    public:
        __host__ __device__ metal(const color& albedo) : albedo(albedo) {}

        __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered)
        const {
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            scattered = ray(rec.p, reflected);
            attenuation = albedo;
            return true;
        } 

    private:
        color albedo;
};


class hittable {
    public:
        enum Type {
            SPHERE,
            WORLD
        };

        Type type;


        __host__ __device__ hittable(Type t) : type(t) {}

        __device__ bool hit(const ray& r, 
                                    interval ray_t, hit_record& rec) const;
};

class sphere : public hittable {
    public:
        __host__ __device__ sphere(const point3& center, float radius, material* mat)
            : center(center), radius(radius), hittable(SPHERE), mat(mat) {}
        
        __device__ bool hit_impl(const ray& r, interval ray_t, hit_record& rec)
            const {
                vec3 oc = center - r.origin();
                auto a = r.direction().length_squared();
                auto h = dot(r.direction(), oc);
                auto c = oc.length_squared() - radius * radius;

                auto discriminant = h*h - a*c;
                if (discriminant < 0) {
                    return false;
                }

                auto sqrtd = sqrt(discriminant);

                auto root = (h - sqrtd) / a;
                if (!ray_t.contains(root)) {
                    root = (h + sqrtd) / a;
                    if (!ray_t.contains(root)) {
                        return false;
                    }
                }

                rec.t = root;
                rec.p = r.at(rec.t);
                vec3 outward_normal = (rec.p - center) / radius;
                rec.set_face_normal(r, outward_normal);

                rec.mat = mat;

                return true;
            }
    private:
        point3 center;
        float radius;
        material* mat;
};

class lambertian : public material {
    public:
        __device__ __host__ lambertian(const color& albedo) : albedo(albedo) {}

        __device__ bool scatter_impl(const ray& r_in, const hit_record& rec, 
                          color& attenuation, ray& scattered,
                          curandState* state) const {
            auto scatter_direction = rec.normal + random_unit_vector(state);

            if (scatter_direction.near_zero()) {
                scatter_direction = rec.normal;
            }

            scattered = ray(rec.p, scatter_direction);
            attenuation = albedo;
            return true;
        }
    
    private:
        color albedo;
};

class device_hittable_list : public hittable {
    public:
        sphere** objects;
        int count;

        __host__ __device__ device_hittable_list(Type type) : hittable(type) {}

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

__device__ bool hittable::hit(const ray& r, interval ray_t, hit_record& rec) const {
    if (type == SPHERE) {
        const sphere* s = reinterpret_cast<const sphere*> (this);
        return s->hit_impl(r, ray_t, rec);
    } else if (type == WORLD) {
        const device_hittable_list* d_world = reinterpret_cast<const device_hittable_list*> (this);
        return d_world->hit_impl(r, ray_t, rec);
    }
    // default
    return false;
}

__device__ bool material::scatter(
    const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* state 
) const {
    if (type == Lambertian) {
        const lambertian* lam = reinterpret_cast<const lambertian*>(this); 
        return lam->scatter_impl(r_in, rec, attenuation, scattered, state);
    }

    return false;
}

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

            device_hittable_list h_world(hittable::WORLD);
            h_world.objects = d_objects;
            h_world.count = count;

            device_hittable_list* d_world;
            CHECK_CUDA_ERROR(cudaMalloc(&d_world, sizeof(device_hittable_list)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_world, &h_world, sizeof(h_world), cudaMemcpyHostToDevice));

            return d_world;
        }

        void free_device_world(device_hittable_list* d_world) {
            device_hittable_list h_world(hittable::WORLD);
            CHECK_CUDA_ERROR(cudaMemcpy(&h_world, d_world, sizeof(d_world), cudaMemcpyDeviceToHost));

            CHECK_CUDA_ERROR(cudaFree(h_world.objects));
            CHECK_CUDA_ERROR(cudaFree(d_world));
        }
};

#endif