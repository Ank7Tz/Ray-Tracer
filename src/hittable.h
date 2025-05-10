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
        
        __host__ void *device_copy() const;
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
        __host__ __device__ metal() : material(Metal) {}

        __host__ __device__ metal(const color& albedo, float fuzz) : albedo(albedo), material(Metal)
                                , fuzz(fuzz < 1 ? fuzz : 1) {}

        __device__ bool scatter_impl(const ray& r_in, const hit_record& rec, color& attenuation, 
                                     ray& scattered, curandState* state)
        const {
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = unit_vector(reflected) + (fuzz * random_unit_vector(state));
            scattered = ray(rec.p, reflected);
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }

        __host__ material* device_copy_impl() const {
            metal* d_metal_copy;
            cudaMalloc(&d_metal_copy, sizeof(metal));
            
            metal temp_metal(albedo, fuzz);

            cudaMemcpy(d_metal_copy, &temp_metal, sizeof(metal), cudaMemcpyHostToDevice);
            
            return d_metal_copy;
        }

    private:
        color albedo;
        float fuzz;
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
        
        __host__ void *device_copy() const;
};

class sphere : public hittable {
    public:
        __host__ __device__ sphere() : hittable(SPHERE) {}

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

            __device__ __host__ void set_center(point3& center) {
                center = center;
            }

            __device__ __host__ void set_radius(float radius) {
                radius = radius;
            }

            __device__ __host__ void set_material(material& material) {
                material = material;
            }

            __host__ sphere* device_copy_impl() const {
                sphere* d_sphere_copy;
                cudaMalloc(&d_sphere_copy, sizeof(sphere));
                
                material* d_mat_copy = (material*)mat->device_copy();
                
                sphere temp_sphere(center, radius, d_mat_copy);
                
                cudaMemcpy(d_sphere_copy, &temp_sphere, sizeof(sphere), cudaMemcpyHostToDevice);
                
                return d_sphere_copy;
            }
    private:
        point3 center;
        float radius;
        material* mat;
};

class lambertian : public material {
    public:
        __device__ __host__ lambertian() : material(Lambertian) {}

        __device__ __host__ lambertian(const color& albedo) : albedo(albedo), material(Lambertian) {}

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

        __host__ material* device_copy_impl() const {
            lambertian* d_lambertian_copy;
            cudaMalloc(&d_lambertian_copy, sizeof(lambertian));
            
            lambertian temp_lambertian(albedo);
            
            cudaMemcpy(d_lambertian_copy, &temp_lambertian, sizeof(lambertian), cudaMemcpyHostToDevice);
            
            return d_lambertian_copy;
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
    } else if (type == Metal) {
        const metal* met = reinterpret_cast<const metal*>(this);
        return met->scatter_impl(r_in, rec, attenuation, scattered, state);
    }

    return false;
}

__host__ void *hittable::device_copy() const {
    if (type == SPHERE) {
        const sphere* obj = reinterpret_cast<const sphere*>(this);
        return obj->device_copy_impl();
    }

    return nullptr;
}

__host__ void *material::device_copy() const {
    if (type == Lambertian) {
        const lambertian* lam = reinterpret_cast<const lambertian*>(this); 
        return lam->device_copy_impl();
    } else if (type == Metal) {
        const metal* met = reinterpret_cast<const metal*>(this);
        return met->device_copy_impl();
    }

    return nullptr;
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
            sphere** device_spheres = new sphere*[count];
            for (int i = 0; i < count; i++) {
                device_spheres[i] = (sphere*) objects[i]->device_copy();
            }
            
            sphere** d_objects;
            CHECK_CUDA_ERROR(cudaMalloc(&d_objects, count * sizeof(sphere*)));
            
            CHECK_CUDA_ERROR(cudaMemcpy(d_objects, device_spheres, count * sizeof(sphere*), cudaMemcpyHostToDevice));
            delete[] device_spheres;  // Free temporary array
            
            device_hittable_list h_world(hittable::WORLD);
            h_world.objects = d_objects;
            h_world.count = count;
            
            device_hittable_list* d_world;
            CHECK_CUDA_ERROR(cudaMalloc(&d_world, sizeof(device_hittable_list)));
            CHECK_CUDA_ERROR(cudaMemcpy(d_world, &h_world, sizeof(device_hittable_list), cudaMemcpyHostToDevice));
            
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