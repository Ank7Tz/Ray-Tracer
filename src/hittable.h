#ifndef HITTABLE_H
#define HITTABLE_H

#include "ray.h"

class hit_record {
    public:
        point3 p;
        vec3 normal;
        float t;
        bool front_face;

        __host__ __device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
            front_face = dot(r.direction(), outward_normal) < 0;
            normal = front_face ? outward_normal : -outward_normal;
        }
};

template <typename Derived>
class hittable {
    public:
        __host__ __device__ bool hit(
            const ray& r, float ray_tmin, float ray_tmax, hit_record& rec) 
            const {
                return static_cast<const Derived*>(this)->hit_impl(r, ray_tmin, ray_tmax, rec);
            }
};

#endif