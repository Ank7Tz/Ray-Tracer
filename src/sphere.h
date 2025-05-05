#ifndef SPHERE_H
#define SPHERE_H

#include "interval.h"

class sphere : public hittable<sphere> {
    public:
        __host__ __device__ sphere(const point3& center, double radius)
            : center(center), radius(radius) {}
        
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

                return true;
            }
    private:
        point3 center;
        double radius;
};

#endif