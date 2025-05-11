#ifndef VEC3_H
#define VEC3_H

#include <curand_kernel.h>
#include <iostream>
#include "utils.h"

class vec3 {
    public:
        float e[3];

        // unsafe, never to use this!
        __host__ __device__ vec3() {}
        
        // __host__ __device__ vec3() : e{0, 0, 0} {}

        __host__ __device__ vec3(float e0, float e1, float e2) 
            : e{e0, e1, e2} {}

        __host__ __device__ float x() const {
            return e[0];
        }

        __host__ __device__ float y() const {
            return e[1];
        }

        __host__ __device__ float z() const {
            return e[2];
        }

        __host__ __device__ vec3 operator-() const {
            return vec3(-e[0], -e[1], -e[2]);
        }

        __host__ __device__ float operator[](int i) const {
            return e[i];
        }

        __host__ __device__ float& operator[](int i) {
            return e[i];
        }

        __host__ __device__ vec3& operator+=(const vec3& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];

            return *this;
        }

        __host__ __device__ vec3& operator*=(float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;

            return *this;
        }

        __host__ __device__ vec3& operator/=(float t) {
            return *this *= (1/t);
        }

        __host__ __device__ float length() const {
            return sqrt(this->length_squared());
        }

        __host__ __device__ float length_squared() const {
            return (e[0] * e[0]) + (e[1] * e[1]) + (e[2] * e[2]);
        }

        __host__ __device__ bool near_zero() const {
            auto s = 1e-8;
            return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }

        static __device__ vec3 cuda_random(curandState* state) {
            return vec3(random_float(state), random_float(state), random_float(state));
        }

        static __device__ vec3 cuda_random(curandState* state, float min, float max) {
            return vec3(random_float(state, min, max), random_float(state, min, max), 
                        random_float(state, min, max));
        }

        static __host__ vec3 random() {
            return vec3(host_random_float(), host_random_float(), host_random_float());
        }

        static __host__ vec3 random(float min, float max) {
            return vec3(host_random_float(min, max), host_random_float(min, max), 
                        host_random_float(min, max));
        }
};

using point3 = vec3;

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, float t) {
    return vec3(u.e[0] * t, u.e[1] * t, u.e[2] * t);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return v * t;
}

__host__ __device__ inline vec3 operator/(const vec3& u, float t) {
    return u * (1 / t);
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return (u.e[0] * v.e[0]) + (u.e[1] * v.e[1]) + (u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& u) {
    return u / u.length();
}

__device__ inline vec3 random_unit_vector(curandState* state) {
    while (true) {
        auto p = vec3::cuda_random(state, -1, 1);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1)
            return p / sqrt(lensq);
    }
}

__device__ inline vec3 random_on_hemisphere(curandState* state, const vec3& normal) {
    vec3 on_unit_sphere = random_unit_vector(state);
    auto value = dot(on_unit_sphere, normal);

    return value > 0.0 ? on_unit_sphere : -on_unit_sphere;
}

__host__ __device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__host__ __device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = fminf(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrtf(fabsf(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ inline vec3 random_in_unit_disk(curandState* state) {
    while (true) {
        auto p = vec3(random_float(state, -1, 1), random_float(state, -1, 1), 0);
        if (p.length_squared() < 1) {
            return p;
        }
    }
}

#endif