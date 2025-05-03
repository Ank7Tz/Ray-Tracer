#ifndef VEC3_H
#define VEC3_H

#include <curand_kernel.h>
#include <iostream>

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

#endif