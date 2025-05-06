#ifndef INTERVAL_H
#define INTERVAL_H

#include <cuda_runtime.h>
#include "utils.h"

class interval {
    public:
        float min, max;
        __host__ __device__ interval() {}

        __host__ __device__ interval(float min, float max) : min(min), max(max) {}

        __device__ float size() const {
            return max - min;
        }

        __device__ bool contains(float x) const {
            return x >= min && x <= max;
        }

        __device__ bool surrounds(float x) const {
            return x > min && x < max;
        }

        __device__ float clamp(float x) const {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }
};

#endif