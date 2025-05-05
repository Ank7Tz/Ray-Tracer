#ifndef INTERVAL_H
#define INTERVAL_H

#include <cuda_runtime.h>
#include "utils.h"

class interval {
    public:
        double min, max;
        __host__ __device__ interval() {}

        __host__ __device__ interval(double min, double max) : min(min), max(max) {}

        __device__ double size() const {
            return max - min;
        }

        __device__ bool contains(double x) const {
            return x >= min && x <= max;
        }

        __device__ bool surrounds(double x) const {
            return x > min && x < max;
        }

        __device__ double clamp(double x) const {
            if (x < min) return min;
            if (x > max) return max;
            return x;
        }
};

#endif