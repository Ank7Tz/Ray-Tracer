#ifndef UTILS_H
#define UTILS_H

#include <curand_kernel.h>
#include <random>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char* const func, const char* file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, 
            line, static_cast<unsigned int>(err), cudaGetErrorString(err), 
            func);
        exit(EXIT_FAILURE);
    }
}

__device__ const float infinity = HUGE_VAL;
__device__ const float pi = 3.1415926535897932385;

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__device__ inline float random_float(curandState* state) {
    return curand_uniform_double(state);
}

__global__ void init_random_states(curandState* states, unsigned long seed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        curand_init(seed, idx, 0, &states[idx]);
    }

}

__device__ inline float random_float(curandState* state, float min, float max) {
    return min + (max - min) * random_float(state);
}

__host__ inline float host_random_float() {
    static std::uniform_real_distribution<float> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

__host__ inline float host_random_float(float min, float max) {
    return min + (max - min) * host_random_float();
}

__device__ inline float linear_to_gamma(float linear_component) {
    return linear_component > 0 ? sqrt(linear_component) : 0;
}

#endif