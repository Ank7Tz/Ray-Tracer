#ifndef UTILS_H
#define UTILS_H

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

__device__ const double infinity = HUGE_VAL;
__device__ const double pi = 3.1415926535897932385;

__host__ __device__ inline double degrees_to_radians(double degrees) {
    return degrees * pi / 180.0;
}

__device__ inline double random_double(curandState* state) {
    return curand_uniform_double(state);
}

__global__ void init_random_states(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__device__ inline double random_double(curandState* state, double min, double max) {
    return min + (max - min) * random_double(state);
}

#endif