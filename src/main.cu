#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

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

__global__ void generate_frame(unsigned int *buffer, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx < total_pixels) {
        int i = idx % width;
        int j = idx / width;

        float r = float(i) / (width - 1);
        float g = float(j) / (height - 1);
        float b = 0.0f;

        int buffer_idx = idx * 3;
        buffer[buffer_idx] = int(255.999 * r);
        buffer[buffer_idx + 1] = int(255.999 * g);
        buffer[buffer_idx + 2] = int(255.999 * b);
    }
}

int main() {
    int image_width = 4096;
    int image_height = 2160;
    int total_pixels = image_width * image_height;
    int buffer_size = total_pixels * 3;
    cudaEvent_t start, stop;
    cudaStream_t stream1;

    int total_blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE; // (256 * 256 + 128 - 1) / 256 = 256 blocks

    unsigned int *frame_buffer; // The Host's frame buffer.

    unsigned int *device_frame_buffer; // Device's frame buffer.

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream1));
    
    CHECK_CUDA_ERROR(cudaMallocHost(&frame_buffer, sizeof(int) * buffer_size));
    CHECK_CUDA_ERROR(cudaMallocAsync(&device_frame_buffer, sizeof(int) * buffer_size, stream1));

    generate_frame<<<total_blocks, BLOCK_SIZE, 0, stream1>>>(device_frame_buffer, image_width, image_height);

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));

    CHECK_CUDA_ERROR(cudaMemcpyAsync(frame_buffer, device_frame_buffer, sizeof(int) * buffer_size, cudaMemcpyDeviceToHost, stream1));

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            int idx = (j * image_width + i) * 3;
            std::cout << frame_buffer[idx] << ' ' << frame_buffer[idx + 1] << ' ' << frame_buffer[idx + 2] << '\n';
        }
    }

    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream1));

    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;

    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA_ERROR(cudaFreeHost(frame_buffer));
    CHECK_CUDA_ERROR(cudaFree(device_frame_buffer));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    std::cerr << "Execution time: " << milliseconds << " milliseconds" << std::endl;

    return 0;
}