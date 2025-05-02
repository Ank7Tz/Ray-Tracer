#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

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
    int image_width = 256;
    int image_height = 256;
    int total_pixels = image_width * image_height;
    int buffer_size = total_pixels * 3;

    int total_blocks = (total_pixels + BLOCK_SIZE - 1) / BLOCK_SIZE; // (256 * 256 + 128 - 1) / 256 = 256 blocks

    unsigned int *frame_buffer = new unsigned int[buffer_size]; // The frame buffer.

    unsigned int *device_frame_buffer;
    
    cudaMalloc(&device_frame_buffer, sizeof(int) * buffer_size);

    // After cudaMalloc
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cerr << "Error after cudaMalloc: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    generate_frame<<<total_blocks, BLOCK_SIZE>>>(device_frame_buffer, image_width, image_height);

    // After kernel launch
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cerr << "Error after kernel launch: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(frame_buffer, device_frame_buffer, sizeof(int) * buffer_size, cudaMemcpyDeviceToHost);

    // After cudaMemcpy
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        std::cerr << "Error after cudaMemcpy: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            int idx = (j * image_width + i) * 3;
            std::cout << frame_buffer[idx] << ' ' << frame_buffer[idx + 1] << ' ' << frame_buffer[idx + 2] << '\n';
        }
    }

    delete[] frame_buffer;
    cudaFree(device_frame_buffer);

    return 0;
}