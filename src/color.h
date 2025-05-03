#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"

using color = vec3;

__host__ __device__ void write_color(unsigned int *frame_buffer, int buffer_idx, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    frame_buffer[buffer_idx] = int(255.999 * r);
    frame_buffer[buffer_idx + 1] = int(255.999 * g);
    frame_buffer[buffer_idx + 2] = int(255.999 * b);
}

#endif