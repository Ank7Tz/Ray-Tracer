#ifndef COLOR_H
#define COLOR_H

#include "vec3.h"
#include "interval.h"

using color = vec3;

__constant__ interval g_intensity;

__device__ void write_color(int *frame_buffer, int buffer_idx, const color& pixel_color) {
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    r = linear_to_gamma(r);
    g = linear_to_gamma(g);
    b = linear_to_gamma(b);

    frame_buffer[buffer_idx] = int(256 * g_intensity.clamp(r));
    frame_buffer[buffer_idx + 1] = int(256 * g_intensity.clamp(g));
    frame_buffer[buffer_idx + 2] = int(256 * g_intensity.clamp(b));
}

#endif