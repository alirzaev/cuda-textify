#pragma once

#include <vector>
#include <cuda_runtime.h>

namespace cuda_helpers {
    struct rgb_pixel {
        rgb_pixel(
        ) : red(0), green(0), blue(0) {}

        rgb_pixel(
                unsigned char red_,
                unsigned char green_,
                unsigned char blue_
        ) : red(red_), green(green_), blue(blue_) {}

        unsigned char red;
        unsigned char green;
        unsigned char blue;
    };

    struct image_t {
        std::vector<rgb_pixel> pixels;

        size_t width;

        size_t height;
    };

    void malloc(void** ptr, size_t size);

    void copy_host2device(void* dst, void* src, size_t size);

    void copy_device2host(void* dst, void* src, size_t size);

    void reset_device();

    void synchronize();

    void mem_device_free(void* ptr);

    void set_device(int n);
}
