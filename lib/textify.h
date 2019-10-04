#pragma once

#include <memory>
#include <new>
#include <limits>
#include <string>

#include <dlib/array.h>
#include <dlib/array2d.h>
#include <dlib/image_io.h>

#include "cuda_helpers.h"

namespace textify {
    namespace details {
        struct gpu_memory_deleter {
            void operator()(void* ptr) const {
                cuda_helpers::mem_device_free(ptr);
            }
        };

        template <class T>
        struct gpu_allocator {
            typedef T value_type;

            gpu_allocator() = default;

            template <class U> constexpr gpu_allocator(const gpu_allocator<U>&) noexcept {}

            T* allocate(std::size_t n) {
                if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
                    throw std::bad_alloc();
                }

                T* p;
                cuda_helpers::malloc(&p, n * sizeof(T));
                return p;
            }
            void deallocate(T* p, std::size_t) noexcept {
                cuda_helpers::mem_device_free(p);
            }
        };
        template <class T, class U>
        bool operator==(const gpu_allocator<T>&, const gpu_allocator<U>&) { return true; }
        template <class T, class U>
        bool operator!=(const gpu_allocator<T>&, const gpu_allocator<U>&) { return false; }
    }

    struct gpu_image {
        dlib::rgb_pixel* pixels;
        size_t width;
        size_t height;
    };

    gpu_image create_gpu_image(size_t width, size_t height);

    gpu_image load_gpu_png(const std::string& path);

    void save_gpu_png(const std::string& path, const gpu_image& device_img);

    gpu_image load_gpu_jpeg(const std::string& path);

    void save_gpu_jpeg(const std::string& path, const gpu_image& device_img);

    struct gpu_gaussian_filter {
        double* ptr;
        size_t size;
    };

    gpu_gaussian_filter create_gpu_gaussian_filter(double sigma);

    void free_gpu_image(gpu_image& device_img);

    void free_gpu_gaussian_filer(gpu_gaussian_filter& filter);

    void gaussian_blur(const gpu_image& src, gpu_image& dst, gpu_gaussian_filter filter);

    void divide(const gpu_image& layer1, gpu_image& layer2);

    void textify(const gpu_image& src, gpu_image& dst);
}
