#include <cstdlib>
#include <cmath>

#include <dlib/image_transforms.h>

#include "textify.h"
#include "cuda_helpers.h"

static __global__ void gpu_gaussian_blur_h(
        textify::gpu_image src,
        textify::gpu_image dst,
        textify::gpu_gaussian_filter gpu_filter
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t width = src.width, height = src.height;
    if (x >= width || y >= height) {
        return;
    }

    auto *filter = gpu_filter.ptr;
    size_t filter_sz = gpu_filter.size;

    dlib::rgb_pixel p = src.pixels[y * width + x];
    unsigned int r = 0, g = 0, b = 0;

    for (long k = filter_sz / -2; k <= filter_sz / 2; ++k) {
        if (k + x >= width || k + x < 0) {
            continue;
        }
        dlib::rgb_pixel t = src.pixels[y * width + x + k];
        r += t.red * filter[k + filter_sz / 2];
        g += t.green * filter[k + filter_sz / 2];
        b += t.blue * filter[k + filter_sz / 2];
    }
    p.red = r / 1024;
    p.green = g / 1024;
    p.blue = b / 1024;
    dst.pixels[y * width + x] = p;
}

static __global__ void gpu_gaussian_blur_v(
        textify::gpu_image src,
        textify::gpu_image dst,
        textify::gpu_gaussian_filter gpu_filter
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t width = src.width, height = src.height;
    if (x >= width || y >= height) {
        return;
    }

    auto *filter = gpu_filter.ptr;
    size_t filter_sz = gpu_filter.size;

    dlib::rgb_pixel p = src.pixels[y * width + x];
    unsigned int r = 0, g = 0, b = 0;

    for (long k = filter_sz / -2; k <= filter_sz / 2; ++k) {
        if (k + y >= height || k + y < 0) {
            continue;
        }
        dlib::rgb_pixel t = src.pixels[(y + k) * width + x];
        r += t.red * filter[k + filter_sz / 2];
        g += t.green * filter[k + filter_sz / 2];
        b += t.blue * filter[k + filter_sz / 2];
    }
    p.red = r / 1024;
    p.green = g / 1024;
    p.blue = b / 1024;
    dst.pixels[y * width + x] = p;
}

static __global__ void gpu_divide(
        textify::gpu_image layer1,
        textify::gpu_image layer2
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t width = layer1.width, height = layer1.height;
    if (x >= width || y >= height) {
        return;
    }

    dlib::rgb_pixel p1 = layer1.pixels[y * width + x];
    dlib::rgb_pixel p2 = layer2.pixels[y * width + x];

    unsigned int r, g, b;
    r = p1.red * 256 / (p2.red + 1);
    g = p1.green * 256 / (p2.green + 1);
    b = p1.blue * 256 / (p2.blue + 1);

    p2.red = r > 255 ? 255 : r;
    p2.green = g > 255 ? 255 : g;
    p2.blue = b > 255 ? 255 : b;
    layer2.pixels[y * width + x] = p2;
}

namespace textify {
    void gaussian_blur(const gpu_image& src, gpu_image& dst, gpu_gaussian_filter filter) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        gpu_image tmp{};
        cudaMalloc((void **) &(tmp.pixels), src.width * src.height * sizeof(dlib::rgb_pixel));
        tmp.width = src.width;
        tmp.height = src.height;

        dim3 thr_per_block(16, 16);
        dim3 blocks_count((src.width + 16) / thr_per_block.x, (src.height + 16) / thr_per_block.y);

        gpu_gaussian_blur_h <<< blocks_count, thr_per_block, 0, stream >>> (src, tmp, filter);
        gpu_gaussian_blur_v <<< blocks_count, thr_per_block, 0, stream >>> (tmp, dst, filter);

        cudaStreamSynchronize(stream);
        cudaFree(tmp.pixels);
        cudaStreamDestroy(stream);
    }

    void divide(const gpu_image& layer1, gpu_image& layer2) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        dim3 thr_per_block(16, 16);
        dim3 blocks_count((layer1.width + 16) / thr_per_block.x, (layer1.height + 16) / thr_per_block.y);

        gpu_divide <<< blocks_count, thr_per_block, 0, stream >>> (layer1, layer2);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    void textify(const gpu_image& src, gpu_image& dst) {
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        gpu_image tmp{};
        cudaMalloc((void **) &(tmp.pixels), src.width * src.height * sizeof(dlib::rgb_pixel));
        tmp.width = src.width;
        tmp.height = src.height;
        textify::gpu_gaussian_filter filter = textify::create_gpu_gaussian_filter(40);

        dim3 thr_per_block(16, 16);
        dim3 blocks_count((src.width + 16) / thr_per_block.x, (src.height + 16) / thr_per_block.y);

        gpu_gaussian_blur_h <<< blocks_count, thr_per_block, 0, stream >>> (src, tmp, filter);
        gpu_gaussian_blur_v <<< blocks_count, thr_per_block, 0, stream >>> (tmp, dst, filter);
        gpu_divide <<< blocks_count, thr_per_block, 0, stream >>> (src, dst);

        cudaStreamSynchronize(stream);
        cudaFree(tmp.pixels);
        cudaFree(filter.ptr);
        cudaStreamDestroy(stream);
    }
}