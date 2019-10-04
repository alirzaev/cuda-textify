#include "textify.h"
#include "cuda_helpers.h"

static double gaussian(double sigma, double x) {
    return 1.0 / (sigma * sqrt(2.0 * dlib::pi)) * exp((x * x) / (-2.0 * sigma * sigma));
}

namespace textify {
    gpu_image create_gpu_image(size_t width, size_t height) {
        gpu_image device_img{};
        device_img.width = width;
        device_img.height = height;
        cuda_helpers::malloc((void **) &(device_img.pixels), width * height * sizeof(dlib::rgb_pixel));

        return device_img;
    }

    gpu_image load_gpu_png(const std::string& path) {
        dlib::array2d<dlib::rgb_pixel> img;
        dlib::load_png(img, path);

        gpu_image device_img{};
        cuda_helpers::malloc((void **) &(device_img.pixels), img.size() * sizeof(dlib::rgb_pixel));
        cuda_helpers::copy_host2device(device_img.pixels, img.begin(), img.size() * sizeof(dlib::rgb_pixel));
        device_img.width = img.nc();
        device_img.height = img.nr();

        return device_img;
    }

    void save_gpu_png(const std::string& path, const gpu_image &device_img) {
        dlib::array2d<dlib::rgb_pixel> img;
        img.set_size(device_img.height, device_img.width);

        cuda_helpers::copy_device2host(img.begin(), device_img.pixels, img.size() * sizeof(dlib::rgb_pixel));
        dlib::save_png(img, path);
    }

    gpu_image load_gpu_jpeg(const std::string& path) {
        dlib::array2d<dlib::rgb_pixel> img;
        dlib::load_jpeg(img, path);

        gpu_image device_img{};
        cuda_helpers::malloc((void **) &(device_img.pixels), img.size() * sizeof(dlib::rgb_pixel));
        cuda_helpers::copy_host2device(device_img.pixels, img.begin(), img.size() * sizeof(dlib::rgb_pixel));
        device_img.width = img.nc();
        device_img.height = img.nr();

        return device_img;
    }

    void save_gpu_jpeg(const std::string& path, const gpu_image &device_img) {
        dlib::array2d<dlib::rgb_pixel> img;
        img.set_size(device_img.height, device_img.width);

        cuda_helpers::copy_device2host(img.begin(), device_img.pixels, img.size() * sizeof(dlib::rgb_pixel));
        dlib::save_jpeg(img, path);
    }

    gpu_gaussian_filter create_gpu_gaussian_filter(double sigma) {
        const size_t filter_sz = 55;
        double *host_ptr = (double *) malloc(filter_sz * sizeof(double));
        double sum = 0;
        for (size_t i = 0; i < filter_sz; ++i) {
            host_ptr[i] = gaussian(sigma, i - filter_sz / 2);
            sum += host_ptr[i];
        }
        for (size_t i = 0; i < filter_sz; ++i) {
            host_ptr[i] /= sum;
        }

        double *device_ptr;
        cuda_helpers::malloc((void **) &device_ptr, filter_sz * sizeof(double));
        cuda_helpers::copy_host2device(device_ptr, host_ptr, filter_sz * sizeof(double));

        return gpu_gaussian_filter{
                device_ptr,
                filter_sz
        };
    }

    void free_gpu_image(gpu_image& device_img) {
        cuda_helpers::mem_device_free(device_img.pixels);
        device_img.pixels = nullptr;
    }

    void free_gpu_gaussian_filer(gpu_gaussian_filter &filter) {
        cuda_helpers::mem_device_free(filter.ptr);
    }
}