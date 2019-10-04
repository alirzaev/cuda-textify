#include "cuda_helpers.h"

namespace cuda_helpers {
    void malloc(void **ptr, size_t size) {
        cudaError_t status = cudaMalloc(ptr, size);
        if (status != cudaSuccess) {
            throw std::exception();
        }
    }

    void copy_host2device(void *dst, void *src, size_t size) {
        cudaError_t status = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            throw std::exception();
        }
    }

    void copy_device2host(void *dst, void *src, size_t size) {
        cudaError_t status = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
            throw std::exception();
        }
    }

    void reset_device() {
        cudaError_t status = cudaDeviceReset();
        if (status != cudaSuccess) {
            throw std::exception();
        }
    }

    void synchronize() {
        cudaError_t status = cudaDeviceSynchronize();
        if (status != cudaSuccess) {
            throw std::exception();
        }
    }

    void mem_device_free(void *ptr) {
        cudaFree(ptr);
    }

    void set_device(int n) {
        cudaSetDevice(n);
    }
}