#include <iostream>
#include <vector>
#include <future>
#include <string>

#include <boost/filesystem.hpp>

#include "lib/textify.h"

using namespace boost::filesystem;

int main(int argc, char* argv[]) {
    if (argc < 4) {
        return -1;
    }
    path input_dir(argv[1]), output_dir(argv[2]);
    if (is_regular_file(input_dir)) {
        std::cerr << input_dir.string() << " is not a directory" << std::endl;
        return -1;
    }
    if (is_regular_file(output_dir)) {
        std::cerr << output_dir.string() << " is not a directory" << std::endl;
        return -1;
    }
    size_t batch_size;
    try {
        batch_size = std::stoull(argv[3]);
    } catch (const std::exception&) {
        std::cerr << "Invalid batch size: " << argv[3] << std::endl;
        return -1;
    }
    std::vector<path> files;
    for (const auto& entry : directory_iterator(input_dir)) {
        if (is_regular_file(entry.path())) {
            auto ext = entry.path().extension();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
                files.push_back(entry.path());
            }
        }
    }

    size_t files_cnt = files.size();
    std::vector<textify::gpu_image> srcs(files_cnt), dsts(files_cnt);
    std::vector<std::future<textify::gpu_image>> futures_srcs(files_cnt);
    std::vector<std::future<void>> futures_dsts(files_cnt);

    for (size_t j = 0; j < (files_cnt + batch_size) / batch_size; ++j) {
        for (size_t i = j * batch_size; i < (j + 1) * batch_size && i < files_cnt; ++i) {
            auto ext = files[i].extension();
            if (ext == ".jpg" || ext == ".jpeg") {
                futures_srcs[i] = std::async(std::launch::async, textify::load_gpu_jpeg, files[i].string());
            } else {
                futures_srcs[i] = std::async(std::launch::async, textify::load_gpu_png, files[i].string());
            }
        }
        for (size_t i = j * batch_size; i < (j + 1) * batch_size && i < files_cnt; ++i) {
            srcs[i] = futures_srcs[i].get();
            dsts[i] = textify::create_gpu_image(srcs[i].width, srcs[i].height);
        }
        std::cout << "Loaded" << std::endl;
        for (size_t i = j * batch_size; i < (j + 1) * batch_size && i < files_cnt; ++i) {
            textify::textify(srcs[i], dsts[i]);
        }
        std::cout << "GPU" << std::endl;
        for (size_t i = j * batch_size; i < (j + 1) * batch_size && i < files_cnt; ++i) {
            auto output_file = output_dir;
            output_file /= files[i].filename();

            auto ext = files[i].extension();
            if (ext == ".jpg" || ext == ".jpeg") {
                futures_dsts[i] = std::async(std::launch::async, textify::save_gpu_jpeg, output_file.string(), dsts[i]);
            } else {
                futures_dsts[i] = std::async(std::launch::async, textify::save_gpu_png, output_file.string(), dsts[i]);
            }
        }
        for (size_t i = j * batch_size; i < (j + 1) * batch_size && i < files_cnt; ++i) {
            futures_dsts[i].get();
            textify::free_gpu_image(dsts[i]);
            textify::free_gpu_image(srcs[i]);
        }
    }

    std::cout << "Finished" << std::endl;

    return 0;
}
