cmake_minimum_required(VERSION 3.15)
project(cuda_textify_app LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(dlib CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system)
find_package(JPEG REQUIRED)

add_library(textify STATIC
        lib/textify.cu
        lib/textify.h
        lib/textify.cpp
        lib/cuda_helpers.h
        lib/cuda_helpers.cpp)
set_target_properties(textify PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(textify PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(textify PRIVATE dlib::dlib)
target_link_libraries(textify PRIVATE ${JPEG_LIBRARIES})

add_executable(cuda_textify_app main.cpp dummy.cpp)
set_target_properties(cuda_textify_app PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cuda_textify_app PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(cuda_textify_app PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(cuda_textify_app PRIVATE ${Boost_LIBRARIES})
target_link_libraries(cuda_textify_app PRIVATE dlib::dlib)
target_link_libraries(cuda_textify_app PRIVATE textify)