cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
# nvcc -I /usr/local/cuda/include -L /usr/local/cuda/lib main.cu .... -std=c++11 -D_MWAITXINTRIN_H_INCLUDEDcudaMalloc

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)   

set (CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_definitions("-D_MWAITXINTRIN_H_INCLUDED")

include_directories(include)

file(GLOB
    SRC_FILES
    src/layers/*.cu
    src/nn_utils/*.cu
    src/*.cu
    src/main.cpp
    src/*.cpp)

cuda_add_executable(main ${SRC_FILES})