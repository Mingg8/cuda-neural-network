cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
# nvcc -I /usr/local/cuda/include -L /usr/local/cuda/lib main.cu .... -std=c++11 -D_MWAITXINTRIN_H_INCLUDEDcudaMalloc

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
add_definitions("-D_MWAITXINTRIN_H_INCLUDED")
find_package(CUDA REQUIRED)

include_directories(include)

file(GLOB
    SRC_FILES
    src/layers/*.cu
    src/nn_utils/*.cu
    src/*.cu)

cuda_add_executable(main ${SRC_FILES})