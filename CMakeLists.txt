cmake_minimum_required(VERSION 3.20.0)
project(cuda-practice VERSION 0.1.0 LANGUAGES CUDA CXX C)

find_package(CUDAToolkit)
# add_subdirectory(1_cuda_reduce_study)
add_subdirectory(2_cuda_sgemm_study)