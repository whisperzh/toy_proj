cmake_minimum_required(VERSION 3.18)
project(nccl_dynamic LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# NCCL & CUDA
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

find_package(PkgConfig REQUIRED)
pkg_check_modules(JSONCPP jsoncpp)

include_directories(${JSONCPP_INCLUDE_DIRS})
link_directories(${JSONCPP_LIBRARY_DIRS})

add_executable(nccl_dynamic main.cpp)

target_link_libraries(nccl_dynamic
    nccl
    cudart
    ${JSONCPP_LIBRARIES}
)
