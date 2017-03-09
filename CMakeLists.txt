cmake_minimum_required(VERSION 3.1)
project (SmithWatermanOpenCL)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED 14)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
    add_definitions(${OpenMP_CXX_FLAGS})
endif(OPENMP_FOUND)

add_executable(main main.cpp SW_kernels.cl)
target_link_libraries(main ${OpenCL_LIBRARY})