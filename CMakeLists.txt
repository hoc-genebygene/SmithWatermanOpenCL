cmake_minimum_required(VERSION 3.1)
project (SmithWatermanOpenCL)

set (CMAKE_CXX_STANDARD 14)
set (CMAKE_CXX_STANDARD_REQUIRED 14)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})

add_executable(main main.cpp square_kernel.cl)
target_link_libraries(main ${OpenCL_LIBRARY})