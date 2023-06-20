cmake_minimum_required(VERSION 3.26)

project(tfctc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mavx2")

find_library(BLIS_LIB "libblis.a" REQUIRED) # find library; store in BLIS_LIB
find_file(BLIS_H blis REQUIRED) # find path to blis.h file; store in BLIS_H
find_path(MARRAY_H "marray.hpp" REQUIRED)

include_directories(
  "${BLIS_H}"
  "${MARRAY_H}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/internal"
)

add_executable(complex.benchmark.o benchmark_complex.cpp)
add_executable(real.benchmark.o benchmark_real.cpp)

target_link_libraries(complex.benchmark.o PUBLIC "${BLIS_LIB}")
target_link_libraries(real.benchmark.o PUBLIC "${BLIS_LIB}")