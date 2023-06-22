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

set(SOURCES_COMPLEX test_complex.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../src/tfctc.cpp")
set(SOURCES_REAL test_real.cpp "${CMAKE_CURRENT_SOURCE_DIR}/../src/tfctc.cpp")

add_executable(complex.test.o "${SOURCES_COMPLEX}")
add_executable(real.test.o "${SOURCES_REAL}")

target_link_libraries(complex.test.o PUBLIC "${BLIS_LIB}")
target_link_libraries(real.test.o PUBLIC "${BLIS_LIB}")