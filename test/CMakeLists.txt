cmake_minimum_required(VERSION 3.26)
include(CMakePrintHelpers)

project(tfctc)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -mavx2")

find_library(BLIS_LIB "libblis.a" REQUIRED) # find library; store in BLIS_LIB
find_file(BLIS_H blis REQUIRED) # find path to blis.h file; store in BLIS_H

include_directories(
  "${BLIS_H}"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/internal"
  "${CMAKE_CURRENT_SOURCE_DIR}/../src/external/marray"
)

add_executable(complex.test.o test_complex.cpp)
add_executable(real.test.o test_real.cpp)

target_link_libraries(complex.test.o PUBLIC "${BLIS_LIB}")
target_link_libraries(real.test.o PUBLIC "${BLIS_LIB}")