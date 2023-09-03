cmake_minimum_required(VERSION 3.0)

project(
  T1M
  VERSION 0.1
  DESCRIPTION "Transposition-free Complex Tensor Contractions"
  LANGUAGES CXX)

include(ExternalProject)
include(CTest)

find_package(OpenMP)
find_library(BLIS_LIB "libblis.a" REQUIRED) # find library; store in BLIS_LIB
find_file(BLIS_H blis REQUIRED) # find path to blis.h file; store in BLIS_H
find_file(MARRAY_H marray REQUIRED)

add_subdirectory(src) # build library
add_subdirectory(test) # build tests
add_subdirectory(benchmark) # build benchmarks