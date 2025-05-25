cmake_minimum_required(VERSION 3.26)

project(
  t1m
  VERSION 0.2
  DESCRIPTION "Transposition-free Complex Tensor Contractions"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CMakePrintHelpers)
include(FetchContent)
include(CTest)

# find_package(OpenMP)

find_library(BLIS_LIBRARIES "libblis.a" REQUIRED)
find_file(BLIS_INCLUDE_DIRS blis REQUIRED)

# Build header-only library.
add_library(t1m INTERFACE)
target_include_directories(t1m INTERFACE ${BLIS_INCLUDE_DIRS})
target_include_directories(t1m INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
  $<INSTALL_INTERFACE:include/>  # <prefix>/include
  )

add_subdirectory(test) # build tests

#add_subdirectory(benchmark) # build benchmarks