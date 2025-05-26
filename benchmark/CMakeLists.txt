file(GLOB bench_sources "*.cpp") # Find all .cpp test files.

set(CMAKE_BUILD_TYPE Release)
add_executable(bench_t1m ${bench_sources})
target_link_libraries(bench_t1m PRIVATE t1m)
target_link_libraries(bench_t1m PRIVATE ${BLIS_LIBRARIES})