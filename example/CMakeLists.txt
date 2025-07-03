file(GLOB bench_sources "*.cpp") # Find all .cpp test files.

set(CMAKE_BUILD_TYPE Debug)
add_executable(example_t1m "example.cpp")
target_link_libraries(example_t1m PRIVATE t1m)
target_link_libraries(example_t1m PRIVATE ${BLIS_LIBRARIES})