include(GoogleTest)

file(GLOB test_sources "*.cpp") # Find all .cpp test files.

enable_testing()

set(CMAKE_BUILD_TYPE Debug)
add_executable(test_t1m ${test_sources})
target_link_libraries(test_t1m PRIVATE t1m)
target_link_libraries(test_t1m PRIVATE ${BLIS_LIBRARIES})
target_link_libraries(test_t1m PRIVATE GTest::gtest_main)

gtest_discover_tests(test_t1m)