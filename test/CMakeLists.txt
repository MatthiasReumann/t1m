FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

file(GLOB test_sources "*.cpp") # Find all .cpp test files.

enable_testing()

add_executable(test_t1m ${test_sources})
target_link_libraries(test_t1m PRIVATE t1m)
target_link_libraries(test_t1m PRIVATE ${BLIS_LIBRARIES})
target_link_libraries(test_t1m PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_t1m)