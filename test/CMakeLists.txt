FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils PRIVATE t1m)
target_link_libraries(test_utils PRIVATE GTest::gtest_main)

add_executable(test_packing test_packing.cpp)
target_link_libraries(test_packing PRIVATE t1m)
target_link_libraries(test_packing PRIVATE GTest::gtest_main)

add_executable(test_contraction test_contraction.cpp)
target_include_directories(t1m INTERFACE "${BLIS_H}")
target_link_libraries(test_contraction PRIVATE t1m)
target_link_libraries(test_contraction PRIVATE "${BLIS_LIB}")
target_link_libraries(test_contraction PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_utils)
gtest_discover_tests(test_packing)
gtest_discover_tests(test_contraction)