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
target_link_libraries(test_utils PUBLIC BLAS::BLAS)
target_link_libraries(test_utils PRIVATE GTest::gtest_main)

add_executable(test_packing test_packing.cpp)
target_link_libraries(test_packing PRIVATE t1m)
target_link_libraries(test_packing PUBLIC BLAS::BLAS)
target_link_libraries(test_packing PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_utils)
gtest_discover_tests(test_packing)

# add_executable(test_packing test_packing.cpp)
# target_link_libraries(test_packing PUBLIC BLAS::BLAS)
# target_link_libraries(test_packing PRIVATE doctest::doctest)
# target_link_libraries(test_packing INTERFACE t1m)
# add_test(NAME test_packingtest COMMAND test_packing)

# add_executable(test_real test_real.cpp)
# target_link_libraries(test_real PRIVATE doctest::doctest)
# target_link_libraries(test_real INTERFACE t1m)
# add_test(NAME test_realtest COMMAND test_real)

# add_executable(testcomplex test_complex.cpp)
# target_link_libraries(testcomplex PUBLIC doctest::doctest)
# target_link_libraries(testcomplex PRIVATE t1m)
# add_test(NAME testcomplextest COMMAND testcomplex)

# add_executable(testreal test_real.cpp)
# target_link_libraries(testreal PUBLIC doctest::doctest)
# target_link_libraries(testreal PRIVATE t1m)
# add_test(NAME testrealtest COMMAND testreal)