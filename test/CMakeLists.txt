FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest
)
FetchContent_MakeAvailable(doctest)

add_executable(test_utils test_utils.cpp)
target_link_libraries(test_utils PUBLIC doctest::doctest)
target_link_libraries(test_utils PRIVATE t1m)
add_test(NAME test_utilstest COMMAND test_utils)

# add_executable(testcomplex test_complex.cpp)
# target_link_libraries(testcomplex PUBLIC doctest::doctest)
# target_link_libraries(testcomplex PRIVATE t1m)
# add_test(NAME testcomplextest COMMAND testcomplex)

# add_executable(testreal test_real.cpp)
# target_link_libraries(testreal PUBLIC doctest::doctest)
# target_link_libraries(testreal PRIVATE t1m)
# add_test(NAME testrealtest COMMAND testreal)