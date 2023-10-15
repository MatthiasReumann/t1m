FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest
)
FetchContent_MakeAvailable(doctest)

add_executable(testcomplex test_complex.cpp)
target_link_libraries(testcomplex PUBLIC doctest::doctest)
target_compile_features(testcomplex PRIVATE cxx_std_17)
target_link_libraries(testcomplex PRIVATE t1m)
add_test(NAME testcomplextest COMMAND testcomplex)

add_executable(testreal test_real.cpp)
target_link_libraries(testreal PUBLIC doctest::doctest)
target_compile_features(testreal PRIVATE cxx_std_17)
target_link_libraries(testreal PRIVATE t1m)
add_test(NAME testrealtest COMMAND testreal)

add_executable(testscatter test_scatter.cpp)
target_link_libraries(testscatter PUBLIC doctest::doctest)
target_compile_features(testscatter PRIVATE cxx_std_17)
target_link_libraries(testscatter PRIVATE t1m)
add_test(NAME testscattertest COMMAND testscatter)

add_executable(testcinterface test_cinterface.cpp)
target_link_libraries(testcinterface PUBLIC doctest::doctest)
target_compile_features(testcinterface PRIVATE cxx_std_17)
target_link_libraries(testcinterface PRIVATE t1m)
add_test(NAME testcinterface COMMAND testcinterface)
