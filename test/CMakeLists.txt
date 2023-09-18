find_package(Git REQUIRED)

ExternalProject_Add(
    doctest
    PREFIX ${CMAKE_BINARY_DIR}/doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    TIMEOUT 10
    UPDATE_COMMAND ${GIT_EXECUTABLE} pull
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    LOG_DOWNLOAD ON
)

ExternalProject_Get_Property(doctest source_dir)
set(DOCTEST_INCLUDE_DIR ${source_dir}/doctest CACHE INTERNAL "Path to include folder for doctest")

add_executable(testcomplex test_complex.cpp)
target_include_directories(testcomplex PUBLIC ${DOCTEST_INCLUDE_DIR})
target_compile_features(testcomplex PRIVATE cxx_std_17)
target_link_libraries(testcomplex PRIVATE t1m)
add_test(NAME testcomplextest COMMAND testcomplex)

add_executable(testreal test_real.cpp)
target_include_directories(testreal PUBLIC ${DOCTEST_INCLUDE_DIR})
target_compile_features(testreal PRIVATE cxx_std_17)
target_link_libraries(testreal PRIVATE t1m)
add_test(NAME testrealtest COMMAND testreal)

add_executable(testscatter test_scatter.cpp)
target_include_directories(testscatter PUBLIC ${DOCTEST_INCLUDE_DIR})
target_compile_features(testscatter PRIVATE cxx_std_17)
target_link_libraries(testscatter PRIVATE t1m)
add_test(NAME testscattertest COMMAND testscatter)