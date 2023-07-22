add_executable(random benchmark_complex.cpp)
target_compile_features(random PRIVATE cxx_std_17)
target_link_libraries(random PRIVATE tfctc)

add_executable(explicit benchmark_complex_explicit.cpp)
target_compile_features(explicit PRIVATE cxx_std_17)
target_link_libraries(explicit PRIVATE tfctc)