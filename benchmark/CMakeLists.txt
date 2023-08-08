add_executable(scaling benchmark_complex.cpp)
target_compile_features(scaling PRIVATE cxx_std_17)
target_link_libraries(scaling PRIVATE t1m)

add_executable(explicit benchmark_complex_explicit.cpp)
target_compile_features(explicit PRIVATE cxx_std_17)
target_link_libraries(explicit PRIVATE t1m)