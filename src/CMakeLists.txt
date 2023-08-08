file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${t1m_SOURCE_DIR}/include/t1m/*.hpp")

add_library(t1m t1m.cpp "${HEADER_LIST}")
target_include_directories(t1m PUBLIC "../include" "${MARRAY_H}" "${BLIS_H}")
target_link_libraries(t1m PRIVATE "${BLIS_LIB}")
if(OpenMP_CXX_FOUND)
    target_link_libraries(t1m PUBLIC OpenMP::OpenMP_CXX)
endif()
target_compile_features(t1m PUBLIC cxx_std_17)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
