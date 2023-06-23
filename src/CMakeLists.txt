file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${TFCTC_SOURCE_DIR}/include/tfctc/*.hpp")

add_library(tfctc tfctc.cpp "${HEADER_LIST}")
target_include_directories(tfctc PUBLIC "../include" "${MARRAY_H}" "${BLIS_H}")
target_link_libraries(tfctc PRIVATE "${BLIS_LIB}")
target_compile_features(tfctc PUBLIC cxx_std_17)

source_group(
  TREE "${PROJECT_SOURCE_DIR}/include"
  PREFIX "Header Files"
  FILES ${HEADER_LIST})
