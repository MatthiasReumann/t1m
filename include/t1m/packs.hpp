#pragma once

#include "scatter_matrix.hpp"
#include "blis.h"

namespace t1m::internal {
  template <typename T>
  void pack_as_cont(T* A, T* buffer, dim_t m, dim_t k, const dim_t MR, inc_t rs, inc_t cs)
  {
    for (size_t j = 0; j < k; j++) {
#pragma omp simd
      for (size_t i = 0; i < m; i++) {
        buffer[i + j * MR] = A[j * cs + i * rs];
      }
    }
  }

  template <typename T>
  void pack_as_scat(BlockScatterMatrix<T>* A, T* buffer, dim_t m, dim_t k, const dim_t MR, size_t off_i, size_t off_j)
  {
    for (size_t i = 0; i < m; i++)
      for (size_t j = 0; j < k; j++)
        buffer[i + j * MR] = (*A)(i + off_i, j + off_j);
  }

  template <typename T>
  void pack_bs_cont(T* B, T* buffer, dim_t k, dim_t n, const dim_t NR, inc_t rs, inc_t cs)
  {
    for (size_t j = 0; j < n; j++) {
#pragma omp simd
      for (size_t i = 0; i < k; i++) {
        buffer[j + i * NR] = B[j * cs + i * rs];
      }
    }
  }

  template <typename T>
  void pack_bs_scat(BlockScatterMatrix<T>* B, T* buffer, dim_t k, dim_t n, const dim_t NR, size_t off_i, size_t off_j)
  {
    for (size_t i = 0; i < k; i++)
      for (size_t j = 0; j < n; j++)
        buffer[j + i * NR] = (*B)(i + off_i, j + off_j);
  }
};