#pragma once

#include "scatter_matrix.hpp"
#include "blis.h"

namespace tfctc {
  namespace internal {
    template <typename T>
    void pack_as_cont(T* A, T* buffer, dim_t m1, dim_t k1, const dim_t MR, inc_t rs, inc_t cs)
    {
      for (size_t j = 0; j < k1; j++)
        for (size_t i = 0; i < m1; i++)
          buffer[i + j * MR] = A[j * cs + i * rs];
    }

    template <typename T>
    void pack_as_scat(BlockScatterMatrix<T>* A, T* buffer, dim_t m1, dim_t k1, const dim_t MR, size_t off_i, size_t off_j)
    {
      for (size_t i = 0; i < m1; i++)
        for (size_t j = 0; j < k1; j++)
          buffer[i + j * MR] = A->get(i + off_i, j + off_j);
    }

    template <typename T>
    void pack_bs_cont(T* B, T* buffer, dim_t k, dim_t n, const dim_t NR, inc_t rs, inc_t cs)
    {
      for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < k; i++)
          buffer[j + i * NR] = B[j * cs + i * rs];
    }

    template <typename T>
    void pack_bs_scat(BlockScatterMatrix<T>* B, T* buffer, dim_t k1, dim_t n1, const dim_t NR, size_t off_i, size_t off_j)
    {
      for (size_t i = 0; i < k1; i++)
        for (size_t j = 0; j < n1; j++)
          buffer[j + i * NR] = B->get(i + off_i, j + off_j);
    }
  };
};