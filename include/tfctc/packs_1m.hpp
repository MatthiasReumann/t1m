#pragma once

#include <complex>
#include "packb_1m.hpp"
#include "block_scatter_matrix.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename U>
    void pack_1m_as_cont(std::complex<U>* A, U* ptr_a, dim_t m, dim_t k, const dim_t MR, inc_t rs, inc_t cs)
    {
      for (size_t j = 0; j < k; j++)
        for (size_t i = 0; i < m; i++)
          pack_1m_ab(ptr_a + 2 * i + 2 * j * MR, A[j * cs + i * rs], MR);
    }

    template <typename U>
    void pack_1m_as_scat(BlockScatterMatrix<std::complex<U>>* A, U* ptr_a, dim_t m, dim_t k, const dim_t MR, size_t off_i, size_t off_j)
    {
      for (size_t j = 0; j < k; j++)
        for (size_t i = 0; i < m; i++)
          pack_1m_ab(ptr_a + 2 * i + 2 * j * MR, (*A)(i + off_i, j + off_j), MR);
    }

    template <typename U>
    void pack_1m_bs_cont(std::complex<U>* B, U* ptr_b, dim_t k, dim_t n, const dim_t NR, inc_t rs, inc_t cs)
    {
      for (size_t j = 0; j < n; j++)
        for (size_t i = 0; i < k; i++)
          pack_1m_bb(ptr_b + j + 2 * i * NR, B[j * cs + i * rs], NR);
    }

    template <typename U>
    void pack_1m_bs_scat(BlockScatterMatrix<std::complex<U>>* B, U* ptr_b, dim_t k, dim_t n, const dim_t NR, size_t off_i, size_t off_j)
    {
      for (size_t i = 0; i < k; i++)
        for (size_t j = 0; j < n; j++)
          pack_1m_bb(ptr_b + j + 2 * i * NR, (*B)(i + off_i, j + off_j), NR);
    }

    template <typename U>
    void unpack_1m_c_cont(std::complex<U>* C, U* out, dim_t M, dim_t N, inc_t rs, inc_t cs)
    {
      M /= 2;
      for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
          C[i * rs + j * cs] += reinterpret_cast<std::complex<U>*>(out)[i + j * M];
    }

    template <typename U>
    void unpack_1m_c_scat(BlockScatterMatrix<std::complex<U>>* C, U* out, size_t off_i, size_t off_j, dim_t M, dim_t N)
    {
      std::complex<U>* ptr = C->data();
      M /= 2;
      for (int j = 0; j < N; j++)
        for (int i = 0; i < M; i++)
          // std::cout << reinterpret_cast<std::complex<U> *>(out)[i + j * M] << std::endl;
          ptr[C->location(i + off_i, j + off_j)] += reinterpret_cast<std::complex<U> *>(out)[i + j * M];
    }
  };
};