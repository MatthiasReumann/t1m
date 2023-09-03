#pragma once

#include <complex>
#include "block_scatter_matrix.hpp"
#include "blis.h"

namespace t1m::internal
{
  template <typename U>
  void pack_1m_as_cont(std::complex<U>* A, U* ptr_a, dim_t m, dim_t k, const dim_t MR, inc_t rs, inc_t cs)
  {
    for (size_t j = 0; j < k; j++)
    {
#pragma omp simd
      for (size_t i = 0; i < m; i++)
      {
        const auto val = A[j * cs + i * rs];
        ptr_a[2 * i + 2 * j * MR] = val.real();
        ptr_a[2 * i + 2 * j * MR + 1] = val.imag();
        ptr_a[2 * i + 2 * j * MR + MR] = -val.imag();
        ptr_a[2 * i + 2 * j * MR + MR + 1] = val.real();
      }
    }
  }

  template <typename U>
  void pack_1m_as_scat(BlockScatterMatrix<std::complex<U>>* A, U* ptr_a, dim_t m, dim_t k, const dim_t MR, size_t off_i, size_t off_j)
  {
    for (size_t j = 0; j < k; j++)
      for (size_t i = 0; i < m; i++)
      {
        const auto val = (*A)(i + off_i, j + off_j);
        ptr_a[2 * i + 2 * j * MR] = val.real();
        ptr_a[2 * i + 2 * j * MR + 1] = val.imag();
        ptr_a[2 * i + 2 * j * MR + MR] = -val.imag();
        ptr_a[2 * i + 2 * j * MR + MR + 1] = val.real();
      }
  }

  template <typename U>
  void pack_1m_bs_cont(std::complex<U>* B, U* ptr_b, dim_t k, dim_t n, const dim_t NR, inc_t rs, inc_t cs)
  {
    for (size_t j = 0; j < n; j++)
    {
#pragma omp simd
      for (size_t i = 0; i < k; i++)
      {
        const auto val = B[j * cs + i * rs];
        ptr_b[j + 2 * i * NR] = val.real();
        ptr_b[j + 2 * i * NR + NR] = val.imag();
      }
    }
  }

  template <typename U>
  void pack_1m_bs_scat(BlockScatterMatrix<std::complex<U>>* B, U* ptr_b, dim_t k, dim_t n, const dim_t NR, size_t off_i, size_t off_j)
  {
    for (size_t i = 0; i < k; i++)
      for (size_t j = 0; j < n; j++)
      {
        const auto val = (*B)(i + off_i, j + off_j);
        ptr_b[j + 2 * i * NR] = val.real();
        ptr_b[j + 2 * i * NR + NR] = val.imag();
      }
  }

  template <typename U>
  void unpack_1m_c_cont(std::complex<U>* C, U* out, dim_t M, dim_t N, inc_t rs, inc_t cs)
  {
    M /= 2;
    {
      for (int j = 0; j < N; j++)
      {
#pragma omp simd
        for (int i = 0; i < M; i++)
          C[i * rs + j * cs] += reinterpret_cast<std::complex<U>*>(out)[i + j * M];
      }
    }
  }

  template <typename U>
  void unpack_1m_c_scat(BlockScatterMatrix<std::complex<U>>* C, U* out, size_t off_i, size_t off_j, dim_t M, dim_t N)
  {
    std::complex<U>* ptr = C->data();
    M /= 2;
    for (int j = 0; j < N; j++)
      for (int i = 0; i < M; i++)
        ptr[C->location(i + off_i, j + off_j)] += reinterpret_cast<std::complex<U> *>(out)[i + j * M];
  }
};