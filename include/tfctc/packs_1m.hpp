#pragma once

#include <complex>
#include "block_scatter_matrix.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename U>
    void unpack_1m_c_scat(BlockScatterMatrix<std::complex<U>>* C, U* out, size_t off_i, size_t off_j, dim_t M, dim_t N)
    {
      const auto buffer_complex = reinterpret_cast<std::complex<U> *>(out);

      std::complex<U>* ptr = C->data();

      M /= 2;

      for (int j = 0; j < N; j++)
      {
        for (int i = 0; i < M; i++)
        {
          ptr[C->location(i + off_i, j + off_j)] += buffer_complex[i + j * M];
        }
      }
    }

    template <typename U>
    void unpack_1m_c_cont(std::complex<U>* C, U* out, dim_t M, dim_t N, inc_t rs, inc_t cs)
    {
      const auto buffer_complex = reinterpret_cast<std::complex<U>*>(out);

      M /= 2;

      for (int j = 0; j < N; j++)
      {
        for (int i = 0; i < M; i++)
        {
          C[i * rs + j * cs] += buffer_complex[i + j * M];
        }
      }
    }

    template <typename U>
    void pack_1m_as_cont(std::complex<U>* A, U* buffer, dim_t m, dim_t k, const dim_t MR, inc_t rs, inc_t cs)
    {
      U* base = buffer;

      for (size_t j = 0; j < k; j++)
      {
        for (size_t i = 0; i < m; i++)
        {
          const auto val = A[j * cs + i * rs];
          if (val != std::complex<U>(0))
          {
            buffer[0] = val.real(); buffer[MR] = -val.imag();
            buffer[1] = val.imag(); buffer[MR + 1] = val.real();
          }
          buffer += 2;
        }

        base += 2 * MR;
        buffer = base;
      }
    }

    template <typename U>
    void pack_1m_as_scat(BlockScatterMatrix<std::complex<U>>* A, U* buffer, dim_t m, dim_t k, const dim_t MR, size_t off_i, size_t off_j)
    {
      U* base = buffer;
      for (size_t j = 0; j < k; j++)
      {
        for (size_t i = 0; i < m; i++)
        {
          const auto val = A->get(i + off_i, j + off_j);
          if (val != std::complex<U>(0))
          {
            buffer[0] = val.real(); buffer[MR] = -val.imag();
            buffer[1] = val.imag(); buffer[MR + 1] = val.real();
          }
          buffer += 2;
        }

        base += 2 * MR;
        buffer = base;
      }
    }

    template <typename U>
    void pack_1m_bs_cont(BlockScatterMatrix<std::complex<U>>* B, U* buffer, dim_t k, dim_t n, const dim_t NR, size_t off_i, size_t off_j, inc_t rs, inc_t cs)
    {
      const auto ptr = B->pointer_at_loc(off_i, off_j);
      const dim_t NR2 = 2 * NR;
      for (size_t j = 0; j < n; j++)
      {
        for (size_t i = 0; i < k; i++)
        {
          const auto val = ptr[j * cs + i * rs];
          if (val != std::complex<U>(0))
          {
            buffer[j + i * NR2] = val.real();
            buffer[j + i * NR2 + NR] = val.imag();
          }
        }
      }
    }

    template <typename U>
    void pack_1m_bs_scat(BlockScatterMatrix<std::complex<U>>* B, U* buffer, dim_t k, dim_t n, const dim_t NR, size_t off_i, size_t off_j)
    {
      const dim_t NR2 = 2 * NR;
      for (size_t i = 0; i < k; i++)
      {
        for (size_t j = 0; j < n; j++)
        {
          const auto val = B->get(i + off_i, j + off_j);
          if (val != std::complex<U>(0))
          {
            buffer[j + i * NR2] = val.real();
            buffer[j + i * NR2 + NR] = val.imag();
          }
        }
      }
    }
  };
};