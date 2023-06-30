#pragma once

#include <complex>
#include "std_ext.hpp"
#include "scatter_matrix.hpp"
#include "packing.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename U>
    void pack_1m_a(ScatterMatrix<std::complex<U>>* A, U* buffer, int off_i, int off_j, dim_t M, dim_t K, dim_t MR)
    {
      U* base = buffer;
      for (int i = 0; i < M; i += MR / 2) // iterate over rows in MR/2 steps
      {
        for (int j = 0; j < K; j++) // iterate over columns
        {
          if (j + off_j >= A->col_size()) break;

          for (int k = 0; k < MR / 2; k++) // iterate over current row with width MR/2 [k, k+MR)
          {
            if (k + i + off_i >= A->row_size()) break;

            const auto val = A->get(k + i + off_i, j + off_j);

            if (val != std::complex<U>(0))
            {
              buffer[0] = val.real(); buffer[MR] = -val.imag();
              buffer[1] = val.imag(); buffer[1 + MR] = val.real();
            }

            buffer += 2;
          }

          base += 2 * MR;
          buffer = base;
        }
      }
    }

    template <typename U>
    void pack_1m_b(BlockScatterMatrix<std::complex<U>>* B, U* buffer, int off_i, int off_j, dim_t K, dim_t N, dim_t NR, dim_t KP)
    {
      const dim_t k1 = K % KP;
      const dim_t n1 = N % NR;
      const size_t off_i_bak = off_i;

      size_t k, n;
      inc_t rsb, csb;

      for (n = 0; n < size_t(N / NR); n++)
      {
        csb = B->col_stride_in_block(n);

        for (k = 0; k < size_t(K / KP); k++)
        {
          rsb = B->row_stride_in_block(k);

          if (rsb > 0 && csb > 0)
          {
            pack_1m_bs_cont(B, buffer, KP, NR, NR, off_i, off_j, rsb, csb);
          }
          else {
            pack_1m_bs_scat(B, buffer, KP, NR, NR, off_i, off_j);
          }

          buffer += NR * (2 * KP);
          off_i += KP;
        }

        if (k1 > 0)
        {
          rsb = B->row_stride_in_block(k);

          if (rsb > 0 && csb > 0)
          {
            pack_1m_bs_cont(B, buffer, k1, NR, NR, off_i, off_j, rsb, csb);
          }
          else {
            pack_1m_bs_scat(B, buffer, k1, NR, NR, off_i, off_j);
          }

          buffer += NR * (2 * k1);
        }

        off_j += NR;
        off_i = off_i_bak;
      }

      if (n1 > 0)
      {
        csb = B->col_stride_in_block(n);

        for (k = 0; k < size_t(K / KP); k++)
        {
          rsb = B->row_stride_in_block(k);

          if (rsb > 0 && csb > 0)
          {
            pack_1m_bs_cont(B, buffer, KP, n1, NR, off_i, off_j, rsb, csb);
          }
          else {
            pack_1m_bs_scat(B, buffer, KP, n1, NR, off_i, off_j);
          }

          buffer += NR * (2 * KP);
          off_i += KP;
        }

        if (k1 > 0)
        {
          rsb = B->row_stride_in_block(k);

          if (rsb > 0 && csb > 0)
          {
            pack_1m_bs_cont(B, buffer, k1, n1, NR, off_i, off_j, rsb, csb);
          }
          else {
            pack_1m_bs_scat(B, buffer, k1, n1, NR, off_i, off_j);
          }
        }
      }
    }

    template <typename U>
    void unpack_1m_c(BlockScatterMatrix<std::complex<U>>* C, U* buffer, int off_i, int off_j, dim_t M, dim_t N)
    {
      std::complex<U>* buffer_complex = reinterpret_cast<std::complex<U> *>(buffer);
      std::complex<U>* ptr = C->data();

      for (int j = 0; j < N; j++)
      {
        for (int i = 0; i < M / 2; i++)
        {
          ptr[C->location(i + off_i, j + off_j)] += buffer_complex[i + j * M / 2];
        }
      }
    }

    template <typename U>
    inline void pack_1m_bs_cont(BlockScatterMatrix<std::complex<U>>* B, U* buffer, dim_t k, dim_t n, dim_t NR, size_t off_i, size_t off_j, inc_t rs, inc_t cs)
    {
      const dim_t NR2 = 2 * NR;
      const auto ptr = B->pointer_at_loc(off_i, off_j);
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
    inline void pack_1m_bs_scat(BlockScatterMatrix<std::complex<U>>* B, U* buffer, dim_t k, dim_t n, dim_t NR, size_t off_i, size_t off_j)
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
