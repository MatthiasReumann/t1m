#pragma once

#include <complex>
#include "std_ext.hpp"
#include "block_scatter_matrix.hpp"
#include "packing.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename U>
    void pack_1m_a(BlockScatterMatrix<std::complex<U>>* A, U* buffer, int off_i, int off_j, dim_t M, dim_t K, const dim_t MR, const dim_t KP)
    {
      const size_t M_blocks = size_t(M / MR);
      const size_t K_blocks = size_t(K / KP);

      const size_t m1 = M % MR;
      const size_t k1 = K % KP;

      const size_t off_j_bak = off_j;
      const size_t start_b_m = size_t(off_i / MR);
      const size_t start_b_k = size_t(off_j / KP);

      // next sliver offsets
      const size_t offns_fullblocks = (K - off_j_bak) * 2 * MR;
      const size_t offns = offns_fullblocks + 2 * k1 * MR;

      const dim_t HALFMR = MR / 2;

      size_t m, k, bm;
      inc_t rsa, csa;

      for (m = start_b_m; m < M_blocks; m++)
      {
        rsa = A->row_stride_in_block(m);

        for (k = start_b_k; k < K_blocks; k++)
        {
          csa = A->col_stride_in_block(k);

          if (rsa > 0 && csa > 0)
          {
            pack_1m_as_cont(A, buffer, HALFMR, KP, MR, off_i, off_j, rsa, csa);
            pack_1m_as_cont(A, buffer + offns, HALFMR, KP, MR, off_i + HALFMR, off_j, rsa, csa);
          }
          else {
            pack_1m_as_scat(A, buffer, HALFMR, KP, MR, off_i, off_j);
            pack_1m_as_scat(A, buffer + offns, HALFMR, KP, MR, off_i + HALFMR, off_j);
          }

          buffer += 2 * MR * KP;
          off_j += KP;
        }

        if (k1 > 0)
        {
          csa = A->col_stride_in_block(k);

          if (rsa > 0 && csa > 0)
          {
            pack_1m_as_cont(A, buffer, HALFMR, k1, MR, off_i, off_j, rsa, csa);
            pack_1m_as_cont(A, buffer + offns, HALFMR, k1, MR, off_i + HALFMR, off_j, rsa, csa);
          }
          else {
            pack_1m_as_scat(A, buffer, HALFMR, k1, MR, off_i, off_j);
            pack_1m_as_scat(A, buffer + offns, HALFMR, k1, MR, off_i + HALFMR, off_j);
          }

          buffer += 2 * MR * k1;
        }

        off_i += MR;
        off_j = off_j_bak;

        buffer += offns;
      }

      if (m1 > 0)
      {
        rsa = A->row_stride_in_block(m);

        if (m1 > HALFMR)
        {
          for (k = start_b_k; k < K_blocks; k++)
          {
            csa = A->col_stride_in_block(k);

            if (rsa > 0 && csa > 0)
            {
              pack_1m_as_cont(A, buffer, HALFMR, KP, MR, off_i, off_j, rsa, csa);
              pack_1m_as_cont(A, buffer + offns, m1 - HALFMR, KP, MR, off_i + HALFMR, off_j, rsa, csa);
            }
            else {
              pack_1m_as_scat(A, buffer, HALFMR, KP, MR, off_i, off_j);
              pack_1m_as_scat(A, buffer, m1 - HALFMR, KP, MR, off_i + HALFMR, off_j);
            }

            buffer += 2 * MR * KP;
            off_j += KP;
          }

          if (k1 > 0)
          {
            csa = A->col_stride_in_block(k);

            if (rsa > 0 && csa > 0)
            {
              pack_1m_as_cont(A, buffer, HALFMR, k1, MR, off_i, off_j, rsa, csa);
              pack_1m_as_cont(A, buffer + offns_fullblocks, m1 - HALFMR, k1, MR, off_i + HALFMR, off_j, rsa, csa);
            }
            else {
              pack_1m_as_scat(A, buffer, HALFMR, k1, MR, off_i, off_j);
              pack_1m_as_scat(A, buffer + offns_fullblocks, m1 - HALFMR, k1, MR, off_i + HALFMR, off_j);
            }
          }
        }
        else
        {
          for (k = start_b_k; k < K_blocks; k++)
          {
            csa = A->col_stride_in_block(k);

            if (rsa > 0 && csa > 0)
            {
              pack_1m_as_cont(A, buffer, m1, KP, MR, off_i, off_j, rsa, csa);
            }
            else {
              pack_1m_as_scat(A, buffer, m1, KP, MR, off_i, off_j);
            }

            buffer += 2 * MR * KP;
            off_j += KP;
          }

          if (k1 > 0)
          {
            csa = A->col_stride_in_block(k);

            if (rsa > 0 && csa > 0)
            {
              pack_1m_as_cont(A, buffer, m1, k1, MR, off_i, off_j, rsa, csa);
            }
            else {
              pack_1m_as_scat(A, buffer, m1, k1, MR, off_i, off_j);
            }
          }
        }
      }
    }

    template <typename U>
    void pack_1m_b(BlockScatterMatrix<std::complex<U>>* B, U* buffer, int off_i, int off_j, dim_t K, dim_t N, const dim_t NR, const dim_t KP)
    {
      const dim_t k1 = K % KP;
      const dim_t n1 = N % NR;
      const size_t off_i_bak = off_i;

      const size_t start_b_k = size_t(off_i / KP);
      const size_t start_b_n = size_t(off_j / NR);

      size_t k, n;
      inc_t rsb, csb;

      for (n = start_b_n; n < size_t(N / NR); n++)
      {
        csb = B->col_stride_in_block(n);

        for (k = start_b_k; k < size_t(K / KP); k++)
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

        for (k = start_b_k; k < size_t(K / KP); k++)
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
    void pack_1m_as_cont(BlockScatterMatrix<std::complex<U>>* A, U* buffer, dim_t m, dim_t k, const dim_t MR, size_t off_i, size_t off_j, inc_t rs, inc_t cs)
    {
      const auto ptr = A->pointer_at_loc(off_i, off_j);
      U* base = buffer;

      for (size_t j = 0; j < k; j++)
      {
        for (size_t i = 0; i < m; i++)
        {
          const auto val = ptr[j * cs + i * rs];
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
