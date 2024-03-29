#pragma once

#include "packs.hpp"
#include "scatter_matrix.hpp"
#include "blis.h"

namespace t1m::internal {
  template <typename T>
  void pack_a(BlockScatterMatrix<T>* A, T* buffer, size_t off_i, size_t off_j, dim_t M, dim_t K, const dim_t MR, const dim_t KP)
  {
    const size_t M_blocks = size_t(M / MR);
    const size_t K_blocks = size_t(K / KP);

    const dim_t m1 = M % MR;
    const dim_t k1 = K % KP;

    const size_t off_j_bak = off_j;
    const size_t start_b_m = size_t(off_i / MR);
    const size_t start_b_k = size_t(off_j / KP);

    size_t m, k;
    inc_t rsa, csa;

    for (m = start_b_m; m < M_blocks; m++)
    {
      rsa = A->row_stride_in_block(m);

      for (k = start_b_k; k < K_blocks; k++)
      {
        csa = A->col_stride_in_block(k);

        if (rsa > 0 && csa > 0)
        {
          pack_as_cont(A->pointer_at_loc(off_i, off_j), buffer, MR, KP, MR, rsa, csa);
        }
        else {
          pack_as_scat(A, buffer, MR, KP, MR, off_i, off_j);
        }

        buffer += MR * KP;
        off_j += KP;
      }

      if (k1 > 0)
      {
        csa = A->col_stride_in_block(k);

        if (rsa > 0 && csa > 0)
        {
          pack_as_cont(A->pointer_at_loc(off_i, off_j), buffer, MR, k1, MR, rsa, csa);
        }
        else {
          pack_as_scat(A, buffer, MR, k1, MR, off_i, off_j);
        }

        buffer += MR * k1;
      }

      off_i += MR;
      off_j = off_j_bak;
    }

    if (m1 > 0)
    {
      rsa = A->row_stride_in_block(m);

      for (k = start_b_k; k < K_blocks; k++)
      {
        csa = A->col_stride_in_block(k);

        if (rsa > 0 && csa > 0)
        {
          pack_as_cont(A->pointer_at_loc(off_i, off_j), buffer, m1, KP, MR, rsa, csa);
        }
        else {
          pack_as_scat(A, buffer, m1, KP, MR, off_i, off_j);
        }

        buffer += MR * KP;
        off_j += KP;
      }

      if (k1 > 0)
      {
        csa = A->col_stride_in_block(k);

        if (rsa > 0 && csa > 0)
        {
          pack_as_cont(A->pointer_at_loc(off_i, off_j), buffer, m1, k1, MR, rsa, csa);
        }
        else {
          pack_as_scat(A, buffer, m1, k1, MR, off_i, off_j);
        }
      }
    }
  }

  template <typename T>
  void pack_b(BlockScatterMatrix<T>* B, T* buffer, size_t off_i, size_t off_j, dim_t K, dim_t N, const dim_t NR, const dim_t KP)
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
          pack_bs_cont(B->pointer_at_loc(off_i, off_j), buffer, KP, NR, NR, rsb, csb);
        }
        else {
          pack_bs_scat(B, buffer, KP, NR, NR, off_i, off_j);
        }

        buffer += NR * KP;
        off_i += KP;
      }

      if (k1 > 0)
      {
        rsb = B->row_stride_in_block(k);

        if (rsb > 0 && csb > 0)
        {
          pack_bs_cont(B->pointer_at_loc(off_i, off_j), buffer, k1, NR, NR, rsb, csb);
        }
        else {
          pack_bs_scat(B, buffer, k1, NR, NR, off_i, off_j);
        }

        buffer += NR * k1;
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
          pack_bs_cont(B->pointer_at_loc(off_i, off_j), buffer, KP, n1, NR, rsb, csb);
        }
        else {
          pack_bs_scat(B, buffer, KP, n1, NR, off_i, off_j);
        }

        buffer += NR * KP;
        off_i += KP;
      }

      if (k1 > 0)
      {
        rsb = B->row_stride_in_block(k);

        if (rsb > 0 && csb > 0)
        {
          pack_bs_cont(B->pointer_at_loc(off_i, off_j), buffer, k1, n1, NR, rsb, csb);
        }
        else {
          pack_bs_scat(B, buffer, k1, n1, NR, off_i, off_j);
        }
      }
    }
  }

  template <typename T>
  void unpack_c_scat(BlockScatterMatrix<T>* C, T* buffer, size_t off_i, size_t off_j, dim_t m, dim_t n)
  {
    T* ptr = C->data();
    for (size_t j = 0; j < n; j++)
    {
      for (size_t i = 0; i < m; i++)
      {
        ptr[C->location(i + off_i, j + off_j)] += buffer[i + j * m];
      }
    }
  }
};
