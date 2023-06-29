#pragma once

#include "std_ext.hpp"
#include "scatter_matrix.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename T>
    void pack_a(BlockScatterMatrix<T>* A, T* buffer, int off_i, int off_j, dim_t M, dim_t K, dim_t MR, dim_t KP)
    {
      dim_t m1, k1;
      size_t m, k, off_j_bak = off_j;
      inc_t rsa, csa;

      for(m = 0; m < size_t(M / MR); m++)
      {
        rsa = A->row_stride_in_block(m);

        for (k = 0; k < size_t(K / KP); k++)
        {
          csa = A->col_stride_in_block(k);

          if (rsa > 0 && csa > 0)
          {
            pack_as_cont(A, buffer, MR, KP, MR, off_i, off_j, rsa, csa);
          }
          else {
            pack_as_scat(A, buffer, MR, KP, MR, off_i, off_j);
          }

          buffer += MR * KP;
          off_j += KP;
        }

        k1 = K % KP;
        if (k1 > 0)
        {
          csa = A->col_stride_in_block(k);

          if (rsa > 0 && csa > 0)
          {
            pack_as_cont(A, buffer, MR, k1, MR, off_i, off_j, rsa, csa);
          }
          else {
            pack_as_scat(A, buffer, MR, k1, MR, off_i, off_j);
          }

          buffer += MR * k1;
        }

        off_i += MR;
        off_j = off_j_bak;
      }

      m1 = M % MR;
      if (m1 > 0)
      {
        rsa = A->row_stride_in_block(m);

        for (k = 0; k < size_t(K / KP); k++)
        {
          csa = A->col_stride_in_block(k);

          if (rsa > 0 && csa > 0)
          {
            pack_as_cont(A, buffer, m1, KP, MR, off_i, off_j, rsa, csa);
          }
          else {
            pack_as_scat(A, buffer, m1, KP, MR, off_i, off_j);
          }

          buffer += MR * KP;
          off_j += KP;
        }

        k1 = K % KP;
        if (k1 > 0)
        {
          csa = A->col_stride_in_block(k);

          if (rsa > 0 && csa > 0)
          {
            pack_as_cont(A, buffer, m1, k1, MR, off_i, off_j, rsa, csa);
          }
          else {
            pack_as_scat(A, buffer, m1, k1, MR, off_i, off_j);
          }
        }
      }
    }

    template <typename T>
    void pack_b(BlockScatterMatrix<T>* B, T* buffer, int off_i, int off_j, dim_t K, dim_t N, dim_t NR, dim_t KP)
    {
      dim_t k1, n1;
      size_t k, n, off_i_bak;
      inc_t rsb, csb;

      off_i_bak = off_i;

      for (n = 0; n < size_t(N / NR); n++)
      {
        csb = B->col_stride_in_block(n);

        for (k = 0; k < size_t(K / KP); k++)
        {
          rsb = B->row_stride_in_block(k);

          if (rsb > 0 && csb > 0)
          {
            pack_bs_cont(B, buffer, KP, NR, NR, off_i, off_j, rsb, csb);
          }
          else {
            pack_bs_scat(B, buffer, KP, NR, NR, off_i, off_j);
          }

          buffer += NR * KP;
          off_i += KP;
        }

        k1 = K % KP;
        if (k1 > 0)
        {
          rsb = B->row_stride_in_block(k);

          if (rsb > 0 && csb > 0)
          {
            pack_bs_cont(B, buffer, k1, NR, NR, off_i, off_j, rsb, csb);
          }
          else {
            pack_bs_scat(B, buffer, k1, NR, NR, off_i, off_j);
          }

          buffer += NR * k1;
        }

        off_j += NR;
        off_i = off_i_bak;
      }

      n1 = N % NR;
      if (n1 > 0)
      {
        csb = B->col_stride_in_block(n);

        for (k = 0; k < size_t(K / KP); k++)
        {
          rsb = B->row_stride_in_block(k);

          if (rsb > 0 && csb > 0)
          {
            pack_bs_cont(B, buffer, KP, n1, NR, off_i, off_j, rsb, csb);
          }
          else {
            pack_bs_scat(B, buffer, KP, n1, NR, off_i, off_j);
          }

          buffer += NR * KP;
          off_i += KP;
        }

        k1 = K % KP;
        if (k1 > 0)
        {
          rsb = B->row_stride_in_block(k);

          if (rsb > 0 && csb > 0)
          {
            pack_bs_cont(B, buffer, k1, n1, NR, off_i, off_j, rsb, csb);
          }
          else {
            pack_bs_scat(B, buffer, k1, n1, NR, off_i, off_j);
          }
        }
      }
    }

    template <typename T>
    void unpack_c_scat(BlockScatterMatrix<T>* C, T* buffer, int off_i, int off_j, dim_t m, dim_t n)
    {
      T* ptr = C->data();
      for (int j = 0; j < n; j++)
      {
        for (int i = 0; i < m; i++)
        {
          ptr[C->location(i + off_i, j + off_j)] += buffer[i + j * m];
        }
      }
    }

    template <typename T>
    void pack_as_cont(ScatterMatrix<T>* A, T* buffer, dim_t m1, dim_t k1, dim_t MR, size_t off_i, size_t off_j, inc_t rs, inc_t cs)
    {
      const auto ptr = A->pointer_at_loc(off_i, off_j);
      for (int j = 0; j < k1; j++)
      {
        for (int i = 0; i < m1; i++)
        {
            const auto val = ptr[j * cs + i * rs];
            if (val != T(0)) buffer[i + j * MR] = val;
        }
      }
    }

    template <typename T>
    void pack_as_scat(ScatterMatrix<T>* A, T* buffer, dim_t m1, dim_t k1, dim_t MR, size_t off_i, size_t off_j)
    {
      for (int i = 0; i < m1; i++)
      {
        for (int j = 0; j < k1; j++)
        {
            const auto val = A->get(i + off_i, j + off_j);
            if (val != T(0)) buffer[i + j * MR] = val;
        }
      }
    }

    template <typename T>
    inline void pack_bs_cont(BlockScatterMatrix<T>* B, T* buffer, dim_t k1, dim_t n1, dim_t NR, size_t off_i, size_t off_j, inc_t rs, inc_t cs)
    {
      const auto ptr = B->pointer_at_loc(off_i, off_j);
      for (int j = 0; j < n1; j++)
      {
        for (int i = 0; i < k1; i++)
        {
          const auto val = ptr[j * cs + i * rs];
          if (val != T(0)) buffer[j + i * NR] = val;
        }
      }
    }


    template <typename T>
    inline void pack_bs_scat(BlockScatterMatrix<T>* B, T* buffer, dim_t k1, dim_t n1, dim_t NR, size_t off_i, size_t off_j)
    {
      for (int i = 0; i < k1; i++)
      {
        for (int j = 0; j < n1; j++)
        {
          const auto val = B->get(i + off_i, j + off_j);
          if (val != T(0)) buffer[j + i * NR] = val;
        }
      }
    }
  };
};
