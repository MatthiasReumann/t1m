#pragma once

#include "std_ext.hpp"
#include "scatter_matrix.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename T>
    void pack_A(ScatterMatrix<T>* A, T* buffer, int off_i, int off_j, dim_t M, dim_t K, dim_t MR)
    {
      for (int i = 0; i < M; i += MR)
      {
        for (int j = 0; j < K; j++)
        {
          if (j + off_j >= A->col_size()) break;
          for (int k = 0; k < MR; k++)
          {
            if (k + i + off_i >= A->row_size()) break;
            const auto val = A->get(k + i + off_i, j + off_j);
            if (val != T(0)) buffer[k + j * MR] = val;
          }
        }
        buffer += MR * K;
      }
    }

    template <typename T>
    void pack_B(BlockScatterMatrix<T>* B, T* buffer, int off_i, int off_j, dim_t K, dim_t N, dim_t NR)
    {
      for (int j = 0; j < N; j += NR)
      {
        for (int i = 0; i < K; i++)
        {
          if (i + off_i >= B->row_size()) break;
          for (int k = 0; k < NR; k++)
          {
            if (k + j + off_j >= B->col_size()) break;
            const auto val = B->get(i + off_i, k + off_j + j);
            if (val != T(0)) buffer[k + i * NR] = val;
          }
        }
        buffer += NR * K;
      }
    }

    template <typename T>
    void pack_B_regular(BlockScatterMatrix<T>* B, T* buffer, int off_i, int off_j, dim_t K, dim_t N, dim_t NR, dim_t KP)
    {
      std::cout << "N" << N << " " << NR << '\n' << "K" << K << " " << KP << '\n';
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
            pack_sliver_scatter(B, buffer, KP, NR, NR, off_i, off_j);
          }
          else {
            pack_sliver_scatter(B, buffer, KP, NR, NR, off_i, off_j);
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
            pack_sliver_scatter(B, buffer, KP, NR, NR, off_i, off_j);
          }
          else {
            pack_sliver_scatter(B, buffer, KP, NR, NR, off_i, off_j);
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
            pack_sliver_scatter(B, buffer, KP, n1, NR, off_i, off_j);
          }
          else {
            pack_sliver_scatter(B, buffer, KP, n1, NR, off_i, off_j);
          }

          buffer += NR * KP;
          off_i += KP;
        }

        k1 = K % KP;
        if (k1 > 0)
        {
          if (rsb > 0 && csb > 0)
          {
            pack_sliver_scatter(B, buffer, k1, n1, NR, off_i, off_j);
          }
          else {
            pack_sliver_scatter(B, buffer, k1, n1, NR, off_i, off_j);
          }
        }
      }
    }



    template <typename T>
    void unpack_C(BlockScatterMatrix<T>* C, T* buffer, int off_i, int off_j, dim_t m, dim_t n)
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
    inline void pack_sliver_scatter(BlockScatterMatrix<T>* B, T* buffer, dim_t k1, dim_t n1, dim_t NR, size_t off_i, size_t off_j)
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
