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
            if (val != T(0))
            {
              buffer[k + j * MR] = val;
            }
          }
        }

        buffer += MR * K;
      }
    }

    template <typename T>
    void pack_B(ScatterMatrix<T>* B, T* buffer, int off_i, int off_j, dim_t K, dim_t N, dim_t NR)
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
            if (val != T(0))
            {
              buffer[k + i * NR] = val;
            }
          }
        }
        buffer += NR * K;
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
  };
};
