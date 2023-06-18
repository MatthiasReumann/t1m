#pragma once

#include "scatter_matrix.hpp"
#include "std_ext.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename T>
    void pack_A(ScatterMatrix<T> *A, T *buffer, int off_i, int off_j, dim_t M, dim_t K, dim_t MR)
    {
      for (int i = 0; i < M; i += MR)
      {
        for (int j = 0; j < K; j++)
        {
          for (int k = 0; k < tfctc::std_ext::min(MR, M - i - off_i); k++)
          {
            const auto val = A->get(k + off_i + i, j + off_j);
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
    void pack_B(ScatterMatrix<T> *B, T *buffer, int off_i, int off_j, dim_t K, dim_t N, dim_t NR)
    {
      for (int j = 0; j < N; j += NR)
      {
        for (int i = 0; i < K; i++)
        {
          for (int k = 0; k < tfctc::std_ext::min(NR, N - j - off_i); k++)
          {
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
    void unpack_C(BlockScatterMatrix<T> *C, T *buffer, int off_i, int off_j, dim_t m, dim_t n)
    {
      T *ptr = C->data();
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
