#pragma once

#include "scatter_matrix.hpp"
#include "blis.h"

template <typename T>
void pack_A(ScatterMatrix<T> *A, T *buffer, int off_i, int off_j, dim_t m, dim_t n, dim_t mr)
{
  const T *ptr = A->cdata();
  for (int i = 0; i < m; i += mr)
  {
    for (int j = 0; j < n; j++)
    {
      const int y = j + off_j;

      for (int k = 0; k < mr; k++)
      {
        const int x = k + off_i + i;

        if (x >= m)
          break;

        buffer[i * n + k + j * mr] = ptr[A->location(x, y)];
      }
    }
  }
}

template <typename T>
void pack_B(ScatterMatrix<T> *B, T *buffer, int off_i, int off_j, dim_t m, dim_t n, dim_t nr)
{
  const T *ptr = B->cdata();

  for (int j = 0; j < n; j += nr)
  {
    for (int i = 0; i < m; i++)
    {
      const int x = i + off_i;
      for (int k = 0; k < nr; k++)
      {
        const int y = k + off_j + j;
        if (y >= n)
          break;

        buffer[j * m + k + i * nr] = ptr[B->location(x, y)];
      }
    }
  }
}

template <typename T>
void unpack_C(ScatterMatrix<T> *C, T *buffer, int off_i, int off_j, dim_t m, dim_t n)
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