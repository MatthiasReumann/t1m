#pragma once

#include <complex>
#include "scatter_matrix.hpp"
#include "packing.hpp"
#include "std_ext.hpp"
#include "blis.h"

template <typename U>
inline void pack_A_1m(ScatterMatrix<std::complex<U>> *A, U *buffer, int off_i, int off_j, dim_t M, dim_t K, dim_t MR)
{
  const dim_t MR2 = 2 * MR;

  for (int i = 0; i < M; i += MR/2) // iterate over rows in MR/2 steps
  {
    for (int j = 0; j < K; j++) // iterate over columns
    {
      for (int k = 0; k < std_ext::min(MR, M - i - off_i); k++) // iterate over current row with width MR [k, k+MR)
      {
        const auto val = A->get(k + i + off_i, j + off_j);

        if (val != std::complex<U>(0))
        {
          buffer[2 * k + 2 * j * MR] = val.real();
          buffer[2 * k + 2 * j * MR + MR] = -val.imag();
          buffer[2 * k + 2 * j * MR + 1] = val.imag();
          buffer[2 * k + 2 * j * MR + MR + 1] = val.real();
        }
      }
    }

    buffer += MR * (2 * K);
  }
}

template <typename U>
inline void pack_B_1m(ScatterMatrix<std::complex<U>> *B, U *buffer, int off_i, int off_j, dim_t K, dim_t N, dim_t NR)
{
  const dim_t NR2 = 2 * NR;

  for (int j = 0; j < N; j += NR) // iterate over columns in NR steps
  {
    for (int i = 0; i < K; i++) // iterate over rows
    {
      for (int k = 0; k < std_ext::min(NR, N - j - off_j); k++) // iterate over current column with width NR [k, k+NR)
      {
        const auto val = B->get(i + off_i, k + j + off_j);

        if (val != std::complex<U>(0))
        {
          buffer[k + i * NR2] = val.real();
          buffer[k + i * NR2 + NR] = val.imag();
        }
      }
    }
    buffer += NR * (2 * K);
  }
}

template <typename U>
void unpack_C_1m(ScatterMatrix<std::complex<U>> *C, U *buffer, int off_i, int off_j, dim_t M, dim_t N)
{
  std::complex<U> *buffer_complex = reinterpret_cast<std::complex<U> *>(buffer);
  std::complex<U> *ptr = C->data();

  for (int j = 0; j < N; j++)
  {
    for (int i = 0; i < M; i++)
    {
      ptr[C->location(i + off_i, j + off_j)] += buffer_complex[i + j * M];
    }
  }
}