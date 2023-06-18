#pragma once

#include "scatter_matrix.hpp"
#include "block_scatter_matrix.hpp"
#include "blis.h"

namespace tfctc
{
  namespace internal
  {
    template <typename T>
    struct gemm_context
    {
      const cntx_t *cntx;
      dim_t NC;
      dim_t KC;
      dim_t MC;
      dim_t NR;
      dim_t MR;
      ScatterMatrix<T> *A;
      ScatterMatrix<T> *B;
      BlockScatterMatrix<T> *C;
      T *alpha;
      T *beta;
      void (*kernel)(dim_t,
                     dim_t,
                     dim_t,
                     const T *restrict,
                     const T *restrict,
                     const T *restrict,
                     const T *restrict,
                     T *restrict,
                     inc_t,
                     inc_t,
                     auxinfo_t *restrict,
                     const cntx_t *restrict);
      void (*pack_A)(ScatterMatrix<T> *, T *, int, int, dim_t, dim_t, dim_t);
      void (*pack_B)(ScatterMatrix<T> *, T *, int, int, dim_t, dim_t, dim_t);
      void (*unpack_C)(BlockScatterMatrix<T> *, T *, int, int, dim_t, dim_t);
    };

    template <typename T, typename U>
    struct gemm_context_1m
    {
      const cntx_t *cntx;
      dim_t NC;
      dim_t KC;
      dim_t MC;
      dim_t NR;
      dim_t MR;
      ScatterMatrix<T> *A;
      ScatterMatrix<T> *B;
      BlockScatterMatrix<T> *C;
      U *alpha;
      U *beta;
      void (*kernel)(dim_t,
                     dim_t,
                     dim_t,
                     const U *restrict,
                     const U *restrict,
                     const U *restrict,
                     const U *restrict,
                     U *restrict,
                     inc_t,
                     inc_t,
                     auxinfo_t *restrict,
                     const cntx_t *restrict);
      void (*pack_A)(ScatterMatrix<T> *, U *, int, int, dim_t, dim_t, dim_t);
      void (*pack_B)(ScatterMatrix<T> *, U *, int, int, dim_t, dim_t, dim_t);
      void (*unpack_C)(BlockScatterMatrix<T> *, U *, int, int, dim_t, dim_t);
    };
  };
};