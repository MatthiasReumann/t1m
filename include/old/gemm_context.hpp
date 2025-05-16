// #pragma once

// #include "scatter_matrix.hpp"
// #include "block_scatter_matrix.hpp"
// #include "blis.h"

// namespace t1m::utils
// {
//   template <typename T>
//   struct gemm_context
//   {
//     const cntx_t *cntx;
//     const dim_t NC;
//     const dim_t KC;
//     const dim_t MC;
//     const dim_t NR;
//     const dim_t MR;
//     const dim_t KP;
//     BlockScatterMatrix<T> *A;
//     BlockScatterMatrix<T> *B;
//     BlockScatterMatrix<T> *C;
//     T *alpha;
//     T *beta;
//     void (*kernel)(dim_t,
//                     dim_t,
//                     dim_t,
//                     const T *restrict,
//                     const T *restrict,
//                     const T *restrict,
//                     const T *restrict,
//                     T *restrict,
//                     inc_t,
//                     inc_t,
//                     auxinfo_t *restrict,
//                     const cntx_t *restrict);
//   };

//   template <typename T, typename U>
//   struct gemm_context_1m
//   {
//     const cntx_t *cntx;
//     const dim_t NC;
//     const dim_t KC;
//     const dim_t MC;
//     const dim_t NR;
//     const dim_t MR;
//     const dim_t KP;
//     BlockScatterMatrix<T> *A;
//     BlockScatterMatrix<T> *B;
//     BlockScatterMatrix<T> *C;
//     U *alpha;
//     U *beta;
//     void (*kernel)(dim_t,
//                     dim_t,
//                     dim_t,
//                     const U *restrict,
//                     const U *restrict,
//                     const U *restrict,
//                     const U *restrict,
//                     U *restrict,
//                     inc_t,
//                     inc_t,
//                     auxinfo_t *restrict,
//                     const cntx_t *restrict);
//   };
// };