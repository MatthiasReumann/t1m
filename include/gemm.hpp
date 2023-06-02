#pragma once

#include "scatter_matrix.hpp"
#include "std_ext.hpp"
#include "utils.hpp"
#include "blis.h"

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
  ScatterMatrix<T> *C;
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
  void (*unpack_C)(ScatterMatrix<T> *, T *, int, int, dim_t, dim_t);
};

template <typename T>
void gemm_internal(const gemm_context<T> *gemm_ctx)
{
  ScatterMatrix<T> *A = gemm_ctx->A;
  ScatterMatrix<T> *B = gemm_ctx->B;
  ScatterMatrix<T> *C = gemm_ctx->C;

  const dim_t NC = gemm_ctx->NC;
  const dim_t KC = gemm_ctx->KC;
  const dim_t MC = gemm_ctx->MC;
  const dim_t NR = gemm_ctx->NR;
  const dim_t MR = gemm_ctx->MR;

  const size_t M = A->row_size();
  const size_t K = A->col_size();
  const size_t N = B->col_size();

  T *buf = nullptr;
  T *A_tilde = nullptr; // A in G^{MC x KC}
  T *B_tilde = nullptr; // B in G^{KC x NC}
  T *C_tilde = nullptr; // C in G^{MC x NC}

  alloc_aligned<T>(&buf, MC * KC + KC * NC + MC * NC);

  A_tilde = buf;
  B_tilde = buf + MC * KC;
  C_tilde = buf + MC * KC + KC * NC;

  dim_t m1, n1, k1, m, n;
  inc_t rsc = 1, csc;

  for (int j_c = 0; j_c < N; j_c += NC)
  {
    for (int p_c = 0; p_c < K; p_c += KC)
    {
      k1 = std_ext::min(KC, static_cast<dim_t>(K - p_c));
      n1 = std_ext::min(NC, static_cast<dim_t>(N - j_c));

      gemm_ctx->pack_B(B, B_tilde, p_c, j_c, k1, n1, NR);

      for (int i_c = 0; i_c < M; i_c += MC)
      {
        m1 = std_ext::min(MC, static_cast<dim_t>(M - i_c));

        gemm_ctx->pack_A(A, A_tilde, i_c, p_c, m1, k1, MR);

        for (int j_r = 0; j_r < n1; j_r += NR)
        {
          n = std_ext::min(NR, n1 - j_r);

          for (int i_r = 0; i_r < m1; i_r += MR)
          {
            m = std_ext::min(MR, m1 - i_r);
            csc = m;

            gemm_ctx->kernel(m,
                             n,
                             k1,
                             gemm_ctx->alpha,
                             A_tilde + i_r * k1,
                             B_tilde + j_r * k1,
                             gemm_ctx->beta,
                             C_tilde, rsc, csc,
                             NULL,
                             gemm_ctx->cntx);
            gemm_ctx->unpack_C(C, C_tilde, i_c, j_c, m, n);
          }
        }
      }
    }
  }

  free(buf);
}

void gemm(double *alpha, ScatterMatrix<double> *A, ScatterMatrix<double> *B, double *beta, ScatterMatrix<double> *C, const cntx_t *cntx)
{
  const gemm_context<double> gemm_ctx = {
      .cntx = cntx,
      .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx),
      .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx),
      .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx),
      .NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx),
      .MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx),
      .A = A,
      .B = B,
      .C = C,
      .alpha = alpha,
      .beta = beta,
      .kernel = bli_dgemm_ukernel,
      .pack_A = pack_A<double>,
      .pack_B = pack_B<double>,
      .unpack_C = unpack_C<double>,
  };

  gemm_internal<double>(&gemm_ctx);
}

void gemm(float *alpha, ScatterMatrix<float> *A, ScatterMatrix<float> *B, float *beta, ScatterMatrix<float> *C, const cntx_t *cntx)
{
  const gemm_context<float> gemm_ctx = {
      .cntx = cntx,
      .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx),
      .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx),
      .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx),
      .NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx),
      .MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx),
      .A = A,
      .B = B,
      .C = C,
      .alpha = alpha,
      .beta = beta,
      .kernel = bli_sgemm_ukernel,
      .pack_A = pack_A<float>,
      .pack_B = pack_B<float>,
      .unpack_C = unpack_C<float>,
  };

  gemm_internal<float>(&gemm_ctx);
}