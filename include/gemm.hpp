#pragma once

#include <complex>
#include "gemm_context.hpp"
#include "scatter_matrix.hpp"
#include "packing.hpp"
#include "packing_1m.hpp"
#include "std_ext.hpp"
#include "utils.hpp"
#include "blis.h"

void gemm_internal_complex(const gemm_context_1m<std::complex<float>, float> *gemm_ctx)
{
  const dim_t NC = gemm_ctx->NC;
  const dim_t KC = gemm_ctx->KC;
  const dim_t MC = gemm_ctx->MC;
  const dim_t NR = gemm_ctx->NR;
  const dim_t MR = gemm_ctx->MR;

  const size_t M = gemm_ctx->A->row_size();
  const size_t K = gemm_ctx->A->col_size();
  const size_t N = gemm_ctx->B->col_size();

  dim_t m1, n1, k1, m, n;
  inc_t rsc = 1, csc;

  float *buf = nullptr;
  float *A_tilde = nullptr; // A in G^{MC x KC}
  float *A_tilde_base = nullptr;

  float *B_tilde = nullptr; // B in G^{KC x NC}
  float *B_tilde_base = nullptr;

  float *C_tilde = nullptr; // C in G^{MC x NC}

  alloc_aligned<float>(&buf, MC * KC + KC * NC + MC * NC);

  A_tilde = buf;
  B_tilde = buf + MC * KC;
  C_tilde = buf + MC * KC + KC * NC;

  A_tilde_base = A_tilde;
  B_tilde_base = B_tilde;

  for (int j_c = 0; j_c < N; j_c += NC)
  {
    for (int p_c = 0; p_c < K; p_c += KC / 2)
    {
      k1 = std_ext::min(KC / 2, static_cast<dim_t>(K - p_c));
      n1 = std_ext::min(NC, static_cast<dim_t>(N - j_c));

      memset(B_tilde, 0, KC * NC * sizeof(float));
      gemm_ctx->pack_B(gemm_ctx->B, B_tilde, p_c, j_c, k1, n1, NR);

      // B is now row-major packed into a KC * NC buffer
      // with the specialized format such that each sliver
      // has stride NR

      for (int i_c = 0; i_c < M; i_c += MC / 2)
      {
        m1 = std_ext::min(MC / 2, static_cast<dim_t>(M - i_c));

        memset(A_tilde, 0, MC * KC * sizeof(float));
        gemm_ctx->pack_A(gemm_ctx->A, A_tilde, i_c, p_c, m1, k1, MR);

        // A is now column-major packed into a MC * KC buffer
        // with the specialized format such that each sliver
        // has stride MR

        m1 = std_ext::min(MC, static_cast<dim_t>(M - i_c) * 2);
        k1 = std_ext::min(KC, static_cast<dim_t>(K - p_c) * 2);
        
        for (int j_r = 0; j_r < n1; j_r += NR)
        {
          n = std_ext::min(NR, n1 - j_r);
          
          for (int i_r = 0; i_r < m1; i_r += MR)
          {
            m = csc = std_ext::min(MR, m1 - i_r);
            gemm_ctx->kernel(m, n, k1,
                             gemm_ctx->alpha,
                             A_tilde,
                             B_tilde,
                             gemm_ctx->beta,
                             C_tilde, rsc, csc,
                             NULL,
                             gemm_ctx->cntx);

            gemm_ctx->unpack_C(gemm_ctx->C, C_tilde, i_c + i_r / 2, j_c, m / 2, n);

            A_tilde += MR * k1;
          }
          B_tilde += k1 * NR;

          A_tilde = A_tilde_base;
        }
        B_tilde = B_tilde_base;
      }
    }
  }

  free(buf);
}

template <typename T, typename U>
void gemm_internal(const gemm_context<T, U> *gemm_ctx)
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
  T *A_tilde_base = nullptr;

  T *B_tilde = nullptr; // B in G^{KC x NC}
  T *B_tilde_base = nullptr;
  T *C_tilde = nullptr; // C in G^{MC x NC}

  alloc_aligned<T>(&buf, MC * KC + KC * NC + MC * NC);

  A_tilde = buf;
  B_tilde = buf + MC * KC;
  C_tilde = buf + MC * KC + KC * NC;

  A_tilde_base = A_tilde;
  B_tilde_base = B_tilde;

  dim_t m1, n1, k1, m, n;
  inc_t rsc = 1, csc;

  for (int j_c = 0; j_c < N; j_c += NC)
  {
    for (int p_c = 0; p_c < K; p_c += KC)
    {
      k1 = std_ext::min(KC, static_cast<dim_t>(K - p_c));
      n1 = std_ext::min(NC, static_cast<dim_t>(N - j_c));

      memset(B_tilde, 0, KC * NC * sizeof(float));
      gemm_ctx->pack_B(B, B_tilde, p_c, j_c, k1, n1, NR);

      for (int i_c = 0; i_c < M; i_c += MC)
      {
        m1 = std_ext::min(MC, static_cast<dim_t>(M - i_c));

        memset(A_tilde, 0, MC * KC * sizeof(float));
        gemm_ctx->pack_A(A, A_tilde, i_c, p_c, m1, k1, MR);

        for (int j_r = 0; j_r < n1; j_r += NR)
        {
          n = std_ext::min(NR, n1 - j_r);

          for (int i_r = 0; i_r < m1; i_r += MR)
          {
            m = csc = std_ext::min(MR, m1 - i_r);
            gemm_ctx->kernel(m,
                             n,
                             k1,
                             gemm_ctx->alpha,
                             reinterpret_cast<U *>(A_tilde),
                             reinterpret_cast<U *>(B_tilde),
                             gemm_ctx->beta,
                             reinterpret_cast<U *>(C_tilde), rsc, csc,
                             NULL,
                             gemm_ctx->cntx);
            gemm_ctx->unpack_C(C, C_tilde, i_c, j_c, m, n);

            A_tilde += MR * k1;
          }
          B_tilde += NR * k1;

          A_tilde = A_tilde_base;
        }
        B_tilde = B_tilde_base;
      }
    }
  }

  free(buf);
}

void gemm(ScatterMatrix<std::complex<float>> *A,
          ScatterMatrix<std::complex<float>> *B,
          ScatterMatrix<std::complex<float>> *C,
          const cntx_t *cntx)
{
  // How to fix this?
  float *a = new float(1.);
  float *b = new float(0.);

  const gemm_context_1m<std::complex<float>, float> gemm_ctx = {
      .cntx = cntx,
      .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx),
      .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx),
      .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx),
      .NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx),
      .MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx),
      .A = A,
      .B = B,
      .C = C,
      .alpha = a,
      .beta = b,
      .kernel = bli_sgemm_ukernel,
      .pack_A = pack_A_1m<float>,
      .pack_B = pack_B_1m<float>,
      .unpack_C = unpack_C_1m<float>,
  };

  gemm_internal_complex(&gemm_ctx);

  free(a);
  free(b);
}

void gemm(double *alpha, ScatterMatrix<double> *A,
          ScatterMatrix<double> *B,
          double *beta, ScatterMatrix<double> *C,
          const cntx_t *cntx)
{
  const gemm_context<double, double> gemm_ctx = {
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

  gemm_internal(&gemm_ctx);
}

void gemm(float *alpha, ScatterMatrix<float> *A,
          ScatterMatrix<float> *B,
          float *beta, ScatterMatrix<float> *C,
          const cntx_t *cntx)
{
  const gemm_context<float, float> gemm_ctx = {
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

  gemm_internal(&gemm_ctx);
}