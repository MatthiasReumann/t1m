#pragma once

/*
Please give this project a better name.
*/
#include "index_bundle_finder.hpp"
#include "scatter_matrix.hpp"
#include "macrokernel.hpp"
#include "std_ext.hpp"
#include "utils.hpp"
#include "marray.hpp"
#include "blis.h"

void gemm(float *alpha, ScatterMatrix *A, ScatterMatrix *B, float *beta, ScatterMatrix *C)
{
  const cntx_t *cntx = bli_gks_query_cntx();

  dim_t NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx);
  dim_t KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx);
  dim_t MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx);
  dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
  dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);

  /*std::cout << "NC: " << NC << '\n'
            << "KC: " << KC << '\n'
            << "MC: " << MC << '\n'
            << "NR: " << NR << '\n'
            << "MR: " << MR << '\n';*/

  size_t M = A->row_size();
  size_t K = A->col_size();
  size_t N = B->col_size();

  /*std::cout << "M: " << M << '\n'
            << "K: " << K << '\n'
            << "N: " << N << '\n';*/

  float *A_tilde = nullptr; // A in G^{MC x KC}
  float *B_tilde = nullptr; // B in G^{KC x NC}
  float *C_tilde = nullptr; // C in G^{MC x NC}

  alloc_aligned<float>(&A_tilde, MC * KC);
  alloc_aligned<float>(&B_tilde, KC * NC);
  alloc_aligned<float>(&C_tilde, MC * NC);

  dim_t m1, n1, k1;
  inc_t rsc = 1, csc;

  auxinfo_t *data = NULL;

  // std::cout << "alpha: " << *alpha << " beta: " << *beta << std::endl;

  for (int j_c = 0; j_c < N; j_c += NC)
  {
    for (int p_c = 0; p_c < K; p_c += KC)
    {
      k1 = std_ext::min(KC, static_cast<dim_t>(K - p_c));
      n1 = std_ext::min(NC, static_cast<dim_t>(N - j_c));
      B->pack_to_cont_buffer_row<float>(B_tilde, p_c, j_c, k1, n1, NR);

      // std::cout << "B~ (KC x NC): ";
      // print_mat_row(B_tilde, k1, n1);

      for (int i_c = 0; i_c < M; i_c += MC)
      {
        m1 = std_ext::min(MC, static_cast<dim_t>(M - i_c));
        A->pack_to_cont_buffer_col<float>(A_tilde, i_c, p_c, m1, k1, MR);

        // std::cout << "A~ (MC x KC): ";
        // print_mat(A_tilde, m1, k1);

        for (int j_r = 0; j_r < n1; j_r += NR)
        {
          dim_t n = std_ext::min(NR, n1 - j_r);

          for (int i_r = 0; i_r < m1; i_r += MR)
          {
            dim_t m = std_ext::min(MR, m1 - i_r);
            csc = m;

            bli_sgemm_ukernel(m,
                              n,
                              k1,
                              alpha,
                              A_tilde + i_r * k1,
                              B_tilde + j_r * k1,
                              beta,
                              C_tilde, rsc, csc,
                              data,
                              cntx);
  
            C->unpack_from_buffer(C_tilde, i_c, j_c, m, n);
          }
        }
      }
    }
  }

  free(A_tilde);
  free(B_tilde);
  free(C_tilde);
}

void contract(float alpha, Tensor<float> A, std::string labelsA,
              Tensor<float> B, std::string labelsB,
              float beta, Tensor<float> C, std::string labelsC)
{
  auto ilf = new IndexBundleFinder(labelsA, labelsB, labelsC);
  auto scatterA = new ScatterMatrix(A, ilf->I, ilf->Pa);
  auto scatterB = new ScatterMatrix(B, ilf->Pb, ilf->J);
  auto scatterC = new ScatterMatrix(C, ilf->Ic, ilf->Jc);

  float *a = new float(alpha);
  float *b = new float(beta);

  gemm(a, scatterA, scatterB, b, scatterC);

  free(a);
  free(b);
}
