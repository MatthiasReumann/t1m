#pragma once

/*
Please give this project a better name.
*/
#include "index_bundle_finder.hpp"
#include "scatter_matrix.hpp"
#include "macrokernel.hpp"
#include "definitions.hpp"
#include "std_ext.hpp"
#include "utils.hpp"
#include "marray.hpp"
#include "blis.h"

template <int mc, int nc, int kc>
void gemm_macrokernel(ScatterMatrix *A, ScatterMatrix *B, ScatterMatrix *C)
{
  float *A_ = A->data();
  float *B_ = B->data();
  float *C_ = C->data();

  size_t m = A->row_size();
  size_t k = A->col_size();
  size_t n = B->col_size();

  float *A_tilde = nullptr; // A in G^{mc x kc}
  float *B_tilde = nullptr; // B in G^{kc x nc}
  float *C_tilde = nullptr; // C in G^{mc x nc}

  alloc_aligned<float>(&A_tilde, mc * kc);
  alloc_aligned<float>(&B_tilde, kc * nc);
  alloc_aligned<float>(&C_tilde, mc * nc);

  for (int j_c = 0; j_c < n - (n % nc); j_c += nc)
  {
    for (int p_c = 0; p_c < k - (k % kc); p_c += kc)
    {
      B->pack_to_cont_buffer_row<float, kc, nc>(B_tilde, p_c, j_c);

      for (int i_c = 0; i_c < m - (m % mc); i_c += mc)
      {
        A->pack_to_cont_buffer_col<float, mc, kc>(A_tilde, i_c, p_c);

        macrokernel_simple<mc, nc, kc>(A_tilde, B_tilde, C_tilde);

        C->add_from_submatrix<float, mc, nc>(C_tilde, i_c, j_c);
      }
    }
  }

  /*for (int i = m - (m % mc); i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
      float c_ij = 0.;
      for (int p = 0; p < k; p++)
      {
        const float a = A_[A->location(i, p)];
        const float b = B_[B->location(p, j)];
        c_ij += a * b;
      }
      C_[C->location(i, j)] = c_ij;
    }
  }

  for (int i = 0; i < m; i++)
  {
    for (int j = n - (n % nc); j < n; j++)
    {
      float c_ij = 0.;
      for (int p = 0; p < k; p++)
      {
        const float a = A_[A->location(i, p)];
        const float b = B_[B->location(p, j)];
        c_ij += a * b;
      }
      C_[C->location(i, j)] = c_ij;
    }
  }*/

  free(A_tilde);
  free(B_tilde);
  free(C_tilde);
}

void gemm(ScatterMatrix *A, ScatterMatrix *B, ScatterMatrix *C)
{
  const cntx_t *cntx = bli_gks_query_cntx();

  const dim_t NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx);
  const dim_t KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx);
  const dim_t MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx);
  const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);
  const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);

  std::cout << "NC: " << NC << '\n'
            << "KC: " << KC << '\n'
            << "MC: " << MC << '\n'
            << "NR: " << NR << '\n'
            << "MR: " << MR << '\n';

  /*
  a1[0] = 2.;
  a1[0 + MR] = 2.7;
  a1[1] = 1.07;
  a1[1 + MR] = 1.;

  b1[0] = 1.;
  b1[1] = 2.;
  b1[0 + NR] = 1.;
  b1[1 + NR] = 2.;

  bli_dgemm_ukernel(m, n, k, alpha, a1, b1, beta, c11, rsc, csc, data, cntx);
  for (int i = 0; i < MR * k; i++) {
    std::cout << a1[i] << " ";
  }
  std::cout << std::endl;

  std::cout << std::endl;

  for (int i = 0; i < NR * k; i++) {
    std::cout << b1[i] << " ";
  }
  std::cout << std::endl;


  print_mat(c11, m, n); */

  size_t m = A->row_size();
  size_t k = A->col_size();
  size_t n = B->col_size();

  std::cout << "m: " << m << '\n'
            << "k: " << k << '\n'
            << "n: " << n << '\n';

  float *A_tilde = nullptr; // A in G^{MC x KC}
  float *B_tilde = nullptr; // B in G^{KC x NC}
  float *C_tilde = nullptr; // C in G^{MC x NC}

  alloc_aligned<float>(&A_tilde, MC * KC);
  alloc_aligned<float>(&B_tilde, KC * NC);
  alloc_aligned<float>(&C_tilde, MC * NC);

  dim_t m1, n1, k1;
  inc_t rsc = 1, csc;
  // inc_t rsc = 1, csc = m;

  double *a1;
  double *b1;
  double *c11;
  double *alpha, *beta; // TODO: As parameter

  auxinfo_t *data = NULL;

  alloc_aligned(&alpha, 1);
  alloc_aligned(&beta, 1);

  alloc_aligned(&a1, MR * KC);
  alloc_aligned(&b1, KC * NR);
  alloc_aligned(&c11, MR * NR);

  *alpha = 1.;
  *beta = 0.;

  for (int j_c = 0; j_c < n; j_c += NC)
  {
    for (int p_c = 0; p_c < k; p_c += KC)
    {
      k1 = std_ext::min(KC, static_cast<dim_t>(k - p_c));
      n1 = std_ext::min(NC, static_cast<dim_t>(n - j_c));
      B->pack_to_cont_buffer_row<float>(B_tilde, p_c, j_c, k1, n1);

      for (int i_c = 0; i_c < m; i_c += MC)
      {
        m1 = std_ext::min(MC, static_cast<dim_t>(m - i_c));
        
        A->pack_to_cont_buffer_col<float>(A_tilde, i_c, p_c, m1, k1);

        for (int j_r = 0; j_r < NC; j_r += NR)
        {
          for (int i_r = 0; i_r < MC; i_r += MR)
          {
            // bli_dgemm_skernel(m, n, k, alpha, a1, b1, beta, c11, rsc, csc, data, cntx);
          }
        }
      }
    }
  }

  free(A_tilde);
  free(B_tilde);
  free(C_tilde);
  free(a1);
  free(b1);
  free(c11);
}

void contract(Tensor<float> A, std::string labelsA,
              Tensor<float> B, std::string labelsB,
              Tensor<float> C, std::string labelsC)
{
  auto indexLabelFinder = new IndexBundleFinder(labelsA, labelsB, labelsC);
  auto scatterA = new ScatterMatrix(A, indexLabelFinder->I, indexLabelFinder->Pa);
  auto scatterB = new ScatterMatrix(B, indexLabelFinder->Pb, indexLabelFinder->J);
  auto scatterC = new ScatterMatrix(C, indexLabelFinder->Ic, indexLabelFinder->Jc);

  gemm(scatterA, scatterB, scatterC);
}
