#pragma once

/*
Please give this project a better name.
*/
#include "marray.hpp"
#include "index_bundle_finder.h"
#include "scatter_matrix.h"
#include "macrokernel.h"
#include "definitions.h"
#include "utils.h"

template <int mc, int nc, int kc>
void gemm(ScatterMatrix *A, ScatterMatrix *B, ScatterMatrix *C)
{
  float *A_ = A->data();
  float *B_ = B->data();
  float *C_ = C->data();

  size_t m = A->row_size();
  size_t k = A->col_size();
  size_t n = B->col_size();

  float *A_tilde = nullptr; // A in R^{mc x kc}
  float *B_tilde = nullptr; // B in G^{kc x nc}
  float *C_tilde = nullptr; // C in G^{mc x nc}

  alloc_aligned<float>(&A_tilde, mc * kc);
  alloc_aligned<float>(&B_tilde, kc * nc);
  alloc_aligned<float>(&C_tilde, mc * nc);

  for (int j_c = 0; j_c < n - (n % nc); j_c += nc)
  {
    for (int p_c = 0; p_c < k - (k % kc); p_c += kc)
    {
      B->pack_to_submatrix<float, kc, nc>(B_tilde,  p_c, j_c);
      
      for (int i_c = 0; i_c < m - (m % mc); i_c += mc)
      {
        A->pack_to_submatrix<float, mc, kc>(A_tilde, i_c, p_c);

        macrokernel_simple<mc, nc, kc>(A_tilde, B_tilde, C_tilde);

        C->add_from_submatrix<float, mc, nc>(C_tilde, i_c, j_c);
      }
    }
  }

  for (int i = m - (m % mc); i < m; i++)
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
  }

  free(A_tilde);
  free(B_tilde);
  free(C_tilde);
}

void contract(Tensor<float> A, std::string labelsA,
              Tensor<float> B, std::string labelsB,
              Tensor<float> C, std::string labelsC)
{
  auto indexLabelFinder = new IndexBundleFinder(labelsA, labelsB, labelsC);
  auto scatterA = new ScatterMatrix(A, indexLabelFinder->I, indexLabelFinder->Pa);
  auto scatterB = new ScatterMatrix(B, indexLabelFinder->Pb, indexLabelFinder->J);
  auto scatterC = new ScatterMatrix(C, indexLabelFinder->Ic, indexLabelFinder->Jc);

  gemm<2, 2, 2>(scatterA, scatterB, scatterC);
}
