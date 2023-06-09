#pragma once

/*
Please give this project a better name.
*/
#include <complex>
#include "index_bundle_finder.hpp"
#include "scatter_matrix.hpp"
#include "packing.hpp"
#include "gemm.hpp"

void contract(Tensor<std::complex<float> > A, std::string labelsA,
              Tensor<std::complex<float> > B, std::string labelsB,
              Tensor<std::complex<float> > C, std::string labelsC)
{
  const cntx_t *cntx = bli_gks_query_cntx();

  auto ilf = new IndexBundleFinder(labelsA, labelsB, labelsC);
  auto scatterA = new ScatterMatrix<std::complex<float> >(A, ilf->I, ilf->Pa);
  auto scatterB = new ScatterMatrix<std::complex<float> >(B, ilf->Pb, ilf->J);
  auto scatterC = new ScatterMatrix<std::complex<float> >(C, ilf->Ic, ilf->Jc);

  // std::complex<float> *a = new std::complex<float> (alpha);
  // std::complex<float> *b = new std::complex<float> (beta);

  gemm(scatterA, scatterB, scatterC, cntx);

  // free(a);
  // free(b);
}

void contract(float alpha, Tensor<float> A, std::string labelsA,
              Tensor<float> B, std::string labelsB,
              float beta, Tensor<float> C, std::string labelsC)
{
  const cntx_t *cntx = bli_gks_query_cntx();

  auto ilf = new IndexBundleFinder(labelsA, labelsB, labelsC);
  auto scatterA = new ScatterMatrix<float>(A, ilf->I, ilf->Pa);
  auto scatterB = new ScatterMatrix<float>(B, ilf->Pb, ilf->J);
  auto scatterC = new ScatterMatrix<float>(C, ilf->Ic, ilf->Jc);

  float *a = new float(alpha);
  float *b = new float(beta);

  gemm(a, scatterA, scatterB, b, scatterC, cntx);

  free(a);
  free(b);
}

void contract(double alpha, Tensor<double> A, std::string labelsA,
              Tensor<double> B, std::string labelsB,
              double beta, Tensor<double> C, std::string labelsC)
{
  const cntx_t *cntx = bli_gks_query_cntx();

  auto ilf = new IndexBundleFinder(labelsA, labelsB, labelsC);
  auto scatterA = new ScatterMatrix<double>(A, ilf->I, ilf->Pa);
  auto scatterB = new ScatterMatrix<double>(B, ilf->Pb, ilf->J);
  auto scatterC = new ScatterMatrix<double>(C, ilf->Ic, ilf->Jc);

  double *a = new double(alpha);
  double *b = new double(beta);

  gemm(a, scatterA, scatterB, b, scatterC, cntx);

  free(a);
  free(b);
}
