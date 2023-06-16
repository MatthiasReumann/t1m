#pragma once

/*
Please give this project a better name.
*/
#include <complex>
#include "index_bundle_finder.hpp"
#include "scatter_matrix.hpp"
#include "block_scatter_matrix.hpp"
#include "packing.hpp"
#include "gemm.hpp"

#include "utils.hpp"

namespace tfctc
{
  void contract(Tensor<std::complex<float>> A, std::string labelsA,
                Tensor<std::complex<float>> B, std::string labelsB,
                Tensor<std::complex<float>> C, std::string labelsC)
  {
    const cntx_t *cntx = bli_gks_query_cntx();

    // TODO: CORRECT BLOCKINGS SIZE FOR 1M?
    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);

    const auto ilf = new internal::IndexBundleFinder(labelsA, labelsB, labelsC);
    
    auto scatterA = new internal::ScatterMatrix<std::complex<float>>(A, ilf->I, ilf->Pa);
    auto scatterB = new internal::ScatterMatrix<std::complex<float>>(B, ilf->Pb, ilf->J);
    auto scatterC = new internal::BlockScatterMatrix<std::complex<float>>(C, ilf->Ic, ilf->Jc, MR, NR);

    internal::gemm(scatterA, scatterB, scatterC, cntx);
  }

  void contract(Tensor<std::complex<double>> A, std::string labelsA,
                Tensor<std::complex<double>> B, std::string labelsB,
                Tensor<std::complex<double>> C, std::string labelsC)
  {
    const cntx_t *cntx = bli_gks_query_cntx();

    // TODO: CORRECT BLOCKING SIZES FOR 1M?
    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);

    const auto ilf = new internal::IndexBundleFinder(labelsA, labelsB, labelsC);

    auto scatterA = new internal::ScatterMatrix<std::complex<double>>(A, ilf->I, ilf->Pa);
    auto scatterB = new internal::ScatterMatrix<std::complex<double>>(B, ilf->Pb, ilf->J);
    auto scatterC = new internal::BlockScatterMatrix<std::complex<double>>(C, ilf->Ic, ilf->Jc, MR, NR);

    internal::gemm(scatterA, scatterB, scatterC, cntx);
  }

  void contract(float alpha, Tensor<float> A, std::string labelsA,
                Tensor<float> B, std::string labelsB,
                float beta, Tensor<float> C, std::string labelsC)
  {
    const cntx_t *cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);

    const auto ilf = new internal::IndexBundleFinder(labelsA, labelsB, labelsC);
    
    auto scatterA = new internal::ScatterMatrix<float>(A, ilf->I, ilf->Pa);
    auto scatterB = new internal::ScatterMatrix<float>(B, ilf->Pb, ilf->J);
    auto scatterC = new internal::BlockScatterMatrix<float>(C, ilf->Ic, ilf->Jc, MR, NR);

    float *a = new float(alpha);
    float *b = new float(beta);

    internal::gemm(a, scatterA, scatterB, b, scatterC, cntx);

    free(a);
    free(b);
  }

  void contract(double alpha, Tensor<double> A, std::string labelsA,
                Tensor<double> B, std::string labelsB,
                double beta, Tensor<double> C, std::string labelsC)
  {
    const cntx_t *cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);

    const auto ilf = new internal::IndexBundleFinder(labelsA, labelsB, labelsC);
    
    auto scatterA = new internal::ScatterMatrix<double>(A, ilf->I, ilf->Pa);
    auto scatterB = new internal::ScatterMatrix<double>(B, ilf->Pb, ilf->J);
    auto scatterC = new internal::BlockScatterMatrix<double>(C, ilf->Ic, ilf->Jc, MR, NR);

    double *a = new double(alpha);
    double *b = new double(beta);

    internal::gemm(a, scatterA, scatterB, b, scatterC, cntx);

    free(a);
    free(b);
  }
};
