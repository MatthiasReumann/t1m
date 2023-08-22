#include <complex>
#include <memory>
#include "t1m/tensor.hpp"
#include "t1m/index_bundle_finder.hpp"
#include "t1m/scatter_matrix.hpp"
#include "t1m/block_scatter_matrix.hpp"
#include "t1m/gemm.hpp"

namespace t1m
{
  void contract(Tensor<std::complex<float>> A, std::string labelsA,
    Tensor<std::complex<float>> B, std::string labelsB,
    Tensor<std::complex<float>> C, std::string labelsC)
  {
    const cntx_t* cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
    const dim_t KP = 4;

    const auto ilf = std::make_unique<internal::IndexBundleFinder>(labelsA, labelsB, labelsC);

    auto scatterA = std::make_unique<internal::BlockScatterMatrix<std::complex<float>>>(A, ilf->I, ilf->Pa, MR, KP);
    auto scatterB = std::make_unique<internal::BlockScatterMatrix<std::complex<float>>>(B, ilf->Pb, ilf->J, KP, NR);
    auto scatterC = std::make_unique<internal::BlockScatterMatrix<std::complex<float>>>(C, ilf->Ic, ilf->Jc, MR, NR);

    float a = 1.;
    float b = 0.;

    const internal::gemm_context_1m<std::complex<float>, float> gemm_ctx = {
        .cntx = cntx,
        .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx),
        .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx),
        .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx),
        .NR = NR,
        .MR = MR,
        .KP = KP,
        .A = scatterA.get(),
        .B = scatterB.get(),
        .C = scatterC.get(),
        .alpha = &a,
        .beta = &b,
        .kernel = bli_sgemm_ukernel
    };

    internal::gemm_1m(&gemm_ctx);
  }

  void contract(Tensor<std::complex<double>> A, std::string labelsA,
    Tensor<std::complex<double>> B, std::string labelsB,
    Tensor<std::complex<double>> C, std::string labelsC)
  {
    const cntx_t* cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);
    const dim_t KP = 4;

    const auto ilf = std::make_unique<internal::IndexBundleFinder>(labelsA, labelsB, labelsC);

    auto scatterA = std::make_unique<internal::BlockScatterMatrix<std::complex<double>>>(A, ilf->I, ilf->Pa, MR, KP);
    auto scatterB = std::make_unique<internal::BlockScatterMatrix<std::complex<double>>>(B, ilf->Pb, ilf->J, KP, NR);
    auto scatterC = std::make_unique<internal::BlockScatterMatrix<std::complex<double>>>(C, ilf->Ic, ilf->Jc, MR, NR);

    double a = 1.;
    double b = 0.;

    const internal::gemm_context_1m<std::complex<double>, double> gemm_ctx = {
        .cntx = cntx,
        .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx),
        .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx),
        .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx),
        .NR = NR,
        .MR = MR,
        .KP = KP,
        .A = scatterA.get(),
        .B = scatterB.get(),
        .C = scatterC.get(),
        .alpha = &a,
        .beta = &b,
        .kernel = bli_dgemm_ukernel
    };
    internal::gemm_1m(&gemm_ctx);
  }

  void contract(float alpha, Tensor<float> A, std::string labelsA,
    Tensor<float> B, std::string labelsB,
    float beta, Tensor<float> C, std::string labelsC)
  {
    const cntx_t* cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
    const dim_t KP = 4;

    const auto ilf = std::make_unique<internal::IndexBundleFinder>(labelsA, labelsB, labelsC);

    auto scatterA = std::make_unique<internal::BlockScatterMatrix<float>>(A, ilf->I, ilf->Pa, MR, KP);
    auto scatterB = std::make_unique<internal::BlockScatterMatrix<float>>(B, ilf->Pb, ilf->J, KP, NR);
    auto scatterC = std::make_unique<internal::BlockScatterMatrix<float>>(C, ilf->Ic, ilf->Jc, MR, NR);

    const internal::gemm_context<float> gemm_ctx = {
          .cntx = cntx,
          .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx),
          .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx),
          .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx),
          .NR = NR,
          .MR = MR,
          .KP = KP,
          .A = scatterA.get(),
          .B = scatterB.get(),
          .C = scatterC.get(),
          .alpha = &alpha,
          .beta = &beta,
          .kernel = bli_sgemm_ukernel
    };

    internal::gemm(&gemm_ctx);
  }

  void contract(double alpha, Tensor<double> A, std::string labelsA,
    Tensor<double> B, std::string labelsB,
    double beta, Tensor<double> C, std::string labelsC)
  {
    const cntx_t* cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);
    const dim_t KP = 4;

    const auto ilf = std::make_unique<internal::IndexBundleFinder>(labelsA, labelsB, labelsC);

    auto scatterA = std::make_unique<internal::BlockScatterMatrix<double>>(A, ilf->I, ilf->Pa, MR, KP);
    auto scatterB = std::make_unique<internal::BlockScatterMatrix<double>>(B, ilf->Pb, ilf->J, KP, NR);
    auto scatterC = std::make_unique<internal::BlockScatterMatrix<double>>(C, ilf->Ic, ilf->Jc, MR, NR);

    const internal::gemm_context<double> gemm_ctx = {
          .cntx = cntx,
          .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx),
          .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx),
          .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx),
          .NR = NR,
          .MR = MR,
          .KP = KP,
          .A = scatterA.get(),
          .B = scatterB.get(),
          .C = scatterC.get(),
          .alpha = &alpha,
          .beta = &beta,
          .kernel = bli_dgemm_ukernel
    };

    internal::gemm(&gemm_ctx);
  }
};
