#include <complex>
#include "tfctc/tensor.hpp"
#include "tfctc/index_bundle_finder.hpp"
#include "tfctc/scatter_matrix.hpp"
#include "tfctc/block_scatter_matrix.hpp"
#include "tfctc/gemm.hpp"

namespace tfctc
{
  void contract(Tensor<std::complex<float>> A, std::string labelsA,
    Tensor<std::complex<float>> B, std::string labelsB,
    Tensor<std::complex<float>> C, std::string labelsC)
  {
    const cntx_t* cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
    const dim_t KP = 4;

    const auto ilf = new internal::IndexBundleFinder(labelsA, labelsB, labelsC);

    auto scatterA = new internal::BlockScatterMatrix(A, ilf->I, ilf->Pa, MR, KP);
    auto scatterB = new internal::BlockScatterMatrix(B, ilf->Pb, ilf->J, KP, NR);
    auto scatterC = new internal::BlockScatterMatrix(C, ilf->Ic, ilf->Jc, MR, NR);

    float* a = new float(1.);
    float* b = new float(0.);

    const internal::gemm_context_1m<std::complex<float>, float> gemm_ctx = {
        .cntx = cntx,
        .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx),
        .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx),
        .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx),
        .NR = NR,
        .MR = MR,
        .KP = KP,
        .A = scatterA,
        .B = scatterB,
        .C = scatterC,
        .alpha = a,
        .beta = b,
        .kernel = bli_sgemm_ukernel
    };

    internal::gemm_1m(&gemm_ctx);

    delete a;
    delete b;
  }

  void contract(Tensor<std::complex<double>> A, std::string labelsA,
    Tensor<std::complex<double>> B, std::string labelsB,
    Tensor<std::complex<double>> C, std::string labelsC)
  {
    const cntx_t* cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);
    const dim_t KP = 4;

    const auto ilf = new internal::IndexBundleFinder(labelsA, labelsB, labelsC);

    auto scatterA = new internal::BlockScatterMatrix(A, ilf->I, ilf->Pa, MR, KP);
    auto scatterB = new internal::BlockScatterMatrix(B, ilf->Pb, ilf->J, KP, MR);
    auto scatterC = new internal::BlockScatterMatrix(C, ilf->Ic, ilf->Jc, MR, NR);

    double* a = new double(1.);
    double* b = new double(0.);

    const internal::gemm_context_1m<std::complex<double>, double> gemm_ctx = {
        .cntx = cntx,
        .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx),
        .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx),
        .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx),
        .NR = NR,
        .MR = MR,
        .KP = KP,
        .A = scatterA,
        .B = scatterB,
        .C = scatterC,
        .alpha = a,
        .beta = b,
        .kernel = bli_dgemm_ukernel
    };
    internal::gemm_1m(&gemm_ctx);
    delete a;
    delete b;
  }

  void contract(float alpha, Tensor<float> A, std::string labelsA,
    Tensor<float> B, std::string labelsB,
    float beta, Tensor<float> C, std::string labelsC)
  {
    const cntx_t* cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx);
    const dim_t KP = 4;

    const auto ilf = new internal::IndexBundleFinder(labelsA, labelsB, labelsC);

    auto scatterA = new internal::BlockScatterMatrix<float>(A, ilf->I, ilf->Pa, MR, KP);
    auto scatterB = new internal::BlockScatterMatrix<float>(B, ilf->Pb, ilf->J, KP, NR);
    auto scatterC = new internal::BlockScatterMatrix<float>(C, ilf->Ic, ilf->Jc, MR, NR);

    float* a = new float(alpha);
    float* b = new float(beta);

    const internal::gemm_context<float> gemm_ctx = {
          .cntx = cntx,
          .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx),
          .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx),
          .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx),
          .NR = NR,
          .MR = MR,
          .KP = KP,
          .A = scatterA,
          .B = scatterB,
          .C = scatterC,
          .alpha = a,
          .beta = b,
          .kernel = bli_sgemm_ukernel
    };

    internal::gemm(&gemm_ctx);

    delete a;
    delete b;
  }

  void contract(double alpha, Tensor<double> A, std::string labelsA,
    Tensor<double> B, std::string labelsB,
    double beta, Tensor<double> C, std::string labelsC)
  {
    const cntx_t* cntx = bli_gks_query_cntx();

    const dim_t NR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx);
    const dim_t MR = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx);
    const dim_t KP = 4;

    const auto ilf = new internal::IndexBundleFinder(labelsA, labelsB, labelsC);

    auto scatterA = new internal::BlockScatterMatrix<double>(A, ilf->I, ilf->Pa, MR, KP);
    auto scatterB = new internal::BlockScatterMatrix<double>(B, ilf->Pb, ilf->J, KP, NR);
    auto scatterC = new internal::BlockScatterMatrix<double>(C, ilf->Ic, ilf->Jc, MR, NR);

    double* a = new double(alpha);
    double* b = new double(beta);

    const internal::gemm_context<double> gemm_ctx = {
          .cntx = cntx,
          .NC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx),
          .KC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx),
          .MC = bli_cntx_get_l3_sup_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx),
          .NR = NR,
          .MR = MR,
          .KP = KP,
          .A = scatterA,
          .B = scatterB,
          .C = scatterC,
          .alpha = a,
          .beta = b,
          .kernel = bli_dgemm_ukernel
    };

    internal::gemm(&gemm_ctx);

    delete a;
    delete b;
  }
};
