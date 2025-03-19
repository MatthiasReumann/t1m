#pragma once

#include <complex>
#include "blis.h"
#include "t1m/internal/packing.h"
#include "t1m/internal/scatter.h"
#include "t1m/internal/utils.h"
#include "t1m/tensor.h"

namespace t1m {

template <typename T>
struct blis_blocksizes {
  const std::size_t MR;
  const std::size_t NR;
  const std::size_t KP;
  const std::size_t MC;
  const std::size_t KC;
  const std::size_t NC;

  blis_blocksizes(const cntx_t* cntx = bli_gks_query_cntx())
      requires(std::same_as<T, float> || std::same_as<T, std::complex<float>>)
      : MR(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx)),
        NR(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx)),
        KP(4),
        MC(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx)),
        KC(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx)),
        NC(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx)) {}

  blis_blocksizes(const cntx_t* cntx = bli_gks_query_cntx())
      requires(std::same_as<T, double> || std::same_as<T, std::complex<double>>)
      : MR(bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MR, cntx)),
        NR(bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NR, cntx)),
        KP(4),
        MC(bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_MC, cntx)),
        KC(bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_KC, cntx)),
        NC(bli_cntx_get_blksz_def_dt(BLIS_DOUBLE, BLIS_NC, cntx)) {}
};

template <typename T>
struct blis_kernel {

  template <typename... Us>
  auto operator()(Us... args)
      requires(std::same_as<T, float> || std::same_as<T, std::complex<float>>) {
    return bli_sgemm_ukernel(args...);
  }

  template <typename... Us>
  auto operator()(Us... args) requires(std::same_as<T, double> ||
                                       std::same_as<T, std::complex<double>>) {
    return bli_dgemm_ukernel(args...);
  }
};

template <typename T, std::size_t ndim_a, std::size_t ndim_b,
          std::size_t ndim_c>
void contract(const T alpha, const tensor<T, ndim_a>& a, std::string labels_a,
              const tensor<T, ndim_b>& b, std::string labels_b, const T beta,
              tensor<T, ndim_c>& c, std::string labels_c) {
  blis_kernel<T> kernel{};
  blis_blocksizes<T> blocksizes{};

  utils::contraction indices({labels_a, labels_b, labels_c});

  // scatter::matrix_view<T, ndim_a> matr_a(a, indices.AI, blocksizes.MR,
  //                                        indices.AP, blocksizes.KP);
  // scatter::matrix_view<T, ndim_b> matr_b(b, indices.BP, blocksizes.KP,
  //                                        indices.BJ, blocksizes.NR);
  // scatter::matrix_view<T, ndim_c> matr_c(c, indices.CI, blocksizes.MR,
  //                                        indices.CJ, blocksizes.NR);

  // packing::lhs<T> wrksp_a(blocksizes.MC * blocksizes.KC);

  // inc_t rsc = 1, csc;
  // const std::size_t M = matr_a.nrows();
  // const std::size_t K = matr_a.ncols();
  // const std::size_t N = matr_b.ncols();
  // for (std::size_t j_c = 0; j_c < N; j_c += blocksizes.NC) {
  //   // std::size_t nc_n = std::min<dim_t>(NC, static_cast<dim_t>(N - j_c));

  //   for (std::size_t p_c = 0; p_c < K; p_c += blocksizes.KC) {
  //     // k = std::min<dim_t>(KC, static_cast<dim_t>(K - p_c));
  //     // pack_b(B, b_packed, p_c, j_c, k, nc_n, NR, KP);

  //     for (std::size_t i_c = 0; i_c < M; i_c += blocksizes.MC) {
  //       // mc_m = std::min<dim_t>(MC, static_cast<dim_t>(M - i_c));

  //       // wrksp_a.pack(matr_a, nullptr, blocksizes.MC, blocksizes.KC);
  //       // pack_a(A, a_packed, i_c, p_c, mc_m, k, MR, KP);

  //       // for (std::size_t j_r = 0; j_r < nc_n; j_r += blocksizes.NR) {
  //       //   // n = std::min<dim_t>(NR, static_cast<dim_t>(nc_n - j_r));
  //       //   // off_j = j_c + j_r;
  //       //   // csc = C->col_stride_in_block(off_j / NR);

  //       //   for (std::size_t i_r = 0; i_r < mc_m; i_r += blocksizes.MR) {
  //       //     // m = std::min<dim_t>(MR, static_cast<dim_t>(mc_m - i_r));
  //       //     // off_i = i_c + i_r;
  //       //     // rsc = C->row_stride_in_block(off_i / MR);

  //       //     // if (rsc > 0 && csc > 0) {
  //       //     //   ctx->kernel(m, n, k, ctx->alpha, a_packed, b_packed, ctx->beta,
  //       //     //               C->pointer_at_loc(off_i, off_j), rsc, csc, nullptr,
  //       //     //               ctx->cntx);
  //       //     // } else {
  //       //     //   ctx->kernel(m, n, k, ctx->alpha, a_packed, b_packed, ctx->beta,
  //       //     //               c_result, 1, m, nullptr, ctx->cntx);

  //       //     //   unpack_c_scat(C, c_result, off_i, off_j, m, n);
  //       //     // }

  //       //     // a_packed += MR * k;
  //       //   }
  //       //   // b_packed += k * NR;

  //       //   // a_packed = a_packed_base;
  //       // }
  //       // // b_packed = b_packed_base;
  //     }
  //   }
  // }
}
};  // namespace t1m
