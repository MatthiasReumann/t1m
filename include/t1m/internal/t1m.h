#pragma once

#include <complex>
#include <cstdlib>
#include <blis/blis.h>
#include "t1m/internal/packing.h"
#include "t1m/internal/scatter.h"
#include "t1m/internal/utils.h"
#include "t1m/internal/tensor.h"

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
void contract(const T alpha, const tensor<T, ndim_a>& a,
              const std::string& labels_a, const tensor<T, ndim_b>& b,
              const std::string& labels_b, const T beta, tensor<T, ndim_c>& c,
              const std::string& labels_c) {
  blis_kernel<T> kernel{};
  blis_blocksizes<T> blocksizes{};
  utils::contraction indices({labels_a, labels_b, labels_c});
  scatter::block_layout layout_a(a.dimensions, a.strides(), indices.AI,
                                 indices.AP, blocksizes.MR, blocksizes.KP);
  scatter::block_layout layout_b(b.dimensions, b.strides(), indices.BP,
                                 indices.BJ, blocksizes.KP, blocksizes.NR);
  scatter::block_layout layout_c(c.dimensions, c.strides(), indices.CI,
                                 indices.CJ, blocksizes.MR, blocksizes.NR);
  scatter::matrix_view matr_a{layout_a.rs,  layout_a.cs, layout_a.br,
                              layout_a.rbs, layout_a.bc, layout_a.cbs};
  scatter::matrix_view matr_b{layout_b.rs,  layout_b.cs, layout_b.br,
                              layout_b.rbs, layout_b.bc, layout_b.cbs};
  scatter::matrix_view matr_c{layout_c.rs,  layout_c.cs, layout_c.br,
                              layout_c.rbs, layout_c.bc, layout_c.cbs};

  std::allocator<T> alloc;
  T* workspace_b = alloc.allocate(blocksizes.KC * blocksizes.NC);

  inc_t rsc = 1, csc;
  const std::size_t M = matr_a.nrows();
  const std::size_t K = matr_a.ncols();
  const std::size_t N = matr_b.ncols();
  for (std::size_t j_c = 0; j_c < N; j_c += blocksizes.NC) {
    std::size_t nc_n =
        std::min<dim_t>(blocksizes.NC, static_cast<dim_t>(N - j_c));
    for (std::size_t p_c = 0; p_c < K; p_c += blocksizes.KC) {
      std::size_t k =
          std::min<dim_t>(blocksizes.KC, static_cast<dim_t>(K - p_c));
      scatter::matrix_view block_b = matr_b.subview(p_c, j_c, k, nc_n);

      packing::pack_block_row_major<T>(block_b, a.data, workspace_b);
    }
  }

  alloc.deallocate(workspace_b, blocksizes.KC * blocksizes.NC);
}
};  // namespace t1m
