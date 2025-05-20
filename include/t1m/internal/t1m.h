#pragma once

#include <blis/blis.h>
#include <complex>
#include <cstdlib>
#include <memory>
#include <print>
#include "t1m/internal/packing.h"
#include "t1m/internal/scatter.h"
#include "t1m/internal/tensor.h"
#include "t1m/internal/utils.h"

namespace t1m {

template <typename T>
struct blis_blocksizes {
  std::size_t MR;
  std::size_t NR;
  std::size_t KP;
  std::size_t MC;
  std::size_t KC;
  std::size_t NC;

  blis_blocksizes(const cntx_t* cntx)
      requires(std::same_as<T, float> || std::same_as<T, std::complex<float>>)
      : MR(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MR, cntx)),
        NR(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NR, cntx)),
        KP(4),
        MC(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_MC, cntx)),
        KC(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_KC, cntx)),
        NC(bli_cntx_get_blksz_def_dt(BLIS_FLOAT, BLIS_NC, cntx)) {}

  blis_blocksizes(const cntx_t* cntx)
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
  constexpr auto operator()(Us... args) const
      requires(std::same_as<T, float> || std::same_as<T, std::complex<float>>) {
    return bli_sgemm_ukernel(args...);
  }

  template <typename... Us>
  constexpr auto operator()(Us... args) const
      requires(std::same_as<T, double> ||
               std::same_as<T, std::complex<double>>) {
    return bli_dgemm_ukernel(args...);
  }
};

template <class T, std::size_t ndim_a, std::size_t ndim_b, std::size_t ndim_c>
void contract(const T alpha, const tensor<T, ndim_a>& a,
              const std::string& labels_a, const tensor<T, ndim_b>& b,
              const std::string& labels_b, const T beta, tensor<T, ndim_c>& c,
              const std::string& labels_c) {
  std::allocator<T> alloc{};

  constexpr blis_kernel<T> kernel{};

  const cntx_t* cntx = bli_gks_query_cntx();
  const blis_blocksizes<T> blocksizes(cntx);
  const contraction indices({labels_a, labels_b, labels_c});

  const block_layout layout_a(a.dimensions, a.strides(), indices.AI, indices.AP,
                              blocksizes.MR, blocksizes.KP);
  const block_layout layout_b(b.dimensions, b.strides(), indices.BP, indices.BJ,
                              blocksizes.KP, blocksizes.NR);
  const block_layout layout_c(c.dimensions, c.strides(), indices.CI, indices.CJ,
                              blocksizes.MR, blocksizes.NR);

  const matrix_view matr_a = matrix_view::from_layout(layout_a);
  const matrix_view matr_b = matrix_view::from_layout(layout_b);
  const matrix_view matr_c = matrix_view::from_layout(layout_c);

  T* workspace_a = alloc.allocate(blocksizes.MC * blocksizes.KC);
  T* workspace_b = alloc.allocate(blocksizes.KC * blocksizes.NC);
  T* workspace_c = alloc.allocate(blocksizes.MR * blocksizes.NR);

  auxinfo_t data;

  const std::size_t M = matr_a.nrows();
  const std::size_t K = matr_a.ncols();
  const std::size_t N = matr_b.ncols();

  for (std::size_t j_c = 0; j_c < N; j_c += blocksizes.NC) {
    std::size_t nc_n = std::min(blocksizes.NC, N - j_c);

    for (std::size_t p_c = 0; p_c < K; p_c += blocksizes.KC) {
      std::size_t k = std::min(blocksizes.KC, K - p_c);

      matrix_view block_b = matr_b.subview(p_c, j_c, k, nc_n);
      pack_block_row_major(block_b, b.data, workspace_b);

      for (size_t i_c = 0; i_c < M; i_c += blocksizes.MC) {
        std::size_t mc_m = std::min(blocksizes.MC, M - i_c);

        matrix_view block_a = matr_a.subview(i_c, p_c, mc_m, k);
        pack_block_col_major(block_a, a.data, workspace_a);

        for (size_t j_r = 0; j_r < nc_n; j_r += blocksizes.NR) {
          std::size_t n = std::min(blocksizes.NR, nc_n - j_r);
          std::size_t off_j = j_c + j_r;

          for (size_t i_r = 0; i_r < mc_m; i_r += blocksizes.MR) {
            std::size_t off_i = i_c + i_r;
            std::size_t m = std::min(blocksizes.MR, mc_m - i_r);

            kernel(m, n, k, &alpha, workspace_a, workspace_b, &beta,
                   workspace_c, 1, m, &data, cntx);
            matrix_view block_c = matr_c.subview(off_i, off_j, m, n);
            unpack(block_c, workspace_c, c.data);
          }
        }
      }
    }
  }

  alloc.deallocate(workspace_c, blocksizes.MC * blocksizes.NC);
  alloc.deallocate(workspace_b, blocksizes.KC * blocksizes.NC);
  alloc.deallocate(workspace_a, blocksizes.MC * blocksizes.KC);
}
};  // namespace t1m
