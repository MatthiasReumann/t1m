#pragma once

#include <blis/blis.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "t1m/internal/blis.h"
#include "t1m/internal/packing.h"
#include "t1m/internal/scatter.h"
#include "t1m/internal/tensor.h"
#include "t1m/internal/utils.h"

namespace t1m {

template <typename T, std::size_t ndim_a, std::size_t ndim_b,
          std::size_t ndim_c>
void contract(const T alpha, const tensor<T, ndim_a>& a,
              const std::string& labels_a, const tensor<T, ndim_b>& b,
              const std::string& labels_b, const T beta, tensor<T, ndim_c>& c,
              const std::string& labels_c) {
  std::allocator<T> alloc{};

  const cntx_t* cntx = bli_gks_query_cntx();
  const bli::block_sizes blocksizes{bli::get_block_sizes<T>(cntx)};

  const contraction indices({labels_a, labels_b, labels_c});
  const block_layout layout_a(a.dims, a.strides(), indices.AI, indices.AP,
                              blocksizes.MR, blocksizes.KP);
  const block_layout layout_b(b.dims, b.strides(), indices.BP, indices.BJ,
                              blocksizes.KP, blocksizes.NR);
  const block_layout layout_c(c.dims, c.strides(), indices.CI, indices.CJ,
                              blocksizes.MR, blocksizes.NR);

  const matrix_view matr_a = matrix_view::from_layout(layout_a);
  const matrix_view matr_b = matrix_view::from_layout(layout_b);
  const matrix_view matr_c = matrix_view::from_layout(layout_c);

  const std::size_t space_size_a = blocksizes.MC * blocksizes.KC;
  const std::size_t space_size_b = blocksizes.KC * blocksizes.NC;
  const std::size_t space_size_c = blocksizes.MR * blocksizes.NR;

  T* space_a = alloc.allocate(space_size_a);
  T* space_b = alloc.allocate(space_size_b);
  T* space_c = alloc.allocate(space_size_c);

  auxinfo_t data;

  const std::size_t M = matr_a.nrows();
  const std::size_t K = matr_a.ncols();
  const std::size_t N = matr_b.ncols();

  for (std::size_t j_c = 0; j_c < N; j_c += blocksizes.NC) {
    std::size_t nc_n = std::min(blocksizes.NC, N - j_c);

    for (std::size_t p_c = 0; p_c < K; p_c += blocksizes.KC) {
      std::size_t k = std::min(blocksizes.KC, K - p_c);

      matrix_view block_b = matr_b.subview(p_c, j_c, k, nc_n);

      std::fill(space_b, space_b + space_size_b, T(0));
      pack_block_row_major(block_b, blocksizes.KC, b.data, space_b);

      for (size_t i_c = 0; i_c < M; i_c += blocksizes.MC) {
        std::size_t mc_m = std::min(blocksizes.MC, M - i_c);

        matrix_view block_a = matr_a.subview(i_c, p_c, mc_m, k);

        std::fill(space_a, space_a + space_size_a, T(0));
        pack_block_col_major(block_a, blocksizes.KC, a.data, space_a);

        for (size_t j_r = 0; j_r < nc_n; j_r += blocksizes.NR) {
          std::size_t n = std::min(blocksizes.NR, nc_n - j_r);
          std::size_t off_j = j_c + j_r;

          T* sliver_b = space_b + blocksizes.KC * j_r;
          for (size_t i_r = 0; i_r < mc_m; i_r += blocksizes.MR) {
            std::size_t off_i = i_c + i_r;
            std::size_t m = std::min(blocksizes.MR, mc_m - i_r);

            T* sliver_a = space_a + i_r * blocksizes.KC;

            matrix_view block_c = matr_c.subview(off_i, off_j, m, n);

            const std::size_t rsc = block_c.rbs[0];
            const std::size_t csc = block_c.cbs[0];
            if (rsc > 0 && csc > 0) {
              bli::gemm_kernel<T>(m, n, k, &alpha, sliver_a, sliver_b, &beta,
                                  &c.data[block_c.rs[0] + block_c.cs[0]], rsc,
                                  csc, &data, cntx);
            } else {
              std::fill(space_c, space_c + space_size_c, T(0));
              bli::gemm_kernel<T>(m, n, k, &alpha, sliver_a, sliver_b, &beta,
                                  space_c, 1, blocksizes.MR, &data, cntx);
              unpack(block_c, space_c, c.data);
            }
          }
        }
      }
    }
  }

  alloc.deallocate(space_c, space_size_c);
  alloc.deallocate(space_b, space_size_b);
  alloc.deallocate(space_a, space_size_a);
}
};  // namespace t1m
