#pragma once

#include <blis.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include "t1m/bli/mappings.h"
#include "t1m/internal/concepts.h"
#include "t1m/internal/packing.h"
#include "t1m/internal/scatter.h"
#include "t1m/internal/utils.h"
#include "t1m/tensor.h"

namespace t1m {

template <internal::Real T, std::size_t ndim_a, std::size_t ndim_b,
          std::size_t ndim_c>
void contract(const T alpha, const tensor<T, ndim_a>& a,
              const std::string& labels_a, const tensor<T, ndim_b>& b,
              const std::string& labels_b, const T beta, tensor<T, ndim_c>& c,
              const std::string& labels_c) {
  using namespace t1m::internal;
  using namespace t1m::bli;

  const cntx_t* cntx = bli_gks_query_cntx();

  const block_sizes& bs = get_block_sizes<T>(cntx);
  const auto [MR, NR, KP, MC, KC, NC] = bs;

  const index_bundles bundles = get_index_bundles(labels_a, labels_b, labels_c);
  const block_layout layout_a(a.dims, a.strides(), bundles.AI, bundles.AP, MR,
                              KP);
  const block_layout layout_b(b.dims, b.strides(), bundles.BP, bundles.BJ, KP,
                              NR);
  const block_layout layout_c(c.dims, c.strides(), bundles.CI, bundles.CJ, MR,
                              NR);

  const matrix_view matr_a = matrix_view::from_layout(layout_a);
  const matrix_view matr_b = matrix_view::from_layout(layout_b);
  const matrix_view matr_c = matrix_view::from_layout(layout_c);

  const std::size_t space_size_a = MC * KC;
  const std::size_t space_size_b = KC * NC;
  const std::size_t space_size_c = MR * NR;

  std::allocator<T> alloc{};
  T* space_a = alloc.allocate(space_size_a);
  T* space_b = alloc.allocate(space_size_b);
  T* space_c = alloc.allocate(space_size_c);

  auxinfo_t data;

  const std::size_t M = matr_a.nrows();
  const std::size_t K = matr_a.ncols();
  const std::size_t N = matr_b.ncols();

  for (std::size_t j_c = 0; j_c < N; j_c += NC) {
    const std::size_t nc_n = std::min(NC, N - j_c);

    for (std::size_t p_c = 0; p_c < K; p_c += KC) {
      const std::size_t k = std::min(KC, K - p_c);
      const matrix_view view_b = matr_b.subview(p_c, j_c, k, nc_n);

      std::fill(space_b, space_b + space_size_b, T(0));
      pack_block_row_major(view_b, KC, b.data, space_b);

      for (size_t i_c = 0; i_c < M; i_c += MC) {
        const std::size_t mc_m = std::min(MC, M - i_c);
        const matrix_view view_a = matr_a.subview(i_c, p_c, mc_m, k);

        std::fill(space_a, space_a + space_size_a, T(0));
        pack_block_col_major(view_a, KC, a.data, space_a);

        for (size_t j_r = 0; j_r < nc_n; j_r += NR) {
          const std::size_t n = std::min(NR, nc_n - j_r);
          const std::size_t off_j = j_c + j_r;

          const T* sliver_b = space_b + KC * j_r;

          for (size_t i_r = 0; i_r < mc_m; i_r += MR) {
            const std::size_t m = std::min(MR, mc_m - i_r);
            const std::size_t off_i = i_c + i_r;

            const T* sliver_a = space_a + i_r * KC;

            const matrix_view view_c = matr_c.subview(off_i, off_j, m, n);

            const std::size_t rsc = view_c.rbs[0];
            const std::size_t csc = view_c.cbs[0];
            if (rsc > 0 && csc > 0) {
              gemm_kernel<T>(m, n, k, &alpha, sliver_a, sliver_b, &beta,
                             &c.data[view_c.rs[0] + view_c.cs[0]], rsc, csc,
                             &data, cntx);
              continue;
            }

            std::fill(space_c, space_c + space_size_c, T(0));
            gemm_kernel<T>(m, n, k, &alpha, sliver_a, sliver_b, &beta, space_c,
                           1, MR, &data, cntx);
            unpack(view_c, space_c, c.data);
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
