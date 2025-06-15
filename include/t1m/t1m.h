#pragma once

#include <blis.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>
#include "t1m/bli/mappings.h"
#include "t1m/internal/packing.h"
#include "t1m/internal/scatter.h"
#include "t1m/tensor.h"

namespace t1m {

template <class T, std::size_t ndim_a, std::size_t ndim_b, std::size_t ndim_c,
          class Allocator = std::allocator<T>>
  requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
void contract(const T alpha, const tensor<T, ndim_a>& a,
              const std::string& labels_a, const tensor<T, ndim_b>& b,
              const std::string& labels_b, const T beta, tensor<T, ndim_c>& c,
              const std::string& labels_c,
              const cntx_t* cntx = bli_gks_query_cntx()) {
  using namespace t1m::internal;
  using namespace t1m::bli;

  const auto& [MR, NR, KP, MC, KC, NC] = get_block_sizes<T>(cntx);

  const auto [bundle_a, bundle_b, bundle_c] =
      get_index_bundles(labels_a, labels_b, labels_c);
  const matrix_layout layout_a(a, bundle_a, MR, KP);
  const matrix_layout layout_b(b, bundle_b, KP, NR);
  const matrix_layout layout_c(c, bundle_c, MR, NR);

  const matrix_view matr_a = matrix_view::from_layout(layout_a);
  const matrix_view matr_b = matrix_view::from_layout(layout_b);
  const matrix_view matr_c = matrix_view::from_layout(layout_c);

  const std::size_t space_size_a = MC * KC;
  const std::size_t space_size_b = KC * NC;
  const std::size_t space_size_c = MR * NR;
  const std::size_t space_total = space_size_a + space_size_b + space_size_c;

  Allocator alloc{};
  T* space_a = std::allocator_traits<Allocator>::allocate(alloc, space_total);
  T* space_b = space_a + space_size_a;
  T* space_c = space_b + space_size_b;

  const std::size_t M = matr_a.nrows();
  const std::size_t K = matr_a.ncols();
  const std::size_t N = matr_b.ncols();

  for (std::size_t j_c = 0; j_c < N; j_c += NC) {
    const std::size_t nc_n = std::min(NC, N - j_c);

    for (std::size_t p_c = 0; p_c < K; p_c += KC) {
      const std::size_t k = std::min(KC, K - p_c);
      const matrix_view view_b = matr_b.subview(p_c, j_c, k, nc_n);

      std::fill(space_b, space_b + space_size_b, T(0));
      pack_block<T, packing_label::B>(view_b, KC, b.data(), space_b);

      for (size_t i_c = 0; i_c < M; i_c += MC) {
        const std::size_t mc_m = std::min(MC, M - i_c);
        const matrix_view view_a = matr_a.subview(i_c, p_c, mc_m, k);

        std::fill(space_a, space_a + space_size_a, T(0));
        pack_block<T, packing_label::A>(view_a, KC, a.data(), space_a);

        for (size_t j_r = 0; j_r < nc_n; j_r += NR) {
          const std::size_t n = std::min(NR, nc_n - j_r);
          const std::size_t cci = j_c + j_r;

          const T* sliver_b = space_b + KC * j_r;

          for (size_t i_r = 0; i_r < mc_m; i_r += MR) {
            const std::size_t m = std::min(MR, mc_m - i_r);
            const std::size_t cri = i_c + i_r;

            const T* sliver_a = space_a + i_r * KC;

            const matrix_view view_c = matr_c.subview(cri, cci, m, n);

            std::fill(space_c, space_c + space_size_c, T(0));

            auxinfo_t data;
            gemm_kernel<T>(m, n, k, &alpha, sliver_a, sliver_b, &beta, space_c,
                           1, MR, &data, cntx);

            unpack(view_c, space_c, c.data());
          }
        }
      }
    }
  }

  std::allocator_traits<Allocator>::deallocate(alloc, space_a, space_total);
}

template <class T, std::size_t ndim_a, std::size_t ndim_b, std::size_t ndim_c,
          class U = typename T::value_type, class Allocator = std::allocator<U>>
  requires(std::is_same_v<T, std::complex<float>> ||
           std::is_same_v<T, std::complex<double>>)
void contract(const tensor<T, ndim_a>& a, const std::string& labels_a,
              const tensor<T, ndim_b>& b, const std::string& labels_b,
              tensor<T, ndim_c>& c, const std::string& labels_c,
              const cntx_t* cntx = bli_gks_query_cntx()) {
  using namespace t1m::internal;
  using namespace t1m::bli;

  const auto& [MR, NR, KP, MC, KC, NC] = get_block_sizes<T>(cntx);

  const auto [bundle_a, bundle_b, bundle_c] =
      get_index_bundles(labels_a, labels_b, labels_c);
  const matrix_layout layout_a(a, bundle_a, MR, KP);
  const matrix_layout layout_b(b, bundle_b, KP, NR);
  const matrix_layout layout_c(c, bundle_c, MR, NR);

  const matrix_view matr_a = matrix_view::from_layout(layout_a);
  const matrix_view matr_b = matrix_view::from_layout(layout_b);
  const matrix_view matr_c = matrix_view::from_layout(layout_c);

  const std::size_t space_size_a = MC * KC;
  const std::size_t space_size_b = KC * NC;
  const std::size_t space_size_c = MR * NR;
  const std::size_t space_total = space_size_a + space_size_b + space_size_c;

  Allocator alloc{};
  U* space_a = std::allocator_traits<Allocator>::allocate(alloc, space_total);
  U* space_b = space_a + space_size_a;
  U* space_c = space_b + space_size_b;

  const U alpha = U(1);
  const U beta = U(0);

  const std::size_t M = matr_a.nrows();
  const std::size_t K = matr_a.ncols();
  const std::size_t N = matr_b.ncols();

  for (std::size_t j_c = 0; j_c < N; j_c += NC) {
    const std::size_t nc_n = std::min(NC, N - j_c);

    for (std::size_t p_c = 0; p_c < K; p_c += KC / 2) {
      const std::size_t k = std::min(KC / 2, K - p_c);
      const dim_t k_real = k * 2;

      const matrix_view view_b = matr_b.subview(p_c, j_c, k, nc_n);

      std::fill(space_b, space_b + space_size_b, U(0));
      pack_block_1m<T, packing_label::B>(view_b, KC, b.data(), space_b);

      for (size_t i_c = 0; i_c < M; i_c += MC / 2) {
        const std::size_t mc_m = std::min(MC / 2, M - i_c);
        const std::size_t mc_m_real = 2 * mc_m;

        const matrix_view view_a = matr_a.subview(i_c, p_c, mc_m, k);

        std::fill(space_a, space_a + space_size_a, U(0));
        pack_block_1m<T, packing_label::A>(view_a, KC, a.data(), space_a);

        for (size_t j_r = 0; j_r < nc_n; j_r += NR) {
          const std::size_t n = std::min(NR, nc_n - j_r);
          const std::size_t cci = j_c + j_r;
          const U* sliver_b = space_b + KC * j_r;

          for (size_t i_r = 0; i_r < mc_m_real; i_r += MR) {
            const std::size_t m = std::min(MR, mc_m_real - i_r);
            const std::size_t cri = i_c + (i_r / 2);

            const U* sliver_a = space_a + i_r * KC;

            const matrix_view view_c = matr_c.subview(cri, cci, m / 2, n);

            std::fill(space_c, space_c + space_size_c, U(0));

            auxinfo_t data;
            gemm_kernel<T>(m, n, k_real, &alpha, sliver_a, sliver_b, &beta,
                           space_c, 1, MR, &data, cntx);

            unpack_1m(view_c, space_c, c.data());
          }
        }
      }
    }
  }

  std::allocator_traits<Allocator>::deallocate(alloc, space_a, space_total);
}
};  // namespace t1m
