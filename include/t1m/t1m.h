#pragma once

#include <blis.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <new>
#include <numeric>
#include <ranges>
#include <vector>
#include "t1m/bli/mappings.h"

namespace t1m {

template <typename T>
concept TensorScalarArithmetic =
    std::is_same_v<T, float> || std::is_same_v<T, double>;

template <typename T>
concept TensorScalarCompound = std::is_same_v<T, std::complex<float>> ||
                               std::is_same_v<T, std::complex<double>>;
template <typename T>
concept TensorScalar = TensorScalarArithmetic<T> || TensorScalarCompound<T>;

enum memory_layout : std::uint8_t { row_major, col_major };

template <TensorScalar T, std::size_t ndim> class tensor {
 public:
  using dim_array = std::array<std::size_t, ndim>;
  using stride_array = std::array<std::size_t, ndim>;

  explicit tensor(const dim_array& dims, T* data,
                  memory_layout layout = col_major)
      : data_(data), layout_{layout}, dims_(dims) {}

  [[nodiscard]] constexpr std::size_t rank() const noexcept { return ndim; }
  [[nodiscard]] constexpr dim_array dims() const noexcept { return dims_; }
  [[nodiscard]] constexpr stride_array strides() const {
    stride_array strides;
    switch (layout_) {
      case col_major:
        strides[0] = 1;
        for (std::size_t i = 1; i < dims_.size(); ++i) {
          strides[i] = strides[i - 1] * dims_[i - 1];
        }
        break;
      case row_major:
        break;
    }
    return strides;
  }
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr memory_layout layout() const noexcept {
    return layout_;
  }

 private:
  T* data_;
  memory_layout layout_;
  std::array<std::size_t, ndim> dims_;
};

namespace internal {

/**
 * @brief Calculate ceil(x / y).
 */
constexpr std::size_t div_ceil(std::size_t x, std::size_t y) {
  return x / y + (x % y > 0);
}

/**
 * @brief Row (X) and Column (Y) indices for tensor labels.
 */
struct index_bundle {
  std::vector<std::size_t> X;
  std::vector<std::size_t> Y;
};

using index_bundle_tuple = std::tuple<index_bundle, index_bundle, index_bundle>;

/**
 * @brief Compute index bundles for each of the labels.
 * @details An index bundle splits the positions (or indices) of a label into 
 *          a row and column set. Effectively, making the tensor a matrix.
 */
[[nodiscard]] inline index_bundle_tuple get_index_bundles(
    const std::string& labels_a, const std::string& labels_b,
    const std::string& labels_c) {

  // Copy for sorting.
  std::string a(labels_a);
  std::string b(labels_b);
  std::string c(labels_c);

  // Sort for set operations.
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::sort(c.begin(), c.end());

  // Apply set operations.
  std::string contracted, free_a, free_b;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                        std::back_inserter(contracted));
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::back_inserter(free_a));
  std::set_difference(b.begin(), b.end(), a.begin(), a.end(),
                      std::back_inserter(free_b));

  // Assign bundles.
  std::vector<std::size_t> AY(contracted.size());
  std::vector<std::size_t> BX(contracted.size());
  for (size_t i = 0; i < contracted.size(); ++i) {
    AY[i] = labels_a.find(contracted[i]);
    BX[i] = labels_b.find(contracted[i]);
  }

  std::vector<std::size_t> AX(free_a.size());
  std::vector<std::size_t> CX(free_a.size());
  for (size_t i = 0; i < free_a.size(); ++i) {
    AX[i] = labels_a.find(free_a[i]);
    CX[i] = labels_c.find(free_a[i]);
  }

  std::vector<std::size_t> BY(free_b.size());
  std::vector<std::size_t> CY(free_b.size());
  for (size_t i = 0; i < free_b.size(); ++i) {
    BY[i] = labels_b.find(free_b[i]);
    CY[i] = labels_c.find(free_b[i]);
  }

  return index_bundle_tuple{{AX, AY}, {BX, BY}, {CX, CY}};
}

/**
 * @brief Compute scatter vector.
 */
template <std::size_t ndim>
[[nodiscard]] std::vector<std::size_t> get_scatter(
    std::span<const std::size_t> indices,
    std::span<const std::size_t, ndim> dimensions,
    std::span<const std::size_t, ndim> strides) {

  // Compute all index-stride combinations.
  std::vector<std::vector<std::size_t>> parts(indices.size());
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const std::size_t idx = indices[i];
    const std::size_t dim = dimensions[idx];
    const std::size_t stride = strides[idx];

    std::vector<std::size_t> v(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      v[j] = j * stride;
    }
    parts[i] = std::move(v);
  }

  // Compute all possible combinations and reduce.
  auto f = [](const std::vector<std::size_t>& av,
              const std::vector<std::size_t>& bv) {
    std::vector<std::size_t> cv(av.size() * bv.size());
    for (std::size_t a = 0; a < av.size(); ++a) {
      for (std::size_t b = 0; b < bv.size(); ++b) {
        cv[b + a * bv.size()] = av[a] + bv[b];
      }
    }
    return cv;
  };

  return std::reduce(std::next(parts.rbegin()), parts.rend(), parts.back(), f);
}

/**
 * @brief Compute block scatter vector.
 */
[[nodiscard]] inline std::vector<std::size_t> get_block_scatter(
    const std::span<const std::size_t> scat, const std::size_t b) {
  std::vector<std::size_t> block_scat;

  const std::size_t sz = scat.size();
  for (std::size_t i = 0; i < sz; i += b) {
    const std::size_t block_sz = std::min<std::size_t>(b, sz - i);
    const std::span block = scat.subspan(i, block_sz);

    if (block.size() == 1) {
      block_scat.push_back(0);
      continue;
    }

    std::vector<std::size_t> dist(block_sz);
    std::adjacent_difference(block.begin(), block.end(), dist.begin());
    const std::size_t s =
        std::ranges::adjacent_find(dist | std::views::drop(1),
                                   std::ranges::not_equal_to()) == dist.end()
            ? dist.back()
            : 0;

    block_scat.push_back(s);
  }

  return block_scat;
}

/**
 * @brief (Sub-)View of a matrix layout.
 */
struct matrix_view {
  std::span<const std::size_t> rs;
  std::span<const std::size_t> cs;

  std::size_t br;
  std::span<const std::size_t> rbs;
  std::size_t bc;
  std::span<const std::size_t> cbs;

  constexpr matrix_view subview(std::size_t ri, std::size_t ci,
                                std::size_t nrows,
                                std::size_t ncols) const noexcept {
    matrix_view view(*this);
    view.slice_rows(ri, nrows);
    view.slice_cols(ci, ncols);
    return view;
  }
  constexpr void slice_rows(const std::size_t offset, const std::size_t n) {
    const std::size_t boffset = offset / br;
    const std::size_t nblocks = div_ceil(n, br);
    rs = rs.subspan(offset, n);
    rbs = rbs.subspan(boffset, nblocks);
  }
  constexpr void slice_cols(const std::size_t offset, const std::size_t n) {
    const std::size_t boffset = offset / bc;
    const std::size_t nblocks = div_ceil(n, bc);
    cs = cs.subspan(offset, n);
    cbs = cbs.subspan(boffset, nblocks);
  }
  constexpr std::size_t nrows() const noexcept { return rs.size(); }
  constexpr std::size_t ncols() const noexcept { return cs.size(); }
  constexpr std::size_t nelems() const noexcept { return nrows() * ncols(); }
};

/**
 * @brief Matricize tensor to matrix layout.
 */
struct matrix_layout {
  template <class T, std::size_t ndim>
  matrix_layout(const tensor<T, ndim>& t, const index_bundle& bundle,
                const std::size_t br, const std::size_t bc)
      : br(br), bc(bc) {
    rs = get_scatter<ndim>(bundle.X, t.dims(), t.strides());
    cs = get_scatter<ndim>(bundle.Y, t.dims(), t.strides());
    rbs = get_block_scatter(rs, br);
    cbs = get_block_scatter(cs, bc);
  }

  constexpr matrix_view to_view() const noexcept {
    return {rs, cs, br, rbs, bc, cbs};
  }

 private:
  std::vector<std::size_t> rs;
  std::vector<std::size_t> cs;

  std::size_t br;
  std::vector<std::size_t> rbs;
  std::size_t bc;
  std::vector<std::size_t> cbs;
};

/**
 * @brief Categorizes the (un-)packing order.
 */
enum packing_label {
  A,  // Pack only. Column Major.
  B,  // Pack only. Row Major.
  C   // Unpack only.
};

template <TensorScalar T, packing_label label,
          typename U = std::conditional_t<TensorScalarCompound<T>,
                                          typename T::value_type, T>>
void pack_cell(const matrix_view& cell, const T* src, U* dest) {
  const std::size_t nrows = cell.nrows();
  const std::size_t ncols = cell.ncols();

  const std::size_t rsc = cell.rbs[0];
  const std::size_t csc = cell.cbs[0];

  const bool is_dense = (rsc > 0 && csc > 0);
  const std::size_t offset = is_dense ? (cell.rs[0] + cell.cs[0]) : 0;

  if constexpr (TensorScalarArithmetic<T>) {
    if constexpr (label == A) {
      for (std::size_t l = 0; l < ncols; ++l) {
        for (std::size_t k = 0; k < nrows; ++k) {
          const std::size_t src_idx = is_dense ? (k * rsc + l * csc + offset)
                                               : (cell.rs[k] + cell.cs[l]);
          dest[k + l * cell.br] = src[src_idx];
        }
      }
    } else {  // B
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          const std::size_t src_idx = is_dense ? (k * rsc + l * csc + offset)
                                               : (cell.rs[k] + cell.cs[l]);
          dest[l + k * cell.bc] = src[src_idx];
        }
      }
    }
  } else {  // Use 1M Packing Format.
    if constexpr (label == A) {
      for (std::size_t l = 0; l < ncols; ++l) {
        for (std::size_t k = 0; k < nrows; ++k) {
          const std::size_t src_idx = is_dense ? (k * rsc + l * csc + offset)
                                               : (cell.rs[k] + cell.cs[l]);
          const auto& val = src[src_idx];
          const auto real = val.real();
          const auto imag = val.imag();

          const std::size_t base = 2 * k + 2 * l * cell.br;
          dest[base] = real;
          dest[base + 1] = imag;
          dest[base + cell.br] = -imag;
          dest[base + cell.br + 1] = real;
        }
      }
    } else {  // B
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          const std::size_t src_idx = is_dense ? (k * rsc + l * csc + offset)
                                               : (cell.rs[k] + cell.cs[l]);
          const auto& val = src[src_idx];
          const std::size_t base = l + 2 * k * cell.bc;
          dest[base] = val.real();
          dest[base + cell.bc] = val.imag();
        }
      }
    }
  }
}

template <TensorScalar T, packing_label label,
          typename U = std::conditional_t<TensorScalarCompound<T>,
                                          typename T::value_type, T>>
void pack_block(const matrix_view& block, const std::size_t length,
                const T* src, U* dest) {
  const std::size_t nrows = block.nrows();
  const std::size_t ncols = block.ncols();

  if constexpr (TensorScalarArithmetic<T>) {
    if constexpr (label == A) {
      for (std::size_t r = 0; r < nrows; r += block.br) {
        for (std::size_t c = 0; c < ncols; c += block.bc) {
          const std::size_t offset = c * block.br + r * length;

          const std::size_t cell_nrows = std::min(block.br, nrows - r);
          const std::size_t cell_ncols = std::min(block.bc, ncols - c);
          const matrix_view cell = block.subview(r, c, cell_nrows, cell_ncols);

          pack_cell<T, label>(cell, src, dest + offset);
        }
      }
    } else {  // B
      for (std::size_t c = 0; c < ncols; c += block.bc) {
        for (std::size_t r = 0; r < nrows; r += block.br) {
          const std::size_t offset = r * block.bc + c * length;

          const std::size_t cell_nrows = std::min(block.br, nrows - r);
          const std::size_t cell_ncols = std::min(block.bc, ncols - c);
          const matrix_view cell = block.subview(r, c, cell_nrows, cell_ncols);

          pack_cell<T, label>(cell, src, dest + offset);
        }
      }
    }
  } else {  // Use 1M Packing Format.
    const std::size_t half_br = block.br / 2;
    const std::size_t half_bc = block.bc / 2;

    if constexpr (label == A) {
      for (std::size_t r = 0; r < nrows; r += half_br) {
        for (std::size_t c = 0; c < ncols; c += half_bc) {
          const std::size_t offset = 2 * (c * block.br + r * length);

          const std::size_t cell_nrows = std::min(half_br, nrows - r);
          const std::size_t cell_ncols = std::min(half_bc, ncols - c);
          const matrix_view cell = block.subview(r, c, cell_nrows, cell_ncols);

          pack_cell<T, label>(cell, src, dest + offset);
        }
      }
    } else {  // B
      for (std::size_t c = 0; c < ncols; c += block.bc) {
        for (std::size_t r = 0; r < nrows; r += half_br) {
          const std::size_t offset = 2 * r * block.bc + c * length;

          const std::size_t cell_nrows = std::min(half_br, nrows - r);
          const std::size_t cell_ncols = std::min(block.bc, ncols - c);
          const matrix_view cell = block.subview(r, c, cell_nrows, cell_ncols);

          pack_cell<T, label>(cell, src, dest + offset);
        }
      }
    }
  }
}

template <TensorScalar T,
          typename U = std::conditional_t<TensorScalarCompound<T>,
                                          typename T::value_type, T>>
void unpack(const matrix_view& block, const U* src, T* dest) {
  const std::size_t nrows = block.nrows();
  const std::size_t ncols = block.ncols();

  const std::size_t rsc = block.rbs[0];
  const std::size_t csc = block.cbs[0];

  const bool is_dense = (rsc > 0 && csc > 0);
  const std::size_t offset = is_dense ? (block.rs[0] + block.cs[0]) : 0;

  for (std::size_t l = 0; l < ncols; ++l) {
    for (std::size_t k = 0; k < nrows; ++k) {
      const std::size_t dest_idx =
          is_dense ? (k * rsc + l * csc + offset) : (block.rs[k] + block.cs[l]);

      if constexpr (TensorScalarArithmetic<T>) {
        dest[dest_idx] += src[k + l * block.br];
      } else {  // Use 1M Packing Format.
        const std::size_t src_idx = 2 * k + l * block.br;
        dest[dest_idx] += T(src[src_idx], src[src_idx + 1]);
      }
    }
  }
}
};  // namespace internal

template <TensorScalarArithmetic T, std::size_t ndim_a, std::size_t ndim_b,
          std::size_t ndim_c>
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

  const matrix_view matr_a = layout_a.to_view();
  const matrix_view matr_b = layout_b.to_view();
  const matrix_view matr_c = layout_c.to_view();

  const std::size_t space_size_a = MC * KC;
  const std::size_t space_size_b = KC * NC;
  const std::size_t space_size_c = MR * NR;
  const std::size_t space_total = space_size_a + space_size_b + space_size_c;

  T* space_a = static_cast<T*>(std::aligned_alloc(64, space_total * sizeof(T)));
  if (!space_a) {
    throw std::bad_alloc();
  }
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

  std::free(space_a);
}

template <TensorScalarCompound T, std::size_t ndim_a, std::size_t ndim_b,
          std::size_t ndim_c, class U = typename T::value_type>
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

  const matrix_view matr_a = layout_a.to_view();
  const matrix_view matr_b = layout_b.to_view();
  const matrix_view matr_c = layout_c.to_view();

  const std::size_t space_size_a = MC * KC;
  const std::size_t space_size_b = KC * NC;
  const std::size_t space_size_c = MR * NR;
  const std::size_t space_total = space_size_a + space_size_b + space_size_c;

  U* space_a = static_cast<U*>(std::aligned_alloc(64, space_total * sizeof(U)));
  if (!space_a) {
    throw std::bad_alloc();
  }
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
      pack_block<T, packing_label::B>(view_b, KC, b.data(), space_b);

      for (size_t i_c = 0; i_c < M; i_c += MC / 2) {
        const std::size_t mc_m = std::min(MC / 2, M - i_c);
        const std::size_t mc_m_real = 2 * mc_m;

        const matrix_view view_a = matr_a.subview(i_c, p_c, mc_m, k);

        std::fill(space_a, space_a + space_size_a, U(0));
        pack_block<T, packing_label::A>(view_a, KC, a.data(), space_a);

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

            unpack(view_c, space_c, c.data());
          }
        }
      }
    }
  }

  std::free(space_a);
}
};  // namespace t1m
