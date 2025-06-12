#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <span>
#include <vector>

namespace t1m {
namespace internal {
template <const std::size_t ndim>
std::vector<std::size_t> get_scatter(
    const std::vector<std::size_t>& indices,
    const std::array<std::size_t, ndim>& dimensions,
    const std::array<std::size_t, ndim>& strides) {
  // first, create 2D vector with each index, stride combination.
  std::vector<std::vector<std::size_t>> parts{};
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const std::size_t& idx = indices[i];

    const std::size_t& dim = dimensions[idx];
    const std::size_t& stride = strides[idx];

    std::vector<std::size_t> v(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      v[j] = j * stride;
    }
    parts.push_back(v);
  }

  // required to be equivalent with the old implementation.
  std::reverse(parts.begin(), parts.end());

  // second, reduce 2D to 1D vector, computing each possible combination.
  auto elementwise_add = [](const std::vector<std::size_t>& av,
                            const std::vector<std::size_t>& bv) {
    std::vector<std::size_t> cv;
    cv.reserve(av.size() * bv.size());
    for (const std::size_t& a : av) {
      for (const std::size_t& b : bv) {
        cv.push_back(a + b);
      }
    }
    return cv;
  };

  return std::reduce(parts.begin() + 1, parts.end(), parts[0], elementwise_add);
}

inline std::vector<std::size_t> get_block_scatter(
    const std::vector<std::size_t>& scat, const std::size_t b) {
  const size_t nblocks = (scat.size() + b - 1) / b;  // ⌈l/b⌉

  std::vector<std::size_t> block_scat(nblocks);
  for (std::size_t i = 0; i < nblocks; ++i) {
    const std::size_t offset = i * b;
    const std::size_t nelem = std::min<std::size_t>(b, scat.size() - offset);

    // Compute pairwise distances.
    const auto first = scat.begin() + offset;
    const std::size_t n = std::distance(first, first + nelem);

    if (n == 1) {
      block_scat[i] = 0;
      continue;
    }

    std::vector<std::size_t> distances(n - 1);
    for (std::size_t j = 0; j < n - 1; ++j) {
      distances[j] = first[j + 1] - first[j];
    }

    // Compare pairwise distances and set block strides.
    std::size_t stride = distances[0];
    if (!std::all_of(distances.begin(), distances.end(),
                     [stride](std::size_t i) { return i == stride; })) {
      stride = 0;
    }
    block_scat[i] = stride;
  }

  return block_scat;
}

struct block_layout {
  template <std::size_t ndim>
  block_layout(const std::array<std::size_t, ndim>& dims,
               const std::array<std::size_t, ndim>& strides,
               const std::vector<std::size_t>& row_indices,
               const std::vector<std::size_t>& col_indices,
               const std::size_t br, const std::size_t bc)
      : br(br), bc(bc) {
    rs = get_scatter<ndim>(row_indices, dims, strides);
    cs = get_scatter<ndim>(col_indices, dims, strides);
    rbs = get_block_scatter(rs, br);
    cbs = get_block_scatter(cs, bc);
  }

  std::vector<std::size_t> rs;
  std::vector<std::size_t> cs;

  std::size_t br;
  std::vector<std::size_t> rbs;
  std::size_t bc;
  std::vector<std::size_t> cbs;
};

struct matrix_view {
  std::span<const std::size_t> rs;
  std::span<const std::size_t> cs;

  std::size_t br;
  std::span<const std::size_t> rbs;

  std::size_t bc;
  std::span<const std::size_t> cbs;

  constexpr static matrix_view from_layout(const block_layout& layout) {
    return {layout.rs, layout.cs, layout.br, layout.rbs, layout.bc, layout.cbs};
  }

  constexpr matrix_view subview(std::size_t ri, std::size_t ci,
                                std::size_t nrows,
                                std::size_t ncols) const noexcept {
    const std::size_t rfrstblck = ri / br;
    const std::size_t cfrstblck = ci / bc;
    const std::size_t rnblcks = (nrows + br - 1) / br;  // ceil(nrows / br)
    const std::size_t cnblcks = (ncols + bc - 1) / bc;  // ceil(ncols / bc)
    return {rs.subspan(ri, nrows),
            cs.subspan(ci, ncols),
            br,
            rbs.subspan(rfrstblck, rnblcks),
            bc,
            cbs.subspan(cfrstblck, cnblcks)};
  }

  constexpr std::size_t nrows() const noexcept { return rs.size(); }
  constexpr std::size_t ncols() const noexcept { return cs.size(); }
  constexpr std::size_t nelems() const noexcept { return nrows() * ncols(); }
  constexpr std::size_t block_nelems() const noexcept { return br * bc; }
};
};  // namespace internal
};  // namespace t1m