#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <numeric>
#include <span>
#include <tuple>
#include <vector>

namespace t1m {
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

inline index_bundle_tuple get_index_bundles(const std::string& labels_a,
                                            const std::string& labels_b,
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

template <std::size_t ndim>
std::vector<std::size_t> get_scatter(
    const std::vector<std::size_t>& indices,
    const std::array<std::size_t, ndim>& dimensions,
    const std::array<std::size_t, ndim>& strides) {

  // Compute all index-stride combinations.
  std::vector<std::vector<std::size_t>> parts(indices.size());
  for (std::size_t i = 0; i < parts.size(); ++i) {
    const std::size_t idx = indices[i];
    const std::size_t dim = dimensions[idx];
    const std::size_t stride = strides[idx];

    parts[i] = [&dim, &stride] {
      std::vector<std::size_t> v(dim);
      for (std::size_t j = 0; j < dim; ++j) {
        v[j] = j * stride;
      }
      return v;
    }();
  }

  // second, reduce 2D to 1D vector, computing each possible combination.
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

inline std::vector<std::size_t> get_block_scatter(
    const std::vector<std::size_t>& scat, const std::size_t b) {
  const size_t nblocks = div_ceil(scat.size(), b);

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
  std::vector<std::size_t> rs;
  std::vector<std::size_t> cs;

  std::size_t br;
  std::vector<std::size_t> rbs;
  std::size_t bc;
  std::vector<std::size_t> cbs;

  template <std::size_t ndim>
  block_layout(const std::array<std::size_t, ndim>& dims,
               const std::array<std::size_t, ndim>& strides,
               const index_bundle& bundle, const std::size_t br,
               const std::size_t bc)
      : br(br), bc(bc) {
    rs = get_scatter<ndim>(bundle.X, dims, strides);
    cs = get_scatter<ndim>(bundle.Y, dims, strides);
    rbs = get_block_scatter(rs, br);
    cbs = get_block_scatter(cs, bc);
  }
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
    const std::size_t rnblcks = div_ceil(nrows, br);
    const std::size_t cnblcks = div_ceil(ncols, bc);
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