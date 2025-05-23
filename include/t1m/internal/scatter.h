#pragma once

#include <array>
#include <cstddef>
#include <span>
#include "t1m/internal/utils.h"

namespace t1m {
namespace internal {
struct block_layout {
  template <std::size_t ndim>
  block_layout(const std::array<std::size_t, ndim>& dims,
               const std::array<std::size_t, ndim>& strides,
               const std::vector<std::size_t>& row_indices,
               const std::vector<std::size_t>& col_indices,
               const std::size_t br, const std::size_t bc)
      : br(br), bc(bc) {
    rs = scatter<ndim>{}(row_indices, dims, strides);
    cs = scatter<ndim>{}(col_indices, dims, strides);
    rbs = block_scatter{br}(rs);
    cbs = block_scatter{bc}(cs);
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