#pragma once

#include <array>
#include <cstddef>
#include <span>
#include "t1m/internal/utils.h"

namespace t1m {

struct matrix_view {
  std::span<const std::size_t> rs;
  std::span<const std::size_t> cs;

  const std::size_t br;
  std::span<const std::size_t> rbs;

  const std::size_t bc;
  std::span<const std::size_t> cbs;

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
  constexpr std::size_t block_nelems() const noexcept {
    return nrows() * ncols();
  }
};

struct block_layout {
  template <const std::size_t ndim>
  block_layout(const std::array<std::size_t, ndim> dimensions,
               const std::array<std::size_t, ndim> strides,
               const std::vector<std::size_t>& row_indices,
               const std::vector<std::size_t>& col_indices,
               const std::size_t br, const std::size_t bc)
      : br(br), bc(bc) {
    rs = scatter<ndim>{}(row_indices, dimensions, strides);
    cs = scatter<ndim>{}(col_indices, dimensions, strides);
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
};  // namespace t1m