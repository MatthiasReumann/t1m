#pragma once

#include <span>
#include "t1m/internal/concepts.h"
#include "t1m/internal/scatter.h"
#include "t1m/tensor.h"

namespace t1m {
namespace internal {

template <Real T, memory_layout layout>
constexpr void pack_cell(const matrix_view& cell, const T* src, T* dest) {
  const std::size_t nrows = cell.nrows();
  const std::size_t ncols = cell.ncols();

  const std::size_t rsc = cell.rbs[0];
  const std::size_t csc = cell.cbs[0];

  const bool is_dense = (rsc > 0 && csc > 0);
  const std::size_t offset = is_dense ? (cell.rs[0] + cell.cs[0]) : 0;

  if constexpr (layout == col_major) {
    for (std::size_t l = 0; l < ncols; ++l) {
      for (std::size_t k = 0; k < nrows; ++k) {
        const std::size_t src_idx =
            is_dense ? (k * rsc + l * csc + offset) : (cell.rs[k] + cell.cs[l]);
        dest[k + l * cell.br] = src[src_idx];
      }
    }
  } else {  // row_major
    for (std::size_t k = 0; k < nrows; ++k) {
      for (std::size_t l = 0; l < ncols; ++l) {
        const std::size_t src_idx =
            is_dense ? (k * rsc + l * csc + offset) : (cell.rs[k] + cell.cs[l]);
        dest[l + k * cell.bc] = src[src_idx];
      }
    }
  }
}

template <Real T>
void pack_block_col_major(const matrix_view& block, const std::size_t width,
                          const T* src, T* dest) {
  const std::size_t nrows = block.nrows();
  const std::size_t ncols = block.ncols();

  //       K
  //   ┌───┬───┐
  //   │   │   │
  // M ├───┼───┤ "block"
  //   │   │   │
  //   └───┴───┘

  for (std::size_t r = 0; r < nrows; r += block.br) {

    //        K
    //    ┌───┬───┐
    // br │   │   │ "sliver"
    //    └───┴───┘

    for (std::size_t c = 0; c < ncols; c += block.bc) {

      //      bc
      //    ┌────┐
      // br │    │  "cell"
      //    └────┘

      const std::size_t cell_nrows = std::min(block.br, nrows - r);
      const std::size_t cell_ncols = std::min(block.bc, ncols - c);
      const matrix_view cell = block.subview(r, c, cell_nrows, cell_ncols);
      const std::size_t offset = c * block.br + r * width;
      pack_cell<T, col_major>(cell, src, dest + offset);
    }
  }
}

template <Real T>
void pack_block_row_major(const matrix_view& block, const std::size_t height,
                          const T* src, T* dest) {
  const std::size_t nrows = block.nrows();
  const std::size_t ncols = block.ncols();

  //       K
  //   ┌───┬───┐
  //   │   │   │
  // M ├───┼───┤ "block"
  //   │   │   │
  //   └───┴───┘

  for (std::size_t c = 0; c < block.ncols(); c += block.bc) {

    //      bc
    //    ┌───┐
    //    │   │
    // M  ├───┤ "sliver"
    //    │   │
    //    └───┘

    for (std::size_t r = 0; r < block.nrows(); r += block.br) {

      //      bc
      //    ┌────┐
      // br │    │  "cell"
      //    └────┘

      const std::size_t cell_nrows = std::min(block.br, nrows - r);
      const std::size_t cell_ncols = std::min(block.bc, ncols - c);
      const matrix_view cell = block.subview(r, c, cell_nrows, cell_ncols);
      const std::size_t offset = r * block.bc + c * height;
      pack_cell<T, row_major>(cell, src, dest + offset);
    }
  }
}

template <Real T>
void unpack(const matrix_view& block, const T* src, T* dest) {
  const std::size_t nrows = block.nrows();
  const std::size_t ncols = block.ncols();

  for (std::size_t k = 0; k < nrows; ++k) {
    for (std::size_t l = 0; l < ncols; ++l) {
      dest[block.rs[k] + block.cs[l]] += src[k + l * block.br];
    }
  }
}
}  // namespace internal
};  // namespace t1m