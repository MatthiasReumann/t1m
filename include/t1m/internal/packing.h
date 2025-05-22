#pragma once

#include <algorithm>
#include <print>
#include <span>
#include "t1m/internal/scatter.h"
#include "t1m/internal/tensor.h"

namespace t1m {

template <typename T, memory_layout layout>
void pack_cell(const matrix_view& cell, const T* src, T* dest) {
  const std::size_t nrows = cell.nrows();
  const std::size_t ncols = cell.ncols();

  const std::size_t rs = cell.rbs[0];
  const std::size_t cs = cell.cbs[0];

  if (rs > 0 && cs > 0) {
    const std::size_t offset = (cell.rs[0] + cell.cs[0]);
    if constexpr (layout == memory_layout::col_major) {
      for (std::size_t l = 0; l < ncols; ++l) {
        for (std::size_t k = 0; k < nrows; ++k) {
          dest[k + l * cell.br] = src[k * rs + l * cs + offset];
        }
      }
    } else {
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          dest[l + k * cell.bc] = src[k * rs + l * cs + offset];
        }
      }
    }
  } else {
    if constexpr (layout == memory_layout::col_major) {
      for (std::size_t l = 0; l < ncols; ++l) {
        for (std::size_t k = 0; k < nrows; ++k) {
          dest[k + l * cell.br] = src[cell.rs[k] + cell.cs[l]];
        }
      }
    } else {
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          dest[l + k * cell.bc] = src[cell.rs[k] + cell.cs[l]];
        }
      }
    }
  }
}

template <typename T>
void pack_block_col_major(const matrix_view& block, const std::size_t width, const T* src, T* dest) {
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

template <typename T>
void pack_block_row_major(const matrix_view& block, const std::size_t height, const T* src, T* dest) {
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

template <typename T>
void unpack(const matrix_view& block, const T* src, T* dest) {
  const std::size_t nrows = block.nrows();
  const std::size_t ncols = block.ncols();

  for (std::size_t k = 0; k < nrows; ++k) {
    for (std::size_t l = 0; l < ncols; ++l) {
      dest[block.rs[k] + block.cs[l]] += src[k + l * block.br];
    }
  }
}
};  // namespace t1m