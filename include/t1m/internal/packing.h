#include <algorithm>
#include <span>
#include "t1m/internal/scatter.h"
#include "t1m/tensor.h"

namespace t1m {
namespace packing {

template <typename T, memory_layout layout>
void pack_cell(const scatter::matrix_view& cell, const T* src, T* dest) {
  const std::size_t nrows = cell.nrows();
  const std::size_t ncols = cell.ncols();

  const std::size_t rs = cell.rbs[0];
  const std::size_t cs = cell.cbs[0];

  if (rs > 0 && cs > 0) {
    const std::size_t offset = (cell.rs[0] + cell.cs[0]);
    if constexpr (layout == memory_layout::COL_MAJOR) {
      for (std::size_t l = 0; l < ncols; ++l) {
        for (std::size_t k = 0; k < nrows; ++k) {
          dest[k + l * nrows] = src[k * rs + l * cs + offset];
        }
      }
    } else {
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          dest[l + k * ncols] = src[k * rs + l * cs + offset];
        }
      }
    }
  } else {
    if constexpr (layout == memory_layout::COL_MAJOR) {
      for (std::size_t l = 0; l < ncols; ++l) {
        for (std::size_t k = 0; k < nrows; ++k) {
          dest[k + l * nrows] = src[cell.rs[k] + cell.cs[l]];
        }
      }
    } else {
      for (std::size_t k = 0; k < nrows; ++k) {
        for (std::size_t l = 0; l < ncols; ++l) {
          dest[l + k * ncols] = src[cell.rs[k] + cell.cs[l]];
        }
      }
    }
  }
}

template <typename T>
void pack_block_col_major(const scatter::matrix_view& block, const T* src,
                          T* dest) {

  //       K
  //   ┌───┬───┐
  //   │   │   │
  // M ├───┼───┤ "block"
  //   │   │   │
  //   └───┴───┘

  std::size_t offset = 0;
  for (std::size_t r = 0; r < block.nrows(); r += block.br) {

    //        K
    //    ┌───┬───┐
    // br │   │   │ "sliver"
    //    └───┴───┘

    for (std::size_t c = 0; c < block.ncols(); c += block.bc) {

      //      bc
      //    ┌────┐
      // br │    │  "cell"
      //    └────┘

      const auto cell =
          block.subview(r, c, std::min(block.br, block.nrows() - r),
                        std::min(block.bc, block.ncols() - c));
      pack_cell<T, COL_MAJOR>(cell, src, dest + offset);
      offset += cell.block_nelems();
    }
  }
}

template <typename T>
void pack_block_row_major(const scatter::matrix_view& block, const T* src,
                          T* dest) {

  //       K
  //   ┌───┬───┐
  //   │   │   │
  // M ├───┼───┤ "block"
  //   │   │   │
  //   └───┴───┘

  std::size_t offset = 0;

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

      const auto cell =
          block.subview(r, c, std::min(block.br, block.nrows() - r),
                        std::min(block.bc, block.ncols() - c));
      pack_cell<T, ROW_MAJOR>(cell, src, dest + offset);
      offset += cell.block_nelems();
    }
  }
}
}  // namespace packing
};  // namespace t1m