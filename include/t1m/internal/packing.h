#pragma once

#include <print>
#include <span>
#include <type_traits>
#include "t1m/internal/scatter.h"

namespace t1m {
namespace internal {

/**
 * @brief Categorizes the (un-)packing order.
 */
enum packing_label {
  A,  // Pack only. Column Major.
  B,  // Pack only. Row Major.
  C   // Unpack only.
};

namespace {
template <class T, packing_label label>
  requires std::is_floating_point_v<T>
void pack_cell(const matrix_view& cell, const T* src, T* dest) {
  const std::size_t nrows = cell.nrows();
  const std::size_t ncols = cell.ncols();

  const std::size_t rsc = cell.rbs[0];
  const std::size_t csc = cell.cbs[0];

  const bool is_dense = (rsc > 0 && csc > 0);
  const std::size_t offset = is_dense ? (cell.rs[0] + cell.cs[0]) : 0;

  if constexpr (label == A) {
    for (std::size_t l = 0; l < ncols; ++l) {
      for (std::size_t k = 0; k < nrows; ++k) {
        const std::size_t src_idx =
            is_dense ? (k * rsc + l * csc + offset) : (cell.rs[k] + cell.cs[l]);
        dest[k + l * cell.br] = src[src_idx];
      }
    }
  } else {  // B
    for (std::size_t k = 0; k < nrows; ++k) {
      for (std::size_t l = 0; l < ncols; ++l) {
        const std::size_t src_idx =
            is_dense ? (k * rsc + l * csc + offset) : (cell.rs[k] + cell.cs[l]);
        dest[l + k * cell.bc] = src[src_idx];
      }
    }
  }
}

template <class T, packing_label label>
void pack_cell(const matrix_view& cell, const T* src,
               typename T::value_type* dest) {
  const std::size_t nrows = cell.nrows();
  const std::size_t ncols = cell.ncols();

  const std::size_t rsc = cell.rbs[0];
  const std::size_t csc = cell.cbs[0];

  const bool is_dense = (rsc > 0 && csc > 0);
  const std::size_t offset = is_dense ? (cell.rs[0] + cell.cs[0]) : 0;

  if constexpr (label == A) {
    for (std::size_t l = 0; l < ncols; ++l) {
      for (std::size_t k = 0; k < nrows; ++k) {
        const std::size_t src_idx =
            is_dense ? (k * rsc + l * csc + offset) : (cell.rs[k] + cell.cs[l]);
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
        const std::size_t src_idx =
            is_dense ? (k * rsc + l * csc + offset) : (cell.rs[k] + cell.cs[l]);
        const auto& val = src[src_idx];

        const std::size_t base = l + 2 * k * cell.bc;
        dest[base] = val.real();
        dest[base + cell.bc] = val.imag();
      }
    }
  }
}
}  // namespace

template <class T, packing_label label>
  requires std::is_floating_point_v<T>
void pack_block(const matrix_view& block, const std::size_t length,
                const T* src, T* dest) {
  const std::size_t nrows = block.nrows();
  const std::size_t ncols = block.ncols();

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
}

template <class T, packing_label label>
void pack_block(const matrix_view& block, const std::size_t length,
                const T* src, typename T::value_type* dest) {
  const std::size_t nrows = block.nrows();
  const std::size_t ncols = block.ncols();
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

template <class T>
  requires std::is_floating_point_v<T>
void unpack(const matrix_view& block, const T* src, T* dest) {
  const std::size_t nrows = block.nrows();
  const std::size_t ncols = block.ncols();

  for (std::size_t l = 0; l < ncols; ++l) {
    for (std::size_t k = 0; k < nrows; ++k) {
      dest[block.rs[k] + block.cs[l]] += src[k + l * block.br];
    }
  }
}

template <class T>
void unpack(const matrix_view& block, const typename T::value_type* src,
            T* dest) {
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
      const std::size_t src_idx = 2 * k + l * block.br;
      dest[dest_idx] += T(src[src_idx], src[src_idx + 1]);
    }
  }
}
}  // namespace internal
};  // namespace t1m