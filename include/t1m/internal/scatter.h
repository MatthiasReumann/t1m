#pragma once
#include <array>
#include <cstddef>
#include "t1m/internal/utils.h"
#include "t1m/tensor.h"

namespace t1m::scatter {

template <typename T, const std::size_t ndim>
class layout {
 public:
  layout(const tensor<T, ndim>& t, const std::vector<std::size_t>& row_indices,
         const std::vector<std::size_t>& col_indices)
      : rscat(utils::scatter<ndim>{}(row_indices, t.dimensions, t.strides())),
        cscat(utils::scatter<ndim>{}(col_indices, t.dimensions, t.strides())) {}

  std::vector<std::size_t> rows() const { return rscat; }
  std::vector<std::size_t> cols() const { return cscat; }
  std::size_t nrows() const noexcept { return rscat.size(); }
  std::size_t ncols() const noexcept { return cscat.size(); }

  std::size_t operator()(std::size_t row, std::size_t col) const noexcept {
    return rscat[row] + cscat[col];
  }

 protected:
  std::vector<std::size_t> rscat;
  std::vector<std::size_t> cscat;
};

template <typename T, const std::size_t ndim>
class block_layout : public layout<T, ndim> {
 public:
  block_layout(const tensor<T, ndim>& t,
               const std::vector<std::size_t>& row_indices,
               const std::vector<std::size_t>& col_indices,
               const std::size_t row_size, const std::size_t col_size)
      : layout<T, ndim>(t, row_indices, col_indices),
        row_size(row_size),
        col_size(col_size),
        block_rscat(utils::block_scatter{row_size}(this->rscat)),
        block_cscat(utils::block_scatter{col_size}(this->cscat)) {}

  /// @brief Return the number of elements in a block.
  std::size_t nelem() const noexcept { return row_size * col_size; }

  std::vector<std::size_t> block_rscat;
  std::vector<std::size_t> block_cscat;
  std::size_t row_size;
  std::size_t col_size;
};

};  // namespace t1m::scatter