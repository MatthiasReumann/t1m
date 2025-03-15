#pragma once
#include <array>
#include <cstddef>

namespace t1m {
enum memory_layout { ROW_MAJOR, COL_MAJOR };

template <typename T, const std::size_t ndim>
struct tensor {
  std::array<std::size_t, ndim> dimensions;
  T* data;
  memory_layout layout;

  constexpr std::array<std::size_t, ndim> strides() const {
    std::array<std::size_t, ndim> strides;

    switch (layout) {
      case ROW_MAJOR:
        throw std::runtime_error("not implemented yet.");
      case COL_MAJOR:
        strides[0] = 1;
        for (std::size_t i = 1; i < dimensions.size(); ++i) {
          strides[i] = strides[i - 1] * dimensions[i - 1];
        }
    }

    return strides;
  }
};

template <typename T, const std::size_t ndim>
class scatter_layout {
 public:
  template <const std::size_t nrows, const std::size_t ncols>
  scatter_layout(const tensor<T, ndim>& tensor,
                 const std::array<std::size_t, nrows>& row_indices,
                 const std::array<std::size_t, ncols>& col_indices) {
    rscat = utils::scatter{row_indices, tensor.dimensions, tensor.strides()}();
    cscat = utils::scatter{col_indices, tensor.dimensions, tensor.strides()}();
  }

 protected:
  std::vector<std::size_t> rscat;
  std::vector<std::size_t> cscat;
};

template <typename T, const std::size_t ndim>
class block_scatter_layout : public scatter_layout<T, ndim> {
 public:
  template <const std::size_t nrows, const std::size_t ncols>
  block_scatter_layout(const tensor<T, ndim>& tensor,
                       const std::array<std::size_t, nrows>& row_indices,
                       const std::array<std::size_t, ncols>& col_indices)
      : scatter_layout<T, ndim>(tensor, row_indices, col_indices) {
    block_rscat = utils::block_scatter{this->rscat}();
    block_cscat = utils::block_scatter{this->cscat}();
  }

 private:
  std::vector<std::size_t> block_rscat;
  std::vector<std::size_t> block_cscat;
};

};  // namespace t1m