#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include "t1m/traits.h"

namespace t1m {
enum memory_layout : std::uint8_t { row_major, col_major };

template <TensorScalar T, std::size_t ndim> class tensor {
 public:
  explicit tensor(const std::array<std::size_t, ndim>& dims, T* data,
                  memory_layout layout = memory_layout::col_major)
      : data_(data), layout_{layout}, dims_(dims) {}

  [[nodiscard]] constexpr std::size_t rank() const noexcept { return ndim; }
  [[nodiscard]] constexpr std::array<std::size_t, ndim> dims() const noexcept {
    return dims_;
  }
  [[nodiscard]] constexpr std::array<std::size_t, ndim> strides() const {
    std::array<std::size_t, ndim> strides;
    switch (layout_) {
      case col_major:
        strides[0] = 1;
        for (std::size_t i = 1; i < dims_.size(); ++i) {
          strides[i] = strides[i - 1] * dims_[i - 1];
        }
        break;
      case row_major:
        throw std::logic_error("not implemented yet.");
    }
    return strides;
  }
  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] constexpr memory_layout layout() const noexcept {
    return layout_;
  }

 private:
  T* data_;
  memory_layout layout_;
  std::array<std::size_t, ndim> dims_;
};
};  // namespace t1m