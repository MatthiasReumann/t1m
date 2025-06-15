#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>

namespace t1m {
enum memory_layout : std::uint8_t { row_major, col_major };

template <typename T, std::size_t ndim> class tensor {
 public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  static constexpr std::size_t rank = ndim;

  explicit tensor(const std::array<std::size_t, ndim>& dims, pointer data,
                  memory_layout layout = memory_layout::col_major)
      : data_(data), layout_{layout}, dims_(dims) {}

  // Move semantics
  tensor(tensor&&) noexcept = default;
  tensor& operator=(tensor&&) noexcept = default;

  // Copy semantics.
  tensor(const tensor&) = default;
  tensor& operator=(const tensor&) = default;

  [[nodiscard]] constexpr std::array<std::size_t, ndim> dims()
      const noexcept {
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

  [[nodiscard]] pointer data() noexcept { return data_; }
  [[nodiscard]] const_pointer data() const noexcept { return data_; }

  [[nodiscard]] constexpr memory_layout layout() const noexcept {
    return layout_;
  }

 private:
  T* data_;
  memory_layout layout_;
  std::array<std::size_t, ndim> dims_;
};
};  // namespace t1m