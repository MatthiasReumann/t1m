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

  std::array<std::size_t, ndim> strides() const {
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
};  // namespace t1m