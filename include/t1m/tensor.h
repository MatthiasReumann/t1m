#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>

namespace t1m {
enum memory_layout { row_major, col_major };

template <typename T, std::size_t ndim> struct tensor {

  std::array<std::size_t, ndim> dims;
  T* data;
  memory_layout layout;

  constexpr std::array<std::size_t, ndim> strides() const {
    std::array<std::size_t, ndim> strides;

    switch (layout) {
      case col_major:
        strides[0] = 1;
        for (std::size_t i = 1; i < dims.size(); ++i) {
          strides[i] = strides[i - 1] * dims[i - 1];
        }
        break;
      case row_major:
        throw std::logic_error("not implemented yet.");
    }

    return strides;
  }
};
};  // namespace t1m