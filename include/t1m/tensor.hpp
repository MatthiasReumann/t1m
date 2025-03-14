#pragma once

#include "marray.h"

namespace t1m {
template <typename T>
class Tensor : public MArray::marray_view<T> {
 public:
  Tensor(std::initializer_list<std::size_t> lengths, T* ptr)
      : MArray::marray_view<T>(lengths, ptr, MArray::COLUMN_MAJOR) {}
  Tensor(std::vector<std::size_t> lengths, T* ptr)
      : MArray::marray_view<T>(lengths, ptr, MArray::COLUMN_MAJOR) {}
};
};  // namespace t1m