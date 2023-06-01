#pragma once

#include "marray.hpp"

template <class T>
class Tensor : public MArray::marray_view<T>
{
public:
  Tensor(std::initializer_list<int> lengths, T *ptr)
      : MArray::marray_view<T>(lengths, ptr, MArray::COLUMN_MAJOR) {}
};