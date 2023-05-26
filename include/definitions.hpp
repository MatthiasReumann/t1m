#pragma once

#include "marray.hpp"

template <class T>
using Tensor = MArray::marray_view<T>;

constexpr MArray::layout COLUMN_MAJOR = MArray::COLUMN_MAJOR;