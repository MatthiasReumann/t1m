#pragma once

#include "tensor.hpp"
#include "scatter_vector.hpp"
#include <vector>
#include "utils.hpp"

namespace t1m::internal
{
  template <typename T>
  class ScatterMatrix : public Tensor<T>
  {
  public:
    ScatterMatrix(Tensor<T>& t, std::vector<size_t> row_indices, std::vector<size_t> col_indices)
      : Tensor<T>(t),
      rscat(this->lengths(), this->strides(), row_indices),
      cscat(this->lengths(), this->strides(), col_indices) {}

    T operator() (size_t i, size_t j) const
    {
      return this->get(i, j);
    }

    T get(size_t i, size_t j) const
    {
      return this->cdata()[this->location(i, j)];
    }

    T* pointer_at_loc(size_t i, size_t j) const
    {
      return &(this->data()[this->location(i, j)]);
    }

    size_t row_size() const
    {
      return this->rscat.size();
    }

    size_t col_size() const
    {
      return this->cscat.size();
    }

    size_t location(size_t i, size_t j) const
    {
      return this->rscat.at(i) + this->cscat.at(j);
    }

    ScatterVector rscat;
    ScatterVector cscat;
  };
};