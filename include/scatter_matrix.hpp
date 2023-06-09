#pragma once

#include "tensor.hpp"
#include "scatter_vector.hpp"
#include "block_scatter_vector.hpp"
#include "utils.hpp"
#include "blis.h"
#include <vector>

template <typename T>
class ScatterMatrix : public Tensor<T>
{
public:
  ScatterMatrix(Tensor<T> &t, std::vector<size_t> row_indices, std::vector<size_t> col_indices)
      : Tensor<T>(t),
        rscat(this->lengths(), this->strides(), row_indices),
        cscat(this->lengths(), this->strides(), col_indices) {}

  T get(int i, int j) {
    return this->cdata()[this->location(i, j)];
  }

  int row_size()
  {
    return this->rscat.size();
  }

  int col_size()
  {
    return this->cscat.size();
  }

  int location(int i, int j)
  {
    return this->rscat.at(i) + this->cscat.at(j);
  }

private:
  ScatterVector rscat;
  ScatterVector cscat;
};