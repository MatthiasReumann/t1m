#pragma once

#include "definitions.h"
#include "scatter_vector.h"
#include <vector>

class ScatterMatrix : public Tensor<float>
{
public:
  ScatterMatrix(Tensor<float> &t, std::vector<size_t> row_indices, std::vector<size_t> col_indices)
      : Tensor<float>(t),
        rscat(this->lengths(), this->strides(), row_indices),
        cscat(this->lengths(), this->strides(), col_indices) { }

  template <typename T, int m, int n> // m x n
  void pack_to_submatrix(T *submatrix, int off_i, int off_j)
  {
    const T *ptr = this->cdata();
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        submatrix[i + j * m] = ptr[this->location(i + off_i, j + off_j)];
      }
    }
  }

  template <typename T, int m, int n> // m x n
  void add_from_submatrix(T *submatrix, int off_i, int off_j)
  {
    T *ptr = this->data();
    for (int j = 0; j < n; j++)
    {
      for (int i = 0; i < m; i++)
      {
        ptr[this->location(i + off_i, j + off_j)] += submatrix[i + j * m];
      }
    }
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

  ScatterVector rscat;
  ScatterVector cscat;
};